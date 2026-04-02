"""Training and preprocessing pipeline for plant validation and disease detection."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from model.constants import (
    ARTIFACTS_DIR,
    CLASS_NAMES,
    DATA_DIR,
    DEFAULT_ARCHITECTURE,
    DISEASE_CHECKPOINT,
    DISEASE_DATA_DIR,
    DISEASE_METRICS_PATH,
    DISPLAY_NAMES,
    LEGACY_TRAIN_DIR,
    LEGACY_VAL_DIR,
    PREPROCESS_SUMMARY_PATH,
    RAW_DATA_DIR,
    RAW_VALIDATION_DIR,
    RAW_WHEAT_DATA_DIR,
    SUPPORTED_ARCHITECTURES,
    VALIDATION_CHECKPOINT,
    VALIDATION_CLASS_NAMES,
    VALIDATION_DATA_DIR,
    VALIDATION_METRICS_PATH,
)
from model.dataset import (
    IMAGE_EXTENSIONS,
    ImageClassificationDataset,
    LegacyFlatDiseaseDataset,
    build_train_transform,
    build_val_transform,
)
from model.losses import FocalLoss
from model.metrics import calculate_classification_metrics
from model.network import build_classifier

SPLITS = ("train", "val", "test")
STAGE_ALIASES = {
    "validation": "validation",
    "plant": "validation",
    "gate": "validation",
    "disease": "disease",
    "stage2": "disease",
    "all": "all",
}


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15


@dataclass(frozen=True)
class StageConfig:
    stage: str
    title: str
    class_names: list[str]
    data_dir: Path
    output: Path
    metrics_output: Path
    supports_legacy_flat: bool = False


def normalize_stage_name(stage: str) -> str:
    try:
        return STAGE_ALIASES[stage]
    except KeyError as exc:
        valid = ", ".join(sorted(STAGE_ALIASES))
        raise ValueError(f"Unsupported stage '{stage}'. Expected one of: {valid}") from exc


def get_stage_config(stage: str) -> StageConfig:
    normalized = normalize_stage_name(stage)
    if normalized == "validation":
        return StageConfig(
            stage="validation",
            title="Plant vs Non-Plant Validation",
            class_names=list(VALIDATION_CLASS_NAMES),
            data_dir=VALIDATION_DATA_DIR,
            output=VALIDATION_CHECKPOINT,
            metrics_output=VALIDATION_METRICS_PATH,
        )
    if normalized == "disease":
        return StageConfig(
            stage="disease",
            title="Healthy and Disease Classification",
            class_names=list(CLASS_NAMES),
            data_dir=DISEASE_DATA_DIR,
            output=DISEASE_CHECKPOINT,
            metrics_output=DISEASE_METRICS_PATH,
            supports_legacy_flat=True,
        )
    raise ValueError(f"Unsupported stage: {stage}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare datasets and train plant validation / disease classifiers."
    )
    parser.add_argument(
        "--stage",
        choices=sorted(STAGE_ALIASES),
        default="validation",
        help="Model stage to train. 'gate' and 'stage2' are accepted as aliases.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--loss", choices=["weighted_ce", "focal"], default="weighted_ce")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--architecture",
        choices=SUPPORTED_ARCHITECTURES,
        default=DEFAULT_ARCHITECTURE,
        help="Transfer-learning backbone to fine-tune.",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Create processed train/val/test splits from discovered raw data first.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Only run preprocessing, then exit.")
    parser.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--force-prepare", action="store_true", help="Rebuild processed datasets even if they already exist.")
    parser.add_argument(
        "--plant-source",
        action="append",
        type=Path,
        default=None,
        help="Optional directory containing plant images for the validation model. Can be passed multiple times.",
    )
    parser.add_argument(
        "--non-plant-source",
        action="append",
        type=Path,
        default=None,
        help="Optional directory containing non-plant images for the validation model. Can be passed multiple times.",
    )
    parser.add_argument(
        "--disease-source",
        action="append",
        type=Path,
        default=None,
        help="Optional directory containing the wheat disease dataset. Can be passed multiple times.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path

        key = str(resolved).lower()
        if key in seen or not resolved.exists():
            continue
        seen.add(key)
        deduped.append(resolved)
    return deduped


def _collect_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _collect_source_images(source_roots: Sequence[Path]) -> list[Path]:
    images: list[Path] = []
    seen: set[str] = set()

    for source_root in _dedupe_paths(source_roots):
        if source_root.is_file():
            key = str(source_root).lower()
            if source_root.suffix.lower() in IMAGE_EXTENSIONS and key not in seen:
                seen.add(key)
                images.append(source_root)
            continue

        for image_path in _collect_images(source_root):
            key = str(image_path).lower()
            if key in seen:
                continue
            seen.add(key)
            images.append(image_path)

    return images


def _resolve_source_dir(
    preferred_dir: Path,
    fallback_dirs: Sequence[Path],
    required_classes: Sequence[str],
) -> Path:
    for candidate in (preferred_dir, *fallback_dirs):
        if candidate.exists() and all((candidate / class_name).exists() for class_name in required_classes):
            return candidate

    searched = ", ".join(str(path) for path in (preferred_dir, *fallback_dirs))
    raise FileNotFoundError(
        f"Could not find a raw dataset folder with classes {list(required_classes)}. "
        f"Searched: {searched}"
    )


def _output_has_images(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in output_dir.rglob("*")
    )


def _split_counts(total: int, ratios: SplitRatios) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        return 1, 1, 0
    if total == 3:
        return 1, 1, 1

    val_count = max(1, int(round(total * ratios.val)))
    test_count = max(1, int(round(total * ratios.test)))
    train_count = total - val_count - test_count

    while train_count < 1 and (val_count > 1 or test_count > 1):
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        train_count = total - val_count - test_count

    if train_count < 1:
        train_count = 1
        overflow = val_count + test_count + train_count - total
        if overflow > 0 and test_count >= overflow:
            test_count -= overflow
        elif overflow > 0:
            val_count = max(0, val_count - overflow)

    return train_count, val_count, test_count


def _split_paths(paths: list[Path], seed: int, ratios: SplitRatios) -> dict[str, list[Path]]:
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    train_count, val_count, test_count = _split_counts(len(shuffled), ratios)
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count : train_count + val_count + test_count],
    }


def _safe_reset_output_dir(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    processed_root = DATA_DIR.resolve()
    if processed_root not in output_dir.parents and output_dir != processed_root:
        raise ValueError(f"Refusing to delete data outside {processed_root}: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def _copy_grouped_split_files(
    grouped_paths: dict[str, list[Path]],
    output_dir: Path,
    class_names: Sequence[str],
    seed: int,
    ratios: SplitRatios,
) -> dict[str, dict[str, int]]:
    counts = {split: {class_name: 0 for class_name in class_names} for split in SPLITS}

    for split in SPLITS:
        for class_name in class_names:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    for class_name in class_names:
        split_map = _split_paths(grouped_paths.get(class_name, []), seed=seed, ratios=ratios)
        for split, split_paths in split_map.items():
            target_dir = output_dir / split / class_name
            for image_path in split_paths:
                destination = target_dir / image_path.name
                if destination.exists():
                    destination = (
                        target_dir
                        / f"{image_path.stem}_{abs(hash(str(image_path))) % 100000}{image_path.suffix.lower()}"
                    )
                shutil.copy2(image_path, destination)
                counts[split][class_name] += 1

    return counts


def _default_validation_source_candidates(raw_data_dir: Path) -> dict[str, list[Path]]:
    home = Path.home()
    return {
        "plant": [
            raw_data_dir / "validation" / "plant",
            RAW_VALIDATION_DIR / "plant",
            raw_data_dir / "gate" / "wheat",
            raw_data_dir / "gate" / "other_plant",
            RAW_WHEAT_DATA_DIR,
            raw_data_dir / "wheat_disease",
            raw_data_dir / "wheat",
            home / "OneDrive" / "Desktop" / "Kaggle_Downloads" / "Guava Fruits_Dataset" / "Guava_leaf",
            home / "Downloads" / "archive" / "CNN256" / "CNN256" / "train",
            home / "Downloads" / "archive" / "CNN256" / "CNN256" / "valid",
            home / "Downloads" / "archive" / "CNN256" / "CNN256" / "test",
            home / "Downloads" / "archive" / "CNN640" / "CNN640" / "train",
            home / "Downloads" / "archive" / "CNN640" / "CNN640" / "valid",
            home / "Downloads" / "archive" / "CNN640" / "CNN640" / "test",
        ],
        "non_plant": [
            raw_data_dir / "validation" / "non_plant",
            RAW_VALIDATION_DIR / "non_plant",
            DATA_DIR / "other" / "human detection dataset",
        ],
    }


def prepare_validation_dataset(
    raw_data_dir: Path = RAW_DATA_DIR,
    output_dir: Path = VALIDATION_DATA_DIR,
    *,
    plant_sources: Sequence[Path] | None = None,
    non_plant_sources: Sequence[Path] | None = None,
    seed: int = 42,
    force: bool = False,
    ratios: SplitRatios = SplitRatios(),
) -> dict[str, object]:
    if _output_has_images(output_dir) and not force:
        return {
            "dataset": "validation",
            "status": "skipped",
            "output_dir": str(output_dir),
            "reason": "Processed files already exist. Use --force-prepare to rebuild them.",
        }

    source_candidates = _default_validation_source_candidates(raw_data_dir)
    if plant_sources:
        source_candidates["plant"] = list(plant_sources)
    if non_plant_sources:
        source_candidates["non_plant"] = list(non_plant_sources)

    grouped_paths = {
        class_name: _collect_source_images(source_candidates[class_name])
        for class_name in VALIDATION_CLASS_NAMES
    }
    source_report = {
        class_name: {
            "sources": [str(path) for path in _dedupe_paths(source_candidates[class_name])],
            "image_count": len(grouped_paths[class_name]),
        }
        for class_name in VALIDATION_CLASS_NAMES
    }

    missing_classes = [class_name for class_name in VALIDATION_CLASS_NAMES if not grouped_paths[class_name]]
    if missing_classes:
        return {
            "dataset": "validation",
            "status": "missing_raw",
            "output_dir": str(output_dir),
            "sources": source_report,
            "reason": f"Missing source images for validation classes: {missing_classes}.",
        }

    if force:
        _safe_reset_output_dir(output_dir)

    counts = _copy_grouped_split_files(
        grouped_paths=grouped_paths,
        output_dir=output_dir,
        class_names=VALIDATION_CLASS_NAMES,
        seed=seed,
        ratios=ratios,
    )

    return {
        "dataset": "validation",
        "status": "prepared",
        "output_dir": str(output_dir),
        "sources": source_report,
        "counts": counts,
    }


def prepare_disease_dataset(
    raw_data_dir: Path = RAW_DATA_DIR,
    output_dir: Path = DISEASE_DATA_DIR,
    *,
    disease_source_dirs: Sequence[Path] | None = None,
    seed: int = 42,
    force: bool = False,
    ratios: SplitRatios = SplitRatios(),
) -> dict[str, object]:
    source_candidates = _dedupe_paths(
        list(disease_source_dirs)
        if disease_source_dirs
        else [
            raw_data_dir / "wheat_disease",
            raw_data_dir / "wheat",
            RAW_WHEAT_DATA_DIR,
        ]
    )

    try:
        if not source_candidates:
            raise FileNotFoundError(
                f"Could not find a disease dataset folder with classes {list(CLASS_NAMES)}."
            )
        source_dir = _resolve_source_dir(
            source_candidates[0],
            source_candidates[1:],
            CLASS_NAMES,
        )
    except FileNotFoundError as exc:
        if _output_has_images(output_dir) and not force:
            return {
                "dataset": "disease",
                "status": "skipped",
                "output_dir": str(output_dir),
                "reason": "Processed files already exist and raw disease data was not found.",
            }
        return {
            "dataset": "disease",
            "status": "missing_raw",
            "output_dir": str(output_dir),
            "sources": [str(path) for path in source_candidates],
            "reason": str(exc),
        }

    if _output_has_images(output_dir) and not force:
        return {
            "dataset": "disease",
            "status": "skipped",
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "reason": "Processed files already exist. Use --force-prepare to rebuild them.",
        }

    if force:
        _safe_reset_output_dir(output_dir)

    counts = _copy_grouped_split_files(
        grouped_paths={
            class_name: _collect_images(source_dir / class_name)
            for class_name in CLASS_NAMES
        },
        output_dir=output_dir,
        class_names=CLASS_NAMES,
        seed=seed,
        ratios=ratios,
    )

    return {
        "dataset": "disease",
        "status": "prepared",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "counts": counts,
    }


def prepare_processed_datasets(
    raw_data_dir: Path = RAW_DATA_DIR,
    *,
    plant_sources: Sequence[Path] | None = None,
    non_plant_sources: Sequence[Path] | None = None,
    disease_source_dirs: Sequence[Path] | None = None,
    seed: int = 42,
    force: bool = False,
    ratios: SplitRatios = SplitRatios(),
    summary_path: Path = PREPROCESS_SUMMARY_PATH,
) -> dict[str, object]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "raw_data_dir": str(raw_data_dir),
        "seed": seed,
        "ratios": {
            "train": ratios.train,
            "val": ratios.val,
            "test": ratios.test,
        },
        "datasets": [
            prepare_validation_dataset(
                raw_data_dir=raw_data_dir,
                plant_sources=plant_sources,
                non_plant_sources=non_plant_sources,
                seed=seed,
                force=force,
                ratios=ratios,
            ),
            prepare_disease_dataset(
                raw_data_dir=raw_data_dir,
                disease_source_dirs=disease_source_dirs,
                seed=seed,
                force=force,
                ratios=ratios,
            ),
        ],
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_sampler(labels: list[int], class_weights: torch.Tensor) -> WeightedRandomSampler:
    sample_weights = [float(class_weights[label]) for label in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def set_backbone_trainable(model: nn.Module, architecture: str, trainable: bool) -> None:
    head_prefixes = {
        "mobilenet_v2": ("classifier.",),
        "resnet18": ("fc.",),
        "resnet34": ("fc.",),
    }
    prefixes = head_prefixes[architecture]

    for name, parameter in model.named_parameters():
        is_head = any(name.startswith(prefix) for prefix in prefixes)
        parameter.requires_grad = trainable or is_head


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    model.train()
    running_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(preds.cpu().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, all_targets, all_predictions


@torch.inference_mode()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    model.eval()
    running_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(preds.cpu().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, all_targets, all_predictions


def create_loss(loss_name: str, class_weights: torch.Tensor, device: torch.device) -> nn.Module:
    class_weights = class_weights.to(device)
    if loss_name == "focal":
        return FocalLoss(alpha=class_weights, gamma=2.0)
    return nn.CrossEntropyLoss(weight=class_weights)


def _is_stage_layout(data_dir: Path, class_names: Sequence[str]) -> bool:
    if not (data_dir / "train").exists() or not (data_dir / "val").exists():
        return False
    return all((data_dir / "train" / class_name).exists() for class_name in class_names)


def _is_flat_dataset_root(data_dir: Path) -> bool:
    train_root = data_dir / "train"
    val_root = data_dir / "val"
    if not train_root.exists() or not val_root.exists():
        return False

    required = set(CLASS_NAMES)
    train_dirs = {path.name.lower() for path in train_root.iterdir() if path.is_dir()}
    val_dirs = {path.name.lower() for path in val_root.iterdir() if path.is_dir()}
    return required.issubset(train_dirs) and required.issubset(val_dirs)


def build_stage_datasets(stage_config: StageConfig, explicit_data_dir: Path | None):
    train_transform = build_train_transform()
    val_transform = build_val_transform()

    candidate_roots: list[Path] = []
    if explicit_data_dir is not None:
        candidate_roots.extend([explicit_data_dir, explicit_data_dir / stage_config.stage])

    candidate_roots.append(stage_config.data_dir)

    if stage_config.supports_legacy_flat:
        if explicit_data_dir is not None and _is_flat_dataset_root(explicit_data_dir):
            return (
                LegacyFlatDiseaseDataset(explicit_data_dir / "train", transform=train_transform),
                LegacyFlatDiseaseDataset(explicit_data_dir / "val", transform=val_transform),
                explicit_data_dir,
                "legacy-flat",
            )
        if _is_flat_dataset_root(DATA_DIR):
            return (
                LegacyFlatDiseaseDataset(LEGACY_TRAIN_DIR, transform=train_transform),
                LegacyFlatDiseaseDataset(LEGACY_VAL_DIR, transform=val_transform),
                DATA_DIR,
                "legacy-flat",
            )

    for candidate_root in candidate_roots:
        if _is_stage_layout(candidate_root, stage_config.class_names):
            return (
                ImageClassificationDataset(
                    candidate_root / "train",
                    class_names=stage_config.class_names,
                    transform=train_transform,
                ),
                ImageClassificationDataset(
                    candidate_root / "val",
                    class_names=stage_config.class_names,
                    transform=val_transform,
                ),
                candidate_root,
                "hierarchical",
            )

    raise FileNotFoundError(
        "No supported dataset layout found. "
        f"Checked processed folders under {stage_config.data_dir} "
        f"and legacy flat folders under {DATA_DIR}."
    )


def train_stage(
    stage_config: StageConfig,
    args: argparse.Namespace,
    *,
    explicit_data_dir: Path | None = None,
    output_override: Path | None = None,
    metrics_override: Path | None = None,
) -> dict[str, object]:
    output_path = output_override or stage_config.output
    metrics_output = metrics_override or stage_config.metrics_output

    train_dataset, val_dataset, data_dir, layout_name = build_stage_datasets(
        stage_config,
        explicit_data_dir,
    )

    train_labels = [label for _, label in train_dataset.samples]
    class_weights = compute_class_weights(train_labels, len(stage_config.class_names))
    sampler = build_sampler(train_labels, class_weights)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(
        num_classes=len(stage_config.class_names),
        architecture=args.architecture,
        pretrained=True,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = create_loss(args.loss, class_weights, device)

    best_val_accuracy = -1.0
    history: list[dict[str, object]] = []

    print(f"\nUsing device: {device}")
    print(f"Training stage: {stage_config.stage} ({stage_config.title})")
    print(f"Architecture: {args.architecture}")
    print(f"Data directory: {data_dir}")
    print(f"Dataset layout: {layout_name}")
    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
    print(f"Class weights: {class_weights.tolist()}")
    print("Class names:", [DISPLAY_NAMES.get(name, name.title()) for name in stage_config.class_names])

    for epoch in range(1, args.epochs + 1):
        set_backbone_trainable(
            model,
            architecture=args.architecture,
            trainable=epoch > args.freeze_backbone_epochs,
        )

        train_loss, train_targets, train_predictions = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        train_metrics = calculate_classification_metrics(
            train_targets,
            train_predictions,
            stage_config.class_names,
        )

        val_loss, val_targets, val_predictions = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        val_metrics = calculate_classification_metrics(
            val_targets,
            val_predictions,
            stage_config.class_names,
        )
        scheduler.step(val_metrics["accuracy"])

        epoch_record = {
            "epoch": epoch,
            "stage": stage_config.stage,
            "architecture": args.architecture,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1_score": train_metrics["f1_score"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1_score": val_metrics["f1_score"],
            "val_confusion_matrix": val_metrics["confusion_matrix"],
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | val_recall={val_metrics['recall']:.4f} | "
            f"val_f1={val_metrics['f1_score']:.4f}"
        )
        print(f"Validation confusion matrix: {val_metrics['confusion_matrix']}")

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": args.architecture,
                    "stage": stage_config.stage,
                    "class_names": stage_config.class_names,
                    "display_names": {
                        name: DISPLAY_NAMES.get(name, name.title())
                        for name in stage_config.class_names
                    },
                    "best_val_accuracy": best_val_accuracy,
                    "epoch": epoch,
                    "input_size": 224,
                    "loss_name": args.loss,
                    "class_weights": class_weights.tolist(),
                },
                output_path,
            )
            print(f"Saved best checkpoint to {output_path} (val_acc={best_val_accuracy:.4f})")

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Saved training history to {metrics_output}")
    print("Final best validation accuracy:", f"{best_val_accuracy:.4f}")

    return {
        "stage": stage_config.stage,
        "checkpoint": str(output_path),
        "metrics_output": str(metrics_output),
        "best_val_accuracy": best_val_accuracy,
        "dataset_layout": layout_name,
        "architecture": args.architecture,
    }


def _preprocess_blockers(
    summary: dict[str, object],
    requested_stages: list[str],
) -> list[str]:
    dataset_records = {
        record["dataset"]: record
        for record in summary.get("datasets", [])
        if isinstance(record, dict) and "dataset" in record
    }
    blockers: list[str] = []

    for stage_name in requested_stages:
        record = dataset_records.get(stage_name)
        if record and record.get("status") == "missing_raw":
            reason = record.get("reason", "Missing source data.")
            blockers.append(f"{stage_name}: {reason}")

    return blockers


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    normalized_stage = normalize_stage_name(args.stage)
    stages = ["validation", "disease"] if normalized_stage == "all" else [normalized_stage]

    if args.prepare or args.prepare_only:
        summary = prepare_processed_datasets(
            raw_data_dir=args.raw_data_dir,
            plant_sources=args.plant_source,
            non_plant_sources=args.non_plant_source,
            disease_source_dirs=args.disease_source,
            seed=args.seed,
            force=args.force_prepare,
        )
        print("Prepared dataset summary:")
        print(json.dumps(summary, indent=2))
        if args.prepare_only:
            return

        blockers = _preprocess_blockers(summary, stages)
        if blockers:
            raise SystemExit(
                "Cannot start training because preprocessing is still missing source data:\n"
                + "\n".join(f"- {blocker}" for blocker in blockers)
            )

    results: list[dict[str, object]] = []
    for stage_name in stages:
        stage_config = get_stage_config(stage_name)
        results.append(
            train_stage(
                stage_config,
                args,
                explicit_data_dir=args.data_dir,
                output_override=args.output if len(stages) == 1 else None,
                metrics_override=args.metrics_output if len(stages) == 1 else None,
            )
        )

    if len(results) > 1:
        print("\nTraining summary:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
