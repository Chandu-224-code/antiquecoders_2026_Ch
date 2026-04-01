"""Full EfficientNetB0 training pipeline for wheat disease classification."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.constants import CLASS_NAMES, DISPLAY_NAMES
from model.dataset import WheatDiseaseDataset, build_train_transform, build_val_transform
from model.losses import FocalLoss
from model.metrics import calculate_classification_metrics
from model.network import build_efficientnet_b0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train wheat disease classifier.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "training" / "data" / "processed")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "model" / "saved_model.pth")
    parser.add_argument("--metrics-output", type=Path, default=PROJECT_ROOT / "model" / "training_metrics.json")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--loss", choices=["weighted_ce", "focal"], default="weighted_ce")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def freeze_backbone(model: nn.Module, freeze: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = not freeze


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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"

    train_dataset = WheatDiseaseDataset(train_dir, transform=build_train_transform())
    val_dataset = WheatDiseaseDataset(val_dir, transform=build_val_transform())

    train_labels = [label for _, label in train_dataset.samples]
    class_weights = compute_class_weights(train_labels, len(CLASS_NAMES))
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
    model = build_efficientnet_b0(num_classes=len(CLASS_NAMES), pretrained=True).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = create_loss(args.loss, class_weights, device)

    best_val_accuracy = -1.0
    history: list[dict] = []

    print(f"Using device: {device}")
    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
    print(f"Class weights: {class_weights.tolist()}")
    print("Class names:", [DISPLAY_NAMES[name] for name in CLASS_NAMES])

    for epoch in range(1, args.epochs + 1):
        freeze_backbone(model, freeze=epoch <= args.freeze_backbone_epochs)

        train_loss, train_targets, train_predictions = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        train_metrics = calculate_classification_metrics(train_targets, train_predictions, CLASS_NAMES)

        val_loss, val_targets, val_predictions = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        val_metrics = calculate_classification_metrics(val_targets, val_predictions, CLASS_NAMES)
        scheduler.step(val_metrics["accuracy"])

        epoch_record = {
            "epoch": epoch,
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
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "display_names": DISPLAY_NAMES,
                    "best_val_accuracy": best_val_accuracy,
                    "epoch": epoch,
                    "input_size": 224,
                    "loss_name": args.loss,
                    "class_weights": class_weights.tolist(),
                },
                args.output,
            )
            print(f"Saved best checkpoint to {args.output} (val_acc={best_val_accuracy:.4f})")

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Saved training history to {args.metrics_output}")
    print("Final best validation accuracy:", f"{best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
