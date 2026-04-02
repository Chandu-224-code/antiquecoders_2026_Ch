"""Shared constants used by training and inference."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training" # Kept for retro compatibility if needed
DATA_DIR = TRAINING_DIR / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = TRAINING_DIR / "artifacts"

VALIDATION_DATA_DIR = DATA_DIR / "validation"
DISEASE_DATA_DIR = DATA_DIR / "disease"
LEGACY_TRAIN_DIR = DATA_DIR / "train"
LEGACY_VAL_DIR = DATA_DIR / "val"
LEGACY_TEST_DIR = DATA_DIR / "test"

RAW_DATA_DIR = TRAINING_DIR / "data" / "raw"
RAW_GATE_DIR = RAW_DATA_DIR / "gate"
RAW_VALIDATION_DIR = RAW_DATA_DIR / "validation"
RAW_WHEAT_DATA_DIR = RAW_DATA_DIR / "wheat_disease"

VALIDATION_CLASS_NAMES = ["plant", "non_plant"]
DISEASE_CLASS_NAMES = ["Aphid", "Black Rust", "Blast", "Brown Rust", "Common Root Rot", "Fusarium Head Blight", "Healthy", "Leaf Blight", "Mildew", "Mite", "Septoria", "Smut", "Stem fly", "Tan spot", "Yellow Rust"]
PREDICTION_CLASS_NAMES = DISEASE_CLASS_NAMES

CLASS_NAMES = PREDICTION_CLASS_NAMES

DISPLAY_NAMES = {
    "plant": "Plant Leaf",
    "non_plant": "Non-Plant Image",
    "Aphid": "Aphid",
    "Black Rust": "Black Rust",
    "Blast": "Blast",
    "Brown Rust": "Brown Rust",
    "Common Root Rot": "Common Root Rot",
    "Fusarium Head Blight": "Fusarium Head Blight",
    "Healthy": "Healthy",
    "Leaf Blight": "Leaf Blight",
    "Mildew": "Powdery Mildew",
    "Mite": "Mite",
    "Septoria": "Septoria",
    "Smut": "Smut",
    "Stem fly": "Stem Fly",
    "Tan spot": "Tan Spot",
    "Yellow Rust": "Yellow Rust",
}

INPUT_SIZE = 224
DEFAULT_TOP_K = 3
DEFAULT_ARCHITECTURE = "mobilenet_v2"
SUPPORTED_ARCHITECTURES = ("mobilenet_v2", "resnet18", "resnet34")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VALIDATION_THRESHOLD = 0.60
DISEASE_CONFIDENCE_THRESHOLD = 0.65

MIN_BLUR_SCORE = 10.0
MIN_BRIGHTNESS_SCORE = 40.0
MAX_BRIGHTNESS_SCORE = 215.0

VALIDATION_CHECKPOINT = CHECKPOINT_DIR / "plant_nonplant_best.pth"
DISEASE_CHECKPOINT = PROJECT_ROOT / "models" / "best_model.pth"
VALIDATION_METRICS_PATH = ARTIFACTS_DIR / "validation_training_metrics.json"
DISEASE_METRICS_PATH = ARTIFACTS_DIR / "disease_training_metrics.json"
PREPROCESS_SUMMARY_PATH = ARTIFACTS_DIR / "preprocessing_summary.json"

# Backward-compatible aliases while the rest of the app moves to the cleaner names.
GATE_DATA_DIR = VALIDATION_DATA_DIR
GATE_CLASS_NAMES = VALIDATION_CLASS_NAMES
GATE_CHECKPOINT = VALIDATION_CHECKPOINT
GATE_METRICS_PATH = VALIDATION_METRICS_PATH
STAGE1_DATA_DIR = DISEASE_DATA_DIR
STAGE2_DATA_DIR = DISEASE_DATA_DIR
STAGE1_CHECKPOINT = DISEASE_CHECKPOINT
STAGE2_CHECKPOINT = DISEASE_CHECKPOINT
STAGE1_METRICS_PATH = DISEASE_METRICS_PATH
STAGE2_METRICS_PATH = DISEASE_METRICS_PATH
ALL_CLASS_NAMES = PREDICTION_CLASS_NAMES
STAGE1_CLASS_NAMES = ["healthy", "diseased"]
GATE_REJECTION_THRESHOLD = VALIDATION_THRESHOLD
STAGE1_HEALTHY_THRESHOLD = 0.80
STAGE2_CONFIDENCE_THRESHOLD = DISEASE_CONFIDENCE_THRESHOLD

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
