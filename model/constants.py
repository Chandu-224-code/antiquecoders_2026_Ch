"""Shared constants used by training and inference."""

CLASS_NAMES = ["healthy", "rust", "blight", "mildew", "spot"]

DISPLAY_NAMES = {
    "healthy": "Healthy",
    "rust": "Rust",
    "blight": "Leaf Blight",
    "mildew": "Powdery Mildew",
    "spot": "Spot Blotch",
}

INPUT_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

