import os
import random
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
def main():
    MODEL_DIR = r"C:\antiquecoders_2026Prashanth\training\models"
    MODEL_PATH = os.path.join(MODEL_DIR, "efficientnet_best.pth")

    os.makedirs(MODEL_DIR, exist_ok=True)
    # ================= CONFIG =================
    DATA_DIR = r"C:\antiquecoders_2026Prashanth\training\data\processed"
    BATCH_SIZE = 16
    IMG_SIZE = 224
    EPOCHS = 15
    LR = 3e-4
    SEED = 42

    # ================= SEED =================
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ================= TRANSFORMS =================
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    # ================= DATASETS =================
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)
    test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_transform)

    class_names = train_dataset.classes
    print("Classes:", class_names)

    # ================= CLASS BALANCE =================
    targets = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in targets]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ================= MODEL =================
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, len(class_names))
    )

    model = model.to(device)

    # ================= LOSS =================
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # ================= OPTIMIZER =================
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ================= TRAIN =================
    def train_one_epoch():
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(train_loader, desc="Train"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

        return total_loss / len(train_loader), correct / total

    # ================= VALIDATE =================
    def validate(loader):
        model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Val"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()

        return total_loss / len(loader), correct / total

    # ================= TRAIN LOOP =================
    best_acc = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = validate(val_loader)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names
    }, MODEL_PATH)

    print(f"✅ Model saved at: {MODEL_PATH}")

    # ================= TEST =================
    print("\nTesting best model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("✅ Loaded best model from disk")

    test_loss, test_acc = validate(test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")
if __name__ == "__main__":
    main()