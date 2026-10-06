from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

import timm
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = PROJECT_ROOT / "ai" / "disease" / "dataset" / "splits.csv"

MODEL_DIR = PROJECT_ROOT / "ai" / "disease" / "models"
RESULTS_DIR = PROJECT_ROOT / "ai" / "disease" / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


MODEL_NAME = "tf_efficientnetv2_b0"

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 0

HEAD_EPOCHS = 3

FINETUNE_EPOCHS = 12

HEAD_LR = 1e-3

FINETUNE_LR = 1e-4

WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42

PATIENCE = 4


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DISEASE DETECTION TRAINING")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print(f"Model: {MODEL_NAME}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Image size: {IMAGE_SIZE}")


# ============================================================
# LOAD SPLIT CSV
# ============================================================

df = pd.read_csv(CSV_FILE)

print(f"\nTotal images: {len(df)}")

classes = sorted(df["class_name"].unique())

class_to_idx = {
    class_name: index
    for index, class_name in enumerate(classes)
}

idx_to_class = {
    index: class_name
    for class_name, index in class_to_idx.items()
}

NUM_CLASSES = len(classes)

print(f"Number of classes: {NUM_CLASSES}")


# ============================================================
# DATASET
# ============================================================

class PlantDiseaseDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = row["path"]
        class_name = row["class_name"]

        image = Image.open(image_path).convert("RGB")

        label = class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.75, 1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(p=0.2),

    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

])


eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

])


# ============================================================
# SPLIT DATA
# ============================================================

train_df = df[df["split"] == "train"].copy()

val_df = df[df["split"] == "val"].copy()

test_df = df[df["split"] == "test"].copy()


train_dataset = PlantDiseaseDataset(
    train_df,
    transform=train_transform
)

val_dataset = PlantDiseaseDataset(
    val_df,
    transform=eval_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    
)


print("\nDataset sizes:")
print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test: {len(test_df)}")


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = (
    train_df["class_name"]
    .value_counts()
    .reindex(classes)
)

# Square-root inverse frequency weighting.
# This is less aggressive than 1/frequency and helps
# minority classes without excessively destabilizing training.

weights = 1.0 / np.sqrt(class_counts.values)

weights = weights / weights.mean()

class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)

print("\nClass weighting enabled.")


# ============================================================
# MODEL
# ============================================================

print("\nCreating EfficientNetV2-B0...")

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=NUM_CLASSES
)

model = model.to(device)

print("Model loaded successfully.")


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)


# ============================================================
# AMP
# ============================================================

use_amp = device.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader, optimizer):

    model.train()

    running_loss = 0.0

    predictions = []
    targets = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += (
            loss.item() * images.size(0)
        )

        preds = outputs.argmax(dim=1)

        predictions.extend(
            preds.detach().cpu().numpy()
        )

        targets.extend(
            labels.detach().cpu().numpy()
        )

    epoch_loss = (
        running_loss / len(loader.dataset)
    )

    epoch_acc = accuracy_score(
        targets,
        predictions
    )

    epoch_f1 = f1_score(
        targets,
        predictions,
        average="macro"
    )

    return epoch_loss, epoch_acc, epoch_f1


# ============================================================
# VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def validate(model, loader):

    model.eval()

    running_loss = 0.0

    predictions = []
    targets = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        running_loss += (
            loss.item() * images.size(0)
        )

        preds = outputs.argmax(dim=1)

        predictions.extend(
            preds.cpu().numpy()
        )

        targets.extend(
            labels.cpu().numpy()
        )

    val_loss = (
        running_loss / len(loader.dataset)
    )

    val_acc = accuracy_score(
        targets,
        predictions
    )

    val_f1 = f1_score(
        targets,
        predictions,
        average="macro"
    )

    return val_loss, val_acc, val_f1


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {
    "train_loss": [],
    "train_accuracy": [],
    "train_macro_f1": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_macro_f1": []
}


best_val_f1 = 0.0

epochs_without_improvement = 0


# ============================================================
# STAGE 1
# TRAIN CLASSIFICATION HEAD
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 1: TRAINING CLASSIFICATION HEAD")
print("=" * 70)


# Freeze backbone
for parameter in model.parameters():
    parameter.requires_grad = False


# Unfreeze classifier
classifier = model.get_classifier()

for parameter in classifier.parameters():
    parameter.requires_grad = True


optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=HEAD_LR,
    weight_decay=WEIGHT_DECAY
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=HEAD_EPOCHS
)


for epoch in range(HEAD_EPOCHS):

    start_time = time.time()

    train_loss, train_acc, train_f1 = train_one_epoch(
        model,
        train_loader,
        optimizer
    )

    val_loss, val_acc, val_f1 = validate(
        model,
        val_loader
    )

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_acc)
    history["train_macro_f1"].append(train_f1)

    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)
    history["val_macro_f1"].append(val_f1)

    elapsed = time.time() - start_time

    print(
        f"Epoch {epoch + 1}/{HEAD_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Train F1: {train_f1:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val F1: {val_f1:.4f} | "
        f"Time: {elapsed:.1f}s"
    )


# ============================================================
# STAGE 2
# FINE-TUNE ENTIRE MODEL
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 2: FINE-TUNING ENTIRE MODEL")
print("=" * 70)


for parameter in model.parameters():
    parameter.requires_grad = True


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=FINETUNE_LR,
    weight_decay=WEIGHT_DECAY
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=FINETUNE_EPOCHS
)


for epoch in range(FINETUNE_EPOCHS):

    start_time = time.time()

    train_loss, train_acc, train_f1 = train_one_epoch(
        model,
        train_loader,
        optimizer
    )

    val_loss, val_acc, val_f1 = validate(
        model,
        val_loader
    )

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_acc)
    history["train_macro_f1"].append(train_f1)

    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)
    history["val_macro_f1"].append(val_f1)

    elapsed = time.time() - start_time

    print(
        f"Epoch {epoch + HEAD_EPOCHS}/{HEAD_EPOCHS + FINETUNE_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Train F1: {train_f1:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val F1: {val_f1:.4f} | "
        f"Time: {elapsed:.1f}s"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        checkpoint = {
            "model_name": MODEL_NAME,
            "num_classes": NUM_CLASSES,
            "classes": classes,
            "model_state_dict": model.state_dict(),
            "best_val_f1": best_val_f1,
            "image_size": IMAGE_SIZE
        }

        torch.save(
            checkpoint,
            MODEL_DIR / "best_disease_model.pth"
        )

        print(
            f"  >>> Best model saved "
            f"(Val Macro F1: {best_val_f1:.4f})"
        )

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            break


# ============================================================
# SAVE CLASS INFORMATION
# ============================================================

with open(
    RESULTS_DIR / "classes.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "num_classes": NUM_CLASSES,
            "classes": classes
        },
        file,
        indent=4
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(
    RESULTS_DIR / "training_history.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        history,
        file,
        indent=4
    )


# ============================================================
# PLOT ACCURACY
# ============================================================

epochs = range(
    1,
    len(history["train_accuracy"]) + 1
)

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "EfficientNetV2-B0 Disease Classification Accuracy"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "accuracy_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT LOSS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "EfficientNetV2-B0 Disease Classification Loss"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "loss_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"Best Validation Macro F1: {best_val_f1:.4f}")

print(
    f"\nModel saved to:\n"
    f"{MODEL_DIR / 'best_disease_model.pth'}"
)

print(
    f"\nResults saved to:\n"
    f"{RESULTS_DIR}"
)

print("\nNext step: run evaluate.py")