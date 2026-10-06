import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

COLOR_DIR = ROOT / "PlantVillage-Dataset" / "raw" / "color" / "Tomato___Early_blight"

SPLIT_DIR = ROOT / "ai" / "severity" / "dataset" / "split_109_v3"

OUTPUT_DIR = ROOT / "ai" / "severity" / "results" / "severity_model_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "val.csv"
TEST_CSV = SPLIT_DIR / "test.csv"

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("EARLY BLIGHT SEVERITY REGRESSION — V2")
print("=" * 70)

print(f"\nPyTorch: {torch.__version__}")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )


# ============================================================
# INDEX ORIGINAL IMAGES
# ============================================================

print("\nIndexing PlantVillage images...")

image_index = {}

for p in COLOR_DIR.glob("*"):
    if p.is_file():
        image_index[p.name] = p

print(f"Indexed images: {len(image_index)}")


# ============================================================
# DATASET
# ============================================================

class SeverityDataset(Dataset):

    def __init__(self, csv_path, transform=None):

        self.df = pd.read_csv(csv_path).copy()
        self.transform = transform

        self.samples = []

        for _, row in self.df.iterrows():

            image_name = str(row["image"])

            if image_name not in image_index:
                raise FileNotFoundError(
                    f"Image not found in PlantVillage dataset:\n{image_name}"
                )

            severity = float(row["severity_percent"])

            self.samples.append(
                (
                    image_index[image_name],
                    severity
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        image_path, severity_percent = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Normalize severity from 0-100 -> 0-1
        severity = severity_percent / 100.0

        return (
            image,
            torch.tensor(severity, dtype=torch.float32),
            image_path.name,
            severity_percent
        )


# ============================================================
# TRANSFORMS
# ============================================================

weights = EfficientNet_B0_Weights.DEFAULT

mean = weights.transforms().mean
std = weights.transforms().std


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.03
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=mean,
        std=std
    )
])


eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=mean,
        std=std
    )
])


# ============================================================
# DATA
# ============================================================

train_ds = SeverityDataset(
    TRAIN_CSV,
    train_transform
)

val_ds = SeverityDataset(
    VAL_CSV,
    eval_transform
)

test_ds = SeverityDataset(
    TEST_CSV,
    eval_transform
)


train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


print("\nDataset:")
print(f"Train: {len(train_ds)}")
print(f"Validation: {len(val_ds)}")
print(f"Test: {len(test_ds)}")


# ============================================================
# MODEL
# ============================================================

print("\nLoading EfficientNet-B0...")

model = efficientnet_b0(
    weights=weights
)

in_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(p=0.30),
    nn.Linear(in_features, 1),
    nn.Sigmoid()
)

model = model.to(device)

print("Model ready.")


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)


# ============================================================
# WEIGHTED SMOOTH L1 LOSS
# ============================================================

def weighted_smooth_l1(pred, target):

    base_loss = nn.functional.smooth_l1_loss(
        pred,
        target,
        reduction="none"
    )

    # Target is normalized 0-1.
    # Higher severity receives somewhat larger weight.
    weight = 1.0 + 1.5 * torch.pow(
        target,
        1.5
    )

    return (base_loss * weight).mean()


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):

    model.eval()

    predictions = []
    targets = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for images, severity, _, _ in loader:

            images = images.to(device)
            severity = severity.to(device).unsqueeze(1)

            outputs = model(images)

            loss = weighted_smooth_l1(
                outputs,
                severity
            )

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions.extend(
                outputs.squeeze(1).cpu().numpy()
            )

            targets.extend(
                severity.squeeze(1).cpu().numpy()
            )

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Convert back to percentage
    predictions_percent = predictions * 100
    targets_percent = targets * 100

    mae = mean_absolute_error(
        targets_percent,
        predictions_percent
    )

    rmse = np.sqrt(
        mean_squared_error(
            targets_percent,
            predictions_percent
        )
    )

    r2 = r2_score(
        targets_percent,
        predictions_percent
    )

    avg_loss = total_loss / total_samples

    return (
        avg_loss,
        mae,
        rmse,
        r2,
        predictions_percent,
        targets_percent
    )


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

best_val_mae = float("inf")
best_epoch = 0
patience_counter = 0

history = {
    "train_loss": [],
    "val_loss": [],
    "train_mae": [],
    "val_mae": [],
    "val_rmse": [],
    "val_r2": []
}


best_model_path = OUTPUT_DIR / "best_severity_model_v2.pth"


for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    train_predictions = []
    train_targets = []

    for images, severity, _, _ in train_loader:

        images = images.to(device)

        severity = severity.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images)

        loss = weighted_smooth_l1(
            outputs,
            severity
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        train_predictions.extend(
            outputs.detach().squeeze(1).cpu().numpy()
        )

        train_targets.extend(
            severity.detach().squeeze(1).cpu().numpy()
        )

    train_loss = running_loss / len(train_ds)

    train_predictions = np.array(train_predictions) * 100
    train_targets = np.array(train_targets) * 100

    train_mae = mean_absolute_error(
        train_targets,
        train_predictions
    )

    (
        val_loss,
        val_mae,
        val_rmse,
        val_r2,
        _,
        _
    ) = evaluate(
        model,
        val_loader
    )

    scheduler.step(val_mae)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_mae"].append(train_mae)
    history["val_mae"].append(val_mae)
    history["val_rmse"].append(val_rmse)
    history["val_r2"].append(val_r2)

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train Loss {train_loss:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Train MAE {train_mae:.2f}% | "
        f"Val MAE {val_mae:.2f}% | "
        f"Val RMSE {val_rmse:.2f} | "
        f"Val R² {val_r2:.3f}"
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_mae < best_val_mae:

        best_val_mae = val_mae
        best_epoch = epoch
        patience_counter = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "val_r2": val_r2
            },
            best_model_path
        )

        print(
            f"  -> BEST MODEL SAVED "
            f"(Val MAE: {val_mae:.2f}%)"
        )

    else:

        patience_counter += 1

    if patience_counter >= PATIENCE:

        print(
            f"\nEarly stopping at epoch {epoch}."
        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

checkpoint = torch.load(
    best_model_path,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(f"Best epoch: {best_epoch}")
print(
    f"Best validation MAE: "
    f"{checkpoint['val_mae']:.2f}%"
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

(
    test_loss,
    test_mae,
    test_rmse,
    test_r2,
    predictions,
    targets
) = evaluate(
    model,
    test_loader
)

print(f"\nTest MAE:  {test_mae:.2f}%")
print(f"Test RMSE: {test_rmse:.2f}%")
print(f"Test R²:   {test_r2:.4f}")

print(
    f"\nPrediction range: "
    f"{predictions.min():.2f}% – "
    f"{predictions.max():.2f}%"
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_names = []

model.eval()

with torch.no_grad():

    for images, severity, names, _ in test_loader:

        test_names.extend(names)


test_df = pd.DataFrame({
    "image": test_names,
    "severity_percent": targets,
    "predicted_severity_percent": predictions
})

test_df["predicted_severity_percent"] = (
    test_df["predicted_severity_percent"]
    .clip(0, 100)
)

test_df["absolute_error_percent"] = (
    abs(
        test_df["severity_percent"]
        -
        test_df["predicted_severity_percent"]
    )
)

test_csv = OUTPUT_DIR / "test_predictions_v2.csv"

test_df.to_csv(
    test_csv,
    index=False
)


# ============================================================
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    OUTPUT_DIR / "training_history_v2.csv",
    index=False
)


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 7))

plt.scatter(
    targets,
    predictions,
    s=70
)

max_value = max(
    targets.max(),
    predictions.max()
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--"
)

plt.xlabel("Actual Severity (%)")
plt.ylabel("Predicted Severity (%)")
plt.title("Actual vs Predicted Severity — V2 Test Set")

plt.xlim(0, max_value + 5)
plt.ylim(0, max_value + 5)

plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted_v2.png",
    dpi=200
)

plt.close()


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Weighted SmoothL1 Loss")
plt.title("Severity Regression Loss — V2")

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "loss_curve_v2.png",
    dpi=200
)

plt.close()


# ============================================================
# MAE CURVE
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    history["train_mae"],
    label="Train MAE"
)

plt.plot(
    history["val_mae"],
    label="Validation MAE"
)

plt.xlabel("Epoch")
plt.ylabel("MAE (%)")
plt.title("Severity Regression MAE — V2")

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "mae_curve_v2.png",
    dpi=200
)

plt.close()


# ============================================================
# WORST TEST ERRORS
# ============================================================

print("\n" + "=" * 70)
print("WORST TEST ERRORS")
print("=" * 70)

print(
    test_df
    .sort_values(
        "absolute_error_percent",
        ascending=False
    )
    [
        [
            "image",
            "severity_percent",
            "predicted_severity_percent",
            "absolute_error_percent"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("SEVERITY MODEL V2 COMPLETE")
print("=" * 70)

print(f"\nModel:")
print(best_model_path)

print("\nTest predictions:")
print(test_csv)

print("\nPlots:")
print(OUTPUT_DIR)

print("\nIMPORTANT:")
print("Predictions are constrained to 0–100%.")
print("V2 must be compared against V1 on the SAME test split.")