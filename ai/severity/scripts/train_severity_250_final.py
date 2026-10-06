from pathlib import Path
import random
import json

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"E:\smart_agriculture\smart_agriculture")

DATASET_DIR = ROOT / "ai" / "severity" / "dataset"

IMAGE_DIR = DATASET_DIR / "images" / "annotation_pool"

V2_CSV = (
    ROOT / "ai" / "severity" / "results"
    / "severity_250_v2"
    / "severity_labels_250_v2.csv"
)

CORRECTED_DIR = (
    ROOT / "ai" / "severity" / "results"
    / "corrected_masks"
)

OUT_DIR = (
    ROOT / "ai" / "severity" / "results"
    / "severity_model_250_final"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# Training
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 60
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10
NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("FINAL 250-IMAGE SEVERITY MODEL")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD V2 LABELS
# ============================================================

df = pd.read_csv(V2_CSV)

print(f"\nV2 records: {len(df)}")

# Detect filename column
filename_col = None

for col in [
    "filename",
    "image",
    "File Name",
    "file_name",
    "image_name"
]:
    if col in df.columns:
        filename_col = col
        break

if filename_col is None:
    raise ValueError(
        f"Filename column not found. Columns: {list(df.columns)}"
    )


# Detect severity column
severity_col = None

for col in [
    "severity_percent",
    "severity",
    "Severity",
    "severity_pct"
]:
    if col in df.columns:
        severity_col = col
        break

if severity_col is None:
    raise ValueError(
        f"Severity column not found. Columns: {list(df.columns)}"
    )


# ============================================================
# CORRECTED SEVERITY VALUES
# ============================================================

corrected_values = {
    "9436": 61.40,
    "8220": 57.39,
    "7597": 51.51,
    "6401": 50.50,
    "8328": 47.13,
    "9411": 36.64,
}


def get_corrected_value(filename):

    name = str(filename)

    for key, value in corrected_values.items():

        if f"RS_Erly.B {key}" in name:
            return value

    return None


# ============================================================
# BUILD FINAL DATASET
# ============================================================

records = []

for _, row in df.iterrows():

    filename = str(row[filename_col])

    image_path = IMAGE_DIR / filename

    # Sometimes filename in CSV does not exactly match
    if not image_path.exists():

        candidates = list(
            IMAGE_DIR.glob(f"*{Path(filename).stem}*")
        )

        if len(candidates) == 0:

            # UUID may differ, try suffix matching
            candidates = [
                p for p in IMAGE_DIR.glob("*")
                if Path(filename).name in p.name
            ]

        if len(candidates) == 0:
            print(f"WARNING image not found: {filename}")
            continue

        image_path = candidates[0]

    corrected = get_corrected_value(filename)

    if corrected is not None:

        severity = corrected
        label_source = "manual_corrected"

    else:

        severity = float(row[severity_col])
        label_source = "v2_pseudo_label"

    records.append({
        "filename": filename,
        "image_path": str(image_path),
        "severity": severity,
        "label_source": label_source
    })


data = pd.DataFrame(records)

print(f"\nFinal usable images: {len(data)}")

print(
    "\nLabel sources:"
)

print(
    data["label_source"].value_counts()
)


if len(data) < 200:
    raise RuntimeError(
        "Too few images found. Check IMAGE_DIR and V2_CSV."
    )


# ============================================================
# SEVERITY BINS FOR STRATIFIED SPLIT
# ============================================================

def severity_bin(x):

    if x < 5:
        return "0-5"

    elif x < 10:
        return "5-10"

    elif x < 20:
        return "10-20"

    elif x < 30:
        return "20-30"

    elif x < 40:
        return "30-40"

    elif x < 50:
        return "40-50"

    else:
        return "50+"


data["severity_bin"] = data["severity"].apply(
    severity_bin
)


# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

# First: 80% temporary train, 20% test
train_val, test = train_test_split(
    data,
    test_size=0.20,
    random_state=SEED,
    stratify=data["severity_bin"]
)


# Then: 80% train, 20% validation
train, val = train_test_split(
    train_val,
    test_size=0.20,
    random_state=SEED,
    stratify=train_val["severity_bin"]
)


train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)


print("\nDataset split:")
print(f"Train: {len(train)}")
print(f"Val  : {len(val)}")
print(f"Test : {len(test)}")


# Save splits
train.to_csv(
    OUT_DIR / "train.csv",
    index=False
)

val.to_csv(
    OUT_DIR / "val.csv",
    index=False
)

test.to_csv(
    OUT_DIR / "test.csv",
    index=False
)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.2
    ),

    transforms.RandomRotation(
        15
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.03
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

class SeverityDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        severity = float(
            row["severity"]
        )

        # Normalize to 0-1
        severity = severity / 100.0

        return (
            image,
            torch.tensor(
                severity,
                dtype=torch.float32
            ),
            row["filename"]
        )


# ============================================================
# DATALOADERS
# ============================================================

train_dataset = SeverityDataset(
    train,
    train_transform
)

val_dataset = SeverityDataset(
    val,
    eval_transform
)

test_dataset = SeverityDataset(
    test,
    eval_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(
    weights=weights
)


# Replace classification head
in_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(p=0.30),
    nn.Linear(in_features, 1),
    nn.Sigmoid()
)

model = model.to(DEVICE)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.SmoothL1Loss(
    beta=0.05
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2


# ============================================================
# TRAINING
# ============================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "val_mae": [],
    "val_rmse": [],
    "val_r2": [],
    "lr": []
}


best_val_mae = float("inf")
best_epoch = 0
patience_counter = 0


for epoch in range(1, NUM_EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_losses = []

    for images, targets, _ in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            images
        ).squeeze(1)

        loss = criterion(
            outputs,
            targets
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        train_losses.append(
            loss.item()
        )


    train_loss = np.mean(
        train_losses
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_losses = []
    val_true = []
    val_pred = []

    with torch.no_grad():

        for images, targets, _ in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                images
            ).squeeze(1)

            loss = criterion(
                outputs,
                targets
            )

            val_losses.append(
                loss.item()
            )

            val_true.extend(
                (targets.cpu().numpy() * 100)
            )

            val_pred.extend(
                (outputs.cpu().numpy() * 100)
            )


    val_loss = np.mean(
        val_losses
    )

    val_mae, val_rmse, val_r2 = calculate_metrics(
        val_true,
        val_pred
    )


    scheduler.step(
        val_mae
    )


    current_lr = optimizer.param_groups[0]["lr"]


    # Save history
    history["train_loss"].append(
        float(train_loss)
    )

    history["val_loss"].append(
        float(val_loss)
    )

    history["val_mae"].append(
        float(val_mae)
    )

    history["val_rmse"].append(
        float(val_rmse)
    )

    history["val_r2"].append(
        float(val_r2)
    )

    history["lr"].append(
        float(current_lr)
    )


    print(
        f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
        f"Train Loss {train_loss:.5f} | "
        f"Val Loss {val_loss:.5f} | "
        f"MAE {val_mae:.2f}% | "
        f"RMSE {val_rmse:.2f}% | "
        f"R2 {val_r2:.3f} | "
        f"LR {current_lr:.2e}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
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
                "val_r2": val_r2,
                "image_size": IMAGE_SIZE
            },
            OUT_DIR / "best_severity_model_250.pth"
        )

        print(
            "  -> Best model saved"
        )

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping at epoch {epoch}"
            )

            break


# ============================================================
# SAVE HISTORY
# ============================================================

with open(
    OUT_DIR / "training_history.json",
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=2
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    OUT_DIR / "best_severity_model_250.pth",
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# TEST
# ============================================================

test_true = []
test_pred = []
test_names = []

with torch.no_grad():

    for images, targets, names in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(
            images
        ).squeeze(1)

        predictions = (
            outputs.cpu().numpy() * 100
        )

        actual = (
            targets.numpy() * 100
        )

        test_pred.extend(
            predictions
        )

        test_true.extend(
            actual
        )

        test_names.extend(
            names
        )


test_mae, test_rmse, test_r2 = calculate_metrics(
    test_true,
    test_pred
)


# ============================================================
# TEST RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Best epoch : {best_epoch}"
)

print(
    f"Test MAE   : {test_mae:.2f}%"
)

print(
    f"Test RMSE  : {test_rmse:.2f}%"
)

print(
    f"Test R²    : {test_r2:.4f}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

pred_df = pd.DataFrame({
    "filename": test_names,
    "actual_severity": test_true,
    "predicted_severity": test_pred
})

pred_df["absolute_error"] = (
    np.abs(
        pred_df["actual_severity"]
        -
        pred_df["predicted_severity"]
    )
)

pred_df = pred_df.sort_values(
    "absolute_error",
    ascending=False
)

pred_df.to_csv(
    OUT_DIR / "test_predictions.csv",
    index=False
)


# ============================================================
# PLOT TRAINING CURVES
# ============================================================

epochs_done = range(
    1,
    len(history["train_loss"]) + 1
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_done,
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    epochs_done,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    "Severity Model Training and Validation Loss"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "loss_curve.png",
    dpi=200
)

plt.close()


# MAE curve
plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_done,
    history["val_mae"],
    label="Validation MAE"
)

plt.xlabel("Epoch")
plt.ylabel("MAE (%)")
plt.title(
    "Severity Model Validation MAE"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "mae_curve.png",
    dpi=200
)

plt.close()


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    test_true,
    test_pred,
    alpha=0.75
)

min_value = min(
    min(test_true),
    min(test_pred)
)

max_value = max(
    max(test_true),
    max(test_pred)
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel(
    "Actual Severity (%)"
)

plt.ylabel(
    "Predicted Severity (%)"
)

plt.title(
    "Actual vs Predicted Severity"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "actual_vs_predicted.png",
    dpi=200
)

plt.close()


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

summary = {
    "dataset_size": len(data),
    "train_size": len(train),
    "validation_size": len(val),
    "test_size": len(test),
    "best_epoch": best_epoch,
    "best_validation_mae": float(best_val_mae),
    "test_mae": float(test_mae),
    "test_rmse": float(test_rmse),
    "test_r2": float(test_r2),
    "device": str(DEVICE),
    "model": "EfficientNet-B0",
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE
}

with open(
    OUT_DIR / "final_summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Output directory:\n{OUT_DIR}"
)

print(
    "\nBest checkpoint:"
)

print(
    OUT_DIR / "best_severity_model_250.pth"
)

print(
    "\nTest predictions:"
)

print(
    OUT_DIR / "test_predictions.csv"
)