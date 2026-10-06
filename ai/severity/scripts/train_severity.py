from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ORIGINAL_DIR = (
    PROJECT_ROOT
    / "PlantVillage-Dataset"
    / "raw"
    / "color"
    / "Tomato___Early_blight"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "dataset"
    / "split_109_v3"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "severity_model"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

SEED = 42

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_WORKERS = 0

EPOCHS = 100

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 15

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


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
# PRINT ENVIRONMENT
# ============================================================

print("=" * 70)
print("EARLY BLIGHT SEVERITY REGRESSION")
print("=" * 70)

print()
print("PyTorch:", torch.__version__)

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )


# ============================================================
# DATASET
# ============================================================

class SeverityDataset(Dataset):

    def __init__(
        self,
        csv_path,
        image_dir,
        transform=None
    ):

        self.df = pd.read_csv(
            csv_path
        )

        self.image_dir = Path(
            image_dir
        )

        self.transform = transform

        # ----------------------------------------------------
        # Verify all images exist
        # ----------------------------------------------------

        missing = []

        for filename in self.df["image"]:

            path = (
                self.image_dir
                / filename
            )

            if not path.exists():
                missing.append(
                    filename
                )

        if missing:

            raise FileNotFoundError(
                f"{len(missing)} images "
                f"are missing. Example: "
                f"{missing[:3]}"
            )

    def __len__(self):

        return len(self.df)

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[index]

        image_path = (
            self.image_dir
            / row["image"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        severity = float(
            row["severity_percent"]
        )

        if self.transform:

            image = self.transform(
                image
            )

        target = torch.tensor(
            severity,
            dtype=torch.float32
        )

        return (
            image,
            target
        )


# ============================================================
# TRANSFORMS
# ============================================================
#
# We intentionally avoid random crops because the target
# severity represents the disease area of the COMPLETE leaf.
#
# Horizontal flip preserves the approximate severity ratio.
# Mild color changes should also preserve severity.
# ============================================================

imagenet_mean = [
    0.485,
    0.456,
    0.406
]

imagenet_std = [
    0.229,
    0.224,
    0.225
]


train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.10,
        hue=0.02
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        imagenet_mean,
        imagenet_std
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        imagenet_mean,
        imagenet_std
    ),
])


# ============================================================
# DATASETS
# ============================================================

train_dataset = SeverityDataset(
    SPLIT_DIR / "train.csv",
    ORIGINAL_DIR,
    train_transform
)

val_dataset = SeverityDataset(
    SPLIT_DIR / "val.csv",
    ORIGINAL_DIR,
    eval_transform
)

test_dataset = SeverityDataset(
    SPLIT_DIR / "test.csv",
    ORIGINAL_DIR,
    eval_transform
)


print()
print(
    f"Train images:      {len(train_dataset)}"
)

print(
    f"Validation images: {len(val_dataset)}"
)

print(
    f"Test images:       {len(test_dataset)}"
)


# ============================================================
# DATALOADERS
# ============================================================

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

print()
print("Loading EfficientNet-B0...")

weights = (
    EfficientNet_B0_Weights.DEFAULT
)

model = efficientnet_b0(
    weights=weights
)


# ------------------------------------------------------------
# Replace classification head with regression head
# ------------------------------------------------------------

in_features = (
    model.classifier[1].in_features
)

model.classifier = nn.Sequential(

    nn.Dropout(
        p=0.30
    ),

    nn.Linear(
        in_features,
        128
    ),

    nn.ReLU(),

    nn.Dropout(
        p=0.20
    ),

    nn.Linear(
        128,
        1
    )
)


model = model.to(
    DEVICE
)


print(
    "Model ready."
)


# ============================================================
# LOSS
# ============================================================
#
# SmoothL1Loss is less sensitive to occasional noisy
# pseudo-labels than pure MSE.
# ============================================================

criterion = nn.SmoothL1Loss(
    beta=5.0
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    min_lr=1e-7
)


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = (
    DEVICE.type == "cuda"
)

if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    targets
):

    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )

    targets = np.asarray(
        targets,
        dtype=np.float32
    )

    mae = np.mean(
        np.abs(
            predictions
            - targets
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                predictions
                - targets
            ) ** 2
        )
    )

    ss_res = np.sum(
        (
            targets
            - predictions
        ) ** 2
    )

    ss_tot = np.sum(
        (
            targets
            - np.mean(targets)
        ) ** 2
    )

    if ss_tot > 0:

        r2 = (
            1
            - ss_res / ss_tot
        )

    else:

        r2 = 0.0

    return (
        float(mae),
        float(rmse),
        float(r2)
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    all_predictions = []

    all_targets = []

    for images, targets in train_loader:

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

        if use_amp:

            with torch.amp.autocast(
                device_type="cuda"
            ):

                outputs = model(
                    images
                ).squeeze(1)

                loss = criterion(
                    outputs,
                    targets
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            outputs = model(
                images
            ).squeeze(1)

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        all_predictions.extend(
            outputs.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_targets.extend(
            targets.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    epoch_loss = (
        running_loss
        / len(train_dataset)
    )

    mae, rmse, r2 = calculate_metrics(
        all_predictions,
        all_targets
    )

    return (
        epoch_loss,
        mae,
        rmse,
        r2
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(loader):

    model.eval()

    running_loss = 0.0

    all_predictions = []

    all_targets = []

    for images, targets in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        if use_amp:

            with torch.amp.autocast(
                device_type="cuda"
            ):

                outputs = model(
                    images
                ).squeeze(1)

                loss = criterion(
                    outputs,
                    targets
                )

        else:

            outputs = model(
                images
            ).squeeze(1)

            loss = criterion(
                outputs,
                targets
            )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        all_predictions.extend(
            outputs.cpu()
            .numpy()
            .tolist()
        )

        all_targets.extend(
            targets.cpu()
            .numpy()
            .tolist()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    mae, rmse, r2 = calculate_metrics(
        all_predictions,
        all_targets
    )

    return (
        epoch_loss,
        mae,
        rmse,
        r2
    )


# ============================================================
# TRAINING
# ============================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "train_mae": [],
    "val_mae": [],
    "train_rmse": [],
    "val_rmse": [],
    "train_r2": [],
    "val_r2": [],
    "learning_rate": [],
}


best_val_mae = float(
    "inf"
)

best_epoch = 0

patience_counter = 0

best_model_path = (
    OUTPUT_DIR
    / "best_severity_efficientnet_b0.pth"
)


print()
print("=" * 70)
print("TRAINING")
print("=" * 70)


for epoch in range(
    1,
    EPOCHS + 1
):

    train_loss, train_mae, train_rmse, train_r2 = (
        train_one_epoch()
    )

    val_loss, val_mae, val_rmse, val_r2 = (
        evaluate(
            val_loader
        )
    )

    scheduler.step(
        val_loss
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )

    history["train_mae"].append(
        train_mae
    )

    history["val_mae"].append(
        val_mae
    )

    history["train_rmse"].append(
        train_rmse
    )

    history["val_rmse"].append(
        val_rmse
    )

    history["train_r2"].append(
        train_r2
    )

    history["val_r2"].append(
        val_r2
    )

    history["learning_rate"].append(
        current_lr
    )

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
    # Save best model based on validation MAE
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
                "image_size": IMAGE_SIZE,
                "model_name": "efficientnet_b0",
            },
            best_model_path
        )

        print(
            f"  -> BEST MODEL SAVED "
            f"(Val MAE: {val_mae:.2f}%)"
        )

    else:

        patience_counter += 1

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print()
        print(
            f"Early stopping at epoch "
            f"{epoch}."
        )

        break


# ============================================================
# SAVE HISTORY
# ============================================================

history_path = (
    OUTPUT_DIR
    / "training_history.json"
)

with open(
    history_path,
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

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

print(
    f"Best epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Best validation MAE: "
    f"{checkpoint['val_mae']:.2f}%"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_loss, test_mae, test_rmse, test_r2 = (
    evaluate(
        test_loader
    )
)

print()
print(
    f"Test MAE:  {test_mae:.2f}%"
)

print(
    f"Test RMSE: {test_rmse:.2f}%"
)

print(
    f"Test R²:   {test_r2:.4f}"
)


# ============================================================
# DETAILED TEST PREDICTIONS
# ============================================================

@torch.no_grad()
def get_predictions(loader):

    model.eval()

    predictions = []

    targets = []

    for images, target in loader:

        images = images.to(
            DEVICE
        )

        output = model(
            images
        ).squeeze(1)

        predictions.extend(
            output.cpu()
            .numpy()
            .tolist()
        )

        targets.extend(
            target.numpy()
            .tolist()
        )

    return (
        np.asarray(predictions),
        np.asarray(targets)
    )


test_predictions, test_targets = (
    get_predictions(
        test_loader
    )
)

test_df = pd.read_csv(
    SPLIT_DIR
    / "test.csv"
)

test_results = test_df.copy()

test_results[
    "predicted_severity_percent"
] = np.round(
    test_predictions,
    2
)

test_results[
    "absolute_error_percent"
] = np.round(
    np.abs(
        test_predictions
        - test_targets
    ),
    2
)

test_results = test_results.sort_values(
    "severity_percent"
)


predictions_path = (
    OUTPUT_DIR
    / "test_predictions.csv"
)

test_results.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# PLOTS
# ============================================================

# ------------------------------------------------------------
# Loss curve
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "SmoothL1 Loss"
)

plt.title(
    "Severity Regression Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "loss_curve.png",
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# MAE curve
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history["train_mae"],
    label="Train MAE"
)

plt.plot(
    history["val_mae"],
    label="Validation MAE"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "MAE (percentage points)"
)

plt.title(
    "Severity Regression MAE"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "mae_curve.png",
    dpi=200
)

plt.close()


# ------------------------------------------------------------
# Actual vs predicted
# ------------------------------------------------------------

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    test_targets,
    test_predictions
)

minimum = min(
    test_targets.min(),
    test_predictions.min()
)

maximum = max(
    test_targets.max(),
    test_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel(
    "Actual Severity (%)"
)

plt.ylabel(
    "Predicted Severity (%)"
)

plt.title(
    "Actual vs Predicted Severity — Test Set"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "actual_vs_predicted.png",
    dpi=200
)

plt.close()


# ============================================================
# SAVE FINAL METRICS
# ============================================================

metrics = {
    "best_epoch": int(
        checkpoint["epoch"]
    ),

    "best_validation_mae": float(
        checkpoint["val_mae"]
    ),

    "best_validation_rmse": float(
        checkpoint["val_rmse"]
    ),

    "best_validation_r2": float(
        checkpoint["val_r2"]
    ),

    "test_mae": float(
        test_mae
    ),

    "test_rmse": float(
        test_rmse
    ),

    "test_r2": float(
        test_r2
    ),

    "train_samples": len(
        train_dataset
    ),

    "validation_samples": len(
        val_dataset
    ),

    "test_samples": len(
        test_dataset
    ),

    "model": "EfficientNet-B0",

    "image_size": IMAGE_SIZE,

    "batch_size": BATCH_SIZE,

    "learning_rate": LEARNING_RATE,

    "weight_decay": WEIGHT_DECAY,

    "seed": SEED,
}


metrics_path = (
    OUTPUT_DIR
    / "final_metrics.json"
)

with open(
    metrics_path,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()
print(
    f"Best model:"
)

print(
    best_model_path
)

print()
print(
    f"Training history:"
)

print(
    history_path
)

print()
print(
    f"Test predictions:"
)

print(
    predictions_path
)

print()
print(
    f"Final metrics:"
)

print(
    metrics_path
)

print()
print(
    f"Loss curve:"
)

print(
    OUTPUT_DIR
    / "loss_curve.png"
)

print()
print(
    f"MAE curve:"
)

print(
    OUTPUT_DIR
    / "mae_curve.png"
)

print()
print(
    f"Actual vs predicted:"
)

print(
    OUTPUT_DIR
    / "actual_vs_predicted.png"
)

print()
print("=" * 70)