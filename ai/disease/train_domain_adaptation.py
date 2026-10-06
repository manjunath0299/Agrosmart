from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import timm
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path(r"C:\smart_agriculture")

TRAIN_CSV = (
    PROJECT
    / "ai"
    / "disease"
    / "dataset"
    / "domain_adaptation"
    / "domain_train.csv"
)

VAL_CSV = (
    PROJECT
    / "ai"
    / "disease"
    / "dataset"
    / "domain_adaptation"
    / "domain_val.csv"
)

BASE_MODEL = (
    PROJECT
    / "ai"
    / "disease"
    / "models"
    / "best_disease_model.pth"
)

OUTPUT_DIR = (
    PROJECT
    / "ai"
    / "disease"
    / "models"
    / "domain_adaptation"
)

RESULTS_DIR = (
    PROJECT
    / "ai"
    / "disease"
    / "results"
    / "domain_adaptation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

MODEL_NAME = "tf_efficientnetv2_b0"

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 0

MAX_EPOCHS = 10

EARLY_STOPPING_PATIENCE = 3

SEED = 42

# PlantDoc oversampling strength.
#
# PlantDoc is ~3% of the combined dataset.
# A multiplier of 8 makes the real-world domain much
# more visible during fine-tuning without creating files.
PLANTDOC_WEIGHT = 8.0

# Low learning rates because we are fine-tuning
# an already trained model.
BACKBONE_LR = 1e-5
CLASSIFIER_LR = 5e-5

WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.05


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = DEVICE.type == "cuda"


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# PRINT ENVIRONMENT
# ============================================================

print("=" * 70)
print("DOMAIN ADAPTATION TRAINING")
print("=" * 70)

print()
print(f"Device       : {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )

print(f"Model        : {MODEL_NAME}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Max epochs   : {MAX_EPOCHS}")
print(f"PlantDoc weight : {PLANTDOC_WEIGHT}x")


# ============================================================
# DATASET
# ============================================================

class DiseaseDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.dataframe = (
            dataframe.reset_index(drop=True)
        )

        self.transform = transform

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = row["path"]

        label = int(row["label"])

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            raise RuntimeError(
                f"Could not read image:\n"
                f"{image_path}\n"
                f"Error: {e}"
            )

        if self.transform is not None:

            image = self.transform(
                image
            )

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.75, 1.0)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomVerticalFlip(
            p=0.15
        ),

        transforms.RandomRotation(
            degrees=20
        ),

        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.20,
            hue=0.05
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        ),
    ]
)


val_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        ),
    ]
)


# ============================================================
# LOAD CSV FILES
# ============================================================

print()
print("Loading datasets...")

train_df = pd.read_csv(
    TRAIN_CSV
)

val_df = pd.read_csv(
    VAL_CSV
)

print(
    f"Training images   : {len(train_df)}"
)

print(
    f"Validation images : {len(val_df)}"
)


# ============================================================
# VERIFY DATASET COLUMNS
# ============================================================

required_columns = {
    "path",
    "class_name",
    "label",
    "source",
    "split",
}

for dataframe, name in [
    (train_df, "training"),
    (val_df, "validation"),
]:

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:

        raise ValueError(
            f"{name} CSV is missing columns: "
            f"{sorted(missing)}"
        )


# ============================================================
# VERIFY LABEL RANGE
# ============================================================

all_labels = sorted(
    train_df["label"]
    .astype(int)
    .unique()
)

if all_labels != list(range(38)):

    raise ValueError(
        "Expected labels 0-37, but found:\n"
        f"{all_labels}"
    )

NUM_CLASSES = 38


# ============================================================
# SOURCE DISTRIBUTION
# ============================================================

print()
print("Training source distribution:")

print(
    train_df["source"]
    .value_counts()
    .to_string()
)

print()
print("Validation source distribution:")

print(
    val_df["source"]
    .value_counts()
    .to_string()
)


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = DiseaseDataset(
    train_df,
    transform=train_transform
)

val_dataset = DiseaseDataset(
    val_df,
    transform=val_transform
)


# ============================================================
# PLANTDOC OVERSAMPLING
# ============================================================

print()
print("=" * 70)
print("CREATING PLANTDOC-WEIGHTED SAMPLER")
print("=" * 70)

sample_weights = np.ones(
    len(train_df),
    dtype=np.float64
)

plantdoc_mask = (
    train_df["source"]
    .astype(str)
    .eq("PlantDoc")
    .values
)

sample_weights[
    plantdoc_mask
] = PLANTDOC_WEIGHT

sample_weights = torch.tensor(
    sample_weights,
    dtype=torch.double
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_df),
    replacement=True
)

print(
    f"PlantDoc samples receiving "
    f"{PLANTDOC_WEIGHT}x sampling weight."
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("Creating EfficientNetV2-B0...")

model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=NUM_CLASSES
)


# ============================================================
# LOAD EXISTING BEST MODEL
# ============================================================

print()
print(
    f"Loading existing model:\n"
    f"{BASE_MODEL}"
)

checkpoint = torch.load(
    BASE_MODEL,
    map_location="cpu",
    weights_only=False
)


# Handle several common checkpoint formats
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        # Sometimes the checkpoint itself
        # is the state dictionary.
        state_dict = checkpoint

else:

    state_dict = checkpoint


# Remove DataParallel prefix if present
clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):

        key = key[
            len("module.") :
        ]

    clean_state_dict[key] = value


missing_keys, unexpected_keys = (
    model.load_state_dict(
        clean_state_dict,
        strict=False
    )
)


print(
    f"Loaded checkpoint."
)

if missing_keys:

    print(
        f"Missing keys: "
        f"{len(missing_keys)}"
    )

if unexpected_keys:

    print(
        f"Unexpected keys: "
        f"{len(unexpected_keys)}"
    )


model = model.to(DEVICE)


# ============================================================
# OPTIMIZER
# ============================================================

# Give the classifier a slightly larger LR
# while keeping the backbone conservative.

classifier_parameters = []
backbone_parameters = []

for name, parameter in model.named_parameters():

    if not parameter.requires_grad:
        continue

    if (
        "classifier" in name
        or "head" in name
    ):

        classifier_parameters.append(
            parameter
        )

    else:

        backbone_parameters.append(
            parameter
        )


optimizer = torch.optim.AdamW(
    [
        {
            "params": backbone_parameters,
            "lr": BACKBONE_LR,
        },
        {
            "params": classifier_parameters,
            "lr": CLASSIFIER_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=LABEL_SMOOTHING
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_EPOCHS,
    eta_min=1e-7
)


# ============================================================
# AMP SCALER
# ============================================================

if USE_AMP:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    all_targets = []
    all_predictions = []

    progress = tqdm(
        train_loader,
        desc="Training",
        leave=False
    )

    for images, targets in progress:

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

        if USE_AMP:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    targets
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            outputs = model(images)

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


        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(dim=1)
        )

        all_targets.extend(
            targets.detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    epoch_loss = (
        running_loss
        / len(train_dataset)
    )

    epoch_accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    epoch_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_f1
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    running_loss = 0.0

    all_targets = []
    all_predictions = []

    progress = tqdm(
        val_loader,
        desc="Validation",
        leave=False
    )

    for images, targets in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        if USE_AMP:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    targets
                )

        else:

            outputs = model(images)

            loss = criterion(
                outputs,
                targets
            )


        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(dim=1)
        )

        all_targets.extend(
            targets.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


    epoch_loss = (
        running_loss
        / len(val_dataset)
    )

    epoch_accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    epoch_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_f1
    )


# ============================================================
# TRAINING LOOP
# ============================================================

history = {
    "epoch": [],
    "train_loss": [],
    "train_accuracy": [],
    "train_f1": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_f1": [],
    "learning_rate_backbone": [],
    "learning_rate_classifier": [],
}

best_val_f1 = -1.0

best_epoch = 0

epochs_without_improvement = 0

best_model_path = (
    OUTPUT_DIR
    / "best_domain_adapted_model.pth"
)


print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(1, MAX_EPOCHS + 1):

    start_time = time.time()

    print()
    print(
        f"Epoch {epoch}/{MAX_EPOCHS}"
    )

    print("-" * 70)

    (
        train_loss,
        train_accuracy,
        train_f1
    ) = train_one_epoch()

    (
        val_loss,
        val_accuracy,
        val_f1
    ) = validate()

    scheduler.step()

    epoch_time = (
        time.time()
        - start_time
    )

    current_backbone_lr = (
        optimizer.param_groups[0]["lr"]
    )

    current_classifier_lr = (
        optimizer.param_groups[1]["lr"]
    )

    history["epoch"].append(
        epoch
    )

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["train_f1"].append(
        train_f1
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )

    history["val_f1"].append(
        val_f1
    )

    history[
        "learning_rate_backbone"
    ].append(
        current_backbone_lr
    )

    history[
        "learning_rate_classifier"
    ].append(
        current_classifier_lr
    )


    print()
    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Train Acc  : "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Train F1   : "
        f"{train_f1:.4f}"
    )

    print(
        f"Val Loss   : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Acc    : "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Val F1     : "
        f"{val_f1:.4f}"
    )

    print(
        f"Time       : "
        f"{epoch_time:.1f}s"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        best_epoch = epoch

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "model_name":
                    MODEL_NAME,

                "num_classes":
                    NUM_CLASSES,

                "best_val_f1":
                    best_val_f1,

                "best_val_accuracy":
                    val_accuracy,

                "epoch":
                    epoch,

                "class_mapping":(
                    train_df[
                        ["class_name", "label"]
                    ]
                    .drop_duplicates()
                    .sort_values("label")
                    .set_index("label")["class_name"]
                    .to_dict()
                ),
                
                "plantdoc_weight":
                    PLANTDOC_WEIGHT,
            },
            best_model_path
        )

        print()
        print(
            ">>> BEST MODEL SAVED"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement: "
            f"{epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print()
        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_file = (
    RESULTS_DIR
    / "training_history.json"
)

with open(
    history_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# PLOT ACCURACY
# ============================================================

epochs = history["epoch"]

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    history["train_accuracy"],
    label="Train Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(
    "Domain Adaptation - Accuracy"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "accuracy_curve.png",
    dpi=200
)

plt.close()


# ============================================================
# PLOT F1
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    history["train_f1"],
    label="Train Macro F1"
)

plt.plot(
    epochs,
    history["val_f1"],
    label="Validation Macro F1"
)

plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title(
    "Domain Adaptation - Macro F1"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "f1_curve.png",
    dpi=200
)

plt.close()


# ============================================================
# PLOT LOSS
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    "Domain Adaptation - Loss"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "loss_curve.png",
    dpi=200
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("DOMAIN ADAPTATION TRAINING COMPLETE")
print("=" * 70)

print()
print(
    f"Best epoch       : {best_epoch}"
)

print(
    f"Best validation F1 : "
    f"{best_val_f1:.4f}"
)

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
    history_file
)

print()
print(
    f"Results directory:"
)

print(
    RESULTS_DIR
)

print()
print(
    "IMPORTANT:"
)

print(
    "The PlantDoc external test set "
    "was never used during training."
)