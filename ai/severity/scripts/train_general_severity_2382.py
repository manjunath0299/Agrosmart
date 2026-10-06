# ============================================================
# GENERAL PLANT DISEASE SEVERITY MODEL
# Dataset: 2382 QC-clean SAM2 pseudo-labels
# Model: EfficientNet-B0
# Task: Continuous severity regression
#
# Severity levels:
#   0–10%       -> LOW
#   >10–40%     -> MEDIUM
#   >40–60%     -> HIGH
#   >60%        -> SEVERE
#
# IMPORTANT:
# Current clean dataset has severity values only up to ~45%.
# Therefore >45% predictions are extrapolation and are NOT
# independently validated by the current training data.
# ============================================================

import os
import json
import random

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

CSV_PATH = (
    r"ai\severity\results\general_severity_sam2_3000_qc"
    r"\severity_labels_sam2_3000_clean.csv"
)

OUTPUT_DIR = (
    r"ai\severity\results\severity_model_general_2382"
)

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 0

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SEVERITY LEVEL FUNCTIONS
# ============================================================

def get_severity_level(severity):
    """
    Convert continuous severity percentage into
    the project's four severity levels.
    """

    if severity <= 10:
        return "LOW"

    elif severity <= 40:
        return "MEDIUM"

    elif severity <= 60:
        return "HIGH"

    else:
        return "SEVERE"


def get_severity_id(severity):
    """
    Numeric severity level:
        0 = LOW
        1 = MEDIUM
        2 = HIGH
        3 = SEVERE
    """

    if severity <= 10:
        return 0

    elif severity <= 40:
        return 1

    elif severity <= 60:
        return 2

    else:
        return 3


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DATASET CLASS
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


    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = row["image_path"]

        severity = float(
            row["severity"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:

            image = self.transform(
                image
            )

        # Convert 0–100% target to 0–1
        target = severity / 100.0

        target = torch.tensor(
            target,
            dtype=torch.float32
        )

        return image, target


# ============================================================
# IMAGE TRANSFORMS
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
    )
])


val_transform = transforms.Compose([

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
    )
])


# ============================================================
# LOAD CSV
# ============================================================

print()
print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(
    CSV_PATH
)

print(
    f"Total rows in CSV: {len(df)}"
)


# ============================================================
# CLEAN DATA
# ============================================================

df["image_path"] = df[
    "image_path"
].astype(str).apply(
    os.path.normpath
)

df["severity"] = pd.to_numeric(
    df["severity"],
    errors="coerce"
)

df["disease"] = df[
    "disease"
].astype(str)


df = df.dropna(
    subset=[
        "image_path",
        "severity",
        "disease"
    ]
).reset_index(
    drop=True
)


# ============================================================
# CHECK IMAGE PATHS
# ============================================================

missing_images = []

for path in df["image_path"]:

    if not os.path.exists(path):

        missing_images.append(path)


if missing_images:

    print()
    print(
        f"ERROR: {len(missing_images)} image files "
        f"were not found."
    )

    print()
    print("First missing paths:")

    for path in missing_images[:10]:

        print(path)

    raise FileNotFoundError(
        "Some image paths are invalid."
    )


print(
    f"Valid images: {len(df)}"
)


# ============================================================
# DATASET SEVERITY STATISTICS
# ============================================================

print()
print("=" * 60)
print("DATASET SEVERITY STATISTICS")
print("=" * 60)

print(
    df["severity"].describe()
)


# ============================================================
# SEVERITY BINS
# ============================================================

severity_bins = [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    100
]

severity_labels = [
    "0-10",
    "10-20",
    "20-30",
    "30-40",
    "40-50",
    "50-60",
    "60-70",
    "70+"
]


df["severity_bin"] = pd.cut(
    df["severity"],
    bins=severity_bins,
    labels=severity_labels,
    right=False,
    include_lowest=True
)


print()
print("Overall severity distribution:")

print(
    df["severity_bin"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# DISEASE-WISE COUNTS
# ============================================================

print()
print("=" * 60)
print("DISEASE COUNTS")
print("=" * 60)

print(
    df["disease"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
#
# IMPORTANT:
# We stratify by disease rather than disease + severity.
#
# Some disease/severity combinations contain only one image.
# sklearn cannot stratify a class containing one sample.
# ============================================================

train_df, temp_df = train_test_split(

    df,

    test_size=0.30,

    random_state=SEED,

    stratify=df["disease"]
)


val_df, test_df = train_test_split(

    temp_df,

    test_size=0.50,

    random_state=SEED,

    stratify=temp_df["disease"]
)


train_df = train_df.copy()

val_df = val_df.copy()

test_df = test_df.copy()


print()
print("=" * 60)
print("DATASET SPLIT")
print("=" * 60)

print(
    f"Train      : {len(train_df)}"
)

print(
    f"Validation : {len(val_df)}"
)

print(
    f"Test       : {len(test_df)}"
)


# ============================================================
# VERIFY DISEASE DISTRIBUTION
# ============================================================

print()
print("=" * 60)
print("DISEASE DISTRIBUTION BY SPLIT")
print("=" * 60)

disease_distribution = pd.DataFrame({

    "Train":
        train_df["disease"].value_counts(),

    "Validation":
        val_df["disease"].value_counts(),

    "Test":
        test_df["disease"].value_counts()

}).fillna(0).astype(int)


print(
    disease_distribution.to_string()
)


# ============================================================
# VERIFY SEVERITY DISTRIBUTION
# ============================================================

print()
print("=" * 60)
print("SEVERITY DISTRIBUTION BY SPLIT")
print("=" * 60)


for name, split in [

    ("Train", train_df),

    ("Validation", val_df),

    ("Test", test_df)

]:

    print()
    print(f"{name}:")

    distribution = pd.cut(

        split["severity"],

        bins=severity_bins,

        labels=severity_labels,

        right=False,

        include_lowest=True

    )

    print(
        distribution
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# SAVE DATASET SPLITS
# ============================================================

train_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "train.csv"
    ),
    index=False
)

val_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "val.csv"
    ),
    index=False
)

test_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test.csv"
    ),
    index=False
)


# ============================================================
# BALANCED SAMPLING
#
# Balance:
#   disease + severity range
#
# This prevents diseases with many samples from dominating
# training.
# ============================================================

train_df["balance_key"] = (

    train_df["disease"].astype(str)

    + "_"

    + train_df["severity_bin"].astype(str)
)


group_counts = (
    train_df["balance_key"]
    .value_counts()
)


sample_weights = (

    train_df["balance_key"]
    .map(
        lambda key:
        1.0 / group_counts[key]
    )
    .values
)


sample_weights = torch.DoubleTensor(
    sample_weights
)


sampler = WeightedRandomSampler(

    weights=sample_weights,

    num_samples=len(train_df),

    replacement=True
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = SeverityDataset(

    train_df,

    transform=train_transform
)


val_dataset = SeverityDataset(

    val_df,

    transform=val_transform
)


test_dataset = SeverityDataset(

    test_df,

    transform=val_transform
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    sampler=sampler,

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
print("=" * 60)
print("BUILDING MODEL")
print("=" * 60)

print(
    "Model: EfficientNet-B0"
)


weights = (
    models.EfficientNet_B0_Weights.DEFAULT
)


model = models.efficientnet_b0(
    weights=weights
)


# Replace final classifier
in_features = (
    model.classifier[1].in_features
)


model.classifier[1] = nn.Linear(
    in_features,
    1
)


model = model.to(
    DEVICE
)


print(
    f"Device: {DEVICE}"
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )


# ============================================================
# LOSS
# ============================================================

criterion = nn.SmoothL1Loss(
    beta=0.05
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
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=3
    )
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

    predictions = []

    targets = []


    with torch.no_grad():

        for images, target in loader:

            images = images.to(

                DEVICE,

                non_blocking=True
            )


            output = model(
                images
            )


            # Sigmoid guarantees 0–1
            output = torch.sigmoid(
                output.squeeze(1)
            )


            predictions.extend(

                (
                    output
                    .cpu()
                    .numpy()
                    * 100
                )
            )


            targets.extend(

                (
                    target
                    .cpu()
                    .numpy()
                    * 100
                )
            )


    predictions = np.array(
        predictions
    )

    targets = np.array(
        targets
    )


    mae = mean_absolute_error(

        targets,

        predictions
    )


    rmse = np.sqrt(

        mean_squared_error(

            targets,

            predictions
        )
    )


    r2 = r2_score(

        targets,

        predictions
    )


    return (
        mae,
        rmse,
        r2,
        predictions,
        targets
    )


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)


best_val_mae = float("inf")

patience_counter = 0

history = []


for epoch in range(

    1,

    EPOCHS + 1

):

    model.train()


    running_loss = 0.0


    progress = tqdm(

        train_loader,

        desc=(
            f"Epoch "
            f"{epoch}/{EPOCHS}"
        )
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


        outputs = model(
            images
        )


        outputs = torch.sigmoid(

            outputs.squeeze(1)
        )


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
        )


        progress.set_postfix(

            loss=(
                f"{loss.item():.4f}"
            )
        )


    train_loss = (

        running_loss

        / len(train_loader)
    )


    (
        val_mae,
        val_rmse,
        val_r2,
        _,
        _
    ) = evaluate(
        val_loader
    )


    scheduler.step(
        val_mae
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    print()

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss : "
        f"{train_loss:.5f}"
    )

    print(
        f"Val MAE    : "
        f"{val_mae:.3f}%"
    )

    print(
        f"Val RMSE   : "
        f"{val_rmse:.3f}%"
    )

    print(
        f"Val R²     : "
        f"{val_r2:.4f}"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.2e}"
    )


    history.append({

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "val_mae":
            val_mae,

        "val_rmse":
            val_rmse,

        "val_r2":
            val_r2,

        "learning_rate":
            current_lr
    })


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_mae < best_val_mae:

        best_val_mae = val_mae

        patience_counter = 0


        checkpoint = {

            "model_state_dict":
                model.state_dict(),

            "best_val_mae":
                best_val_mae,

            "epoch":
                epoch,

            "model":
                "EfficientNet-B0",

            "task":
                "continuous severity regression",

            "severity_definition":
                "diseased leaf pixels / total leaf pixels * 100",

            "severity_levels": {

                "LOW":
                    "0-10%",

                "MEDIUM":
                    ">10-40%",

                "HIGH":
                    ">40-60%",

                "SEVERE":
                    ">60%"
            },

            "training_dataset_size":
                len(train_df),

            "validation_dataset_size":
                len(val_df),

            "test_dataset_size":
                len(test_df),

            "maximum_clean_training_severity":
                float(
                    df["severity"].max()
                )
        }


        torch.save(

            checkpoint,

            os.path.join(

                OUTPUT_DIR,

                "best_general_severity_model.pth"
            )
        )


        print()

        print(
            "*** BEST MODEL SAVED ***"
        )

        print(
            f"Best validation MAE: "
            f"{best_val_mae:.3f}%"
        )


    else:

        patience_counter += 1


        print(
            f"No improvement: "
            f"{patience_counter}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )


        if (
            patience_counter
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

with open(

    os.path.join(

        OUTPUT_DIR,

        "training_history.json"
    ),

    "w"

) as file:

    json.dump(

        history,

        file,

        indent=2
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

best_model_path = os.path.join(

    OUTPUT_DIR,

    "best_general_severity_model.pth"
)


checkpoint = torch.load(

    best_model_path,

    map_location=DEVICE,

    weights_only=False
)


model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ]
)


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)


(
    test_mae,
    test_rmse,
    test_r2,
    test_predictions,
    test_targets
) = evaluate(

    test_loader
)


print()
print(
    f"Test MAE  : "
    f"{test_mae:.3f}%"
)

print(
    f"Test RMSE : "
    f"{test_rmse:.3f}%"
)

print(
    f"Test R²   : "
    f"{test_r2:.4f}"
)


# ============================================================
# TEST PREDICTION DETAILS
# ============================================================

print()
print(
    "Generating test predictions..."
)


test_results = []


model.eval()


with torch.no_grad():

    for index in range(
        len(test_df)
    ):

        row = test_df.iloc[
            index
        ]


        image_path = (
            row["image_path"]
        )


        image = Image.open(

            image_path

        ).convert("RGB")


        image_tensor = (

            val_transform(image)

            .unsqueeze(0)

            .to(DEVICE)
        )


        output = model(

            image_tensor
        )


        predicted = (

            torch.sigmoid(output)

            .item()

            * 100
        )


        actual = float(

            row["severity"]
        )


        absolute_error = abs(

            predicted - actual
        )


        test_results.append({

            "image_path":
                image_path,

            "disease":
                row["disease"],

            "actual_severity":
                actual,

            "predicted_severity":
                predicted,

            "absolute_error":
                absolute_error,

            "actual_level":
                get_severity_level(
                    actual
                ),

            "predicted_level":
                get_severity_level(
                    predicted
                ),

            "actual_level_id":
                get_severity_id(
                    actual
                ),

            "predicted_level_id":
                get_severity_id(
                    predicted
                )
        })


results_df = pd.DataFrame(
    test_results
)


results_df.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "test_predictions.csv"
    ),

    index=False
)


# ============================================================
# DISEASE-WISE PERFORMANCE
# ============================================================

disease_results = (

    results_df

    .groupby("disease")

    .agg(

        count=(
            "actual_severity",
            "size"
        ),

        mae=(
            "absolute_error",
            "mean"
        ),

        actual_mean=(
            "actual_severity",
            "mean"
        ),

        predicted_mean=(
            "predicted_severity",
            "mean"
        )
    )

    .reset_index()
)


disease_results.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "disease_wise_performance.csv"
    ),

    index=False
)


# ============================================================
# SEVERITY LEVEL PERFORMANCE
# ============================================================

severity_results = (

    results_df

    .groupby("actual_level")

    .agg(

        count=(
            "actual_severity",
            "size"
        ),

        mae=(
            "absolute_error",
            "mean"
        ),

        actual_mean=(
            "actual_severity",
            "mean"
        ),

        predicted_mean=(
            "predicted_severity",
            "mean"
        )
    )

    .reset_index()
)


severity_results.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "severity_level_performance.csv"
    ),

    index=False
)


# ============================================================
# SEVERITY LEVEL CONFUSION MATRIX
# ============================================================

level_order = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "SEVERE"
]


level_confusion = pd.crosstab(

    results_df["actual_level"],

    results_df["predicted_level"],

    rownames=["Actual"],

    colnames=["Predicted"],

    dropna=False
)


level_confusion = level_confusion.reindex(

    index=level_order,

    columns=level_order,

    fill_value=0
)


level_confusion.to_csv(

    os.path.join(

        OUTPUT_DIR,

        "severity_level_confusion_matrix.csv"
    )
)


# ============================================================
# LEVEL ACCURACY
# ============================================================

level_accuracy = (

    results_df["actual_level"]

    == results_df["predicted_level"]

).mean()


print()
print(
    f"Severity Level Accuracy: "
    f"{level_accuracy * 100:.2f}%"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

maximum_training_severity = float(

    df["severity"].max()
)


summary = {

    "dataset": {

        "total":
            len(df),

        "train":
            len(train_df),

        "validation":
            len(val_df),

        "test":
            len(test_df)
    },


    "model":
        "EfficientNet-B0",


    "task":
        "continuous severity regression",


    "severity_definition":
        "diseased leaf pixels / total leaf pixels * 100",


    "severity_levels": {

        "LOW":
            "0-10%",

        "MEDIUM":
            ">10-40%",

        "HIGH":
            ">40-60%",

        "SEVERE":
            ">60%"
    },


    "test_metrics": {

        "MAE_percent":
            float(test_mae),

        "RMSE_percent":
            float(test_rmse),

        "R2":
            float(test_r2),

        "severity_level_accuracy":
            float(level_accuracy)
    },


    "maximum_clean_training_severity":
        maximum_training_severity,


    "warning":
        (
            "The current QC-clean dataset contains "
            "severity values only up to approximately "
            "45%. Therefore predictions above this "
            "range are extrapolations and the HIGH/"
            "SEVERE categories are not fully validated."
        )
}


with open(

    os.path.join(

        OUTPUT_DIR,

        "final_summary.json"
    ),

    "w"

) as file:

    json.dump(

        summary,

        file,

        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print()
print(
    "Best model:"
)

print(
    best_model_path
)

print()
print(
    f"Test MAE  : "
    f"{test_mae:.3f}%"
)

print(
    f"Test RMSE : "
    f"{test_rmse:.3f}%"
)

print(
    f"Test R²   : "
    f"{test_r2:.4f}"
)

print(
    f"Level Accuracy: "
    f"{level_accuracy * 100:.2f}%"
)

print()
print("Severity mapping:")

print(
    "0–10%       -> LOW"
)

print(
    ">10–40%     -> MEDIUM"
)

print(
    ">40–60%     -> HIGH"
)

print(
    ">60%        -> SEVERE"
)

print()
print(
    "IMPORTANT: Current clean training data "
    f"maximum = {maximum_training_severity:.2f}%."
)

print(
    "High/Severe predictions require additional "
    "validated high-severity data before being "
    "claimed as reliable."
)

print()