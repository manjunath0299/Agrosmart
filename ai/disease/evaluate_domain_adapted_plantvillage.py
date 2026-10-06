from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import timm

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

SPLITS_FILE = (
    BASE_DIR
    / "ai"
    / "disease"
    / "dataset"
    / "splits.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "ai"
    / "disease"
    / "models"
    / "domain_adaptation"
    / "best_domain_adapted_model.pth"
)

RESULTS_DIR = (
    BASE_DIR
    / "ai"
    / "disease"
    / "results"
    / "domain_adaptation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class PlantVillageDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform


    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = row["path"]

        label = int(row["label"])

        image = (
            Image.open(image_path)
            .convert("RGB")
        )

        if self.transform:

            image = self.transform(image)

        return image, label


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DOMAIN-ADAPTED MODEL - PLANTVILLAGE TEST")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

print()


if not SPLITS_FILE.exists():

    raise FileNotFoundError(
        f"Splits file not found:\n"
        f"{SPLITS_FILE}"
    )


df = pd.read_csv(
    SPLITS_FILE
)


# Only original PlantVillage test set
test_df = df[
    df["split"] == "test"
].copy()


print(
    f"PlantVillage test images: "
    f"{len(test_df)}"
)


# ============================================================
# CLASS MAPPING
# ============================================================

class_mapping = (
    df[
        ["label", "class_name"]
    ]
    .drop_duplicates()
    .sort_values("label")
)

label_to_class = dict(
    zip(
        class_mapping["label"],
        class_mapping["class_name"]
    )
)

num_classes = len(
    label_to_class
)

print(
    f"Number of classes: "
    f"{num_classes}"
)

print()


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose(
    [

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
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
    ]
)


# ============================================================
# DATALOADER
# ============================================================

dataset = PlantVillageDataset(
    test_df,
    transform
)

loader = DataLoader(
    dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

print(
    "Loading domain-adapted model..."
)

model = timm.create_model(
    "tf_efficientnetv2_b0",

    pretrained=False,

    num_classes=num_classes
)


checkpoint = torch.load(
    MODEL_PATH,

    map_location=DEVICE,

    weights_only=False
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict"
    in checkpoint
):

    state_dict = (
        checkpoint[
            "model_state_dict"
        ]
    )

elif (
    isinstance(checkpoint, dict)
    and "state_dict"
    in checkpoint
):

    state_dict = (
        checkpoint[
            "state_dict"
        ]
    )

else:

    state_dict = checkpoint


clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):

        key = key[
            len("module.") :
        ]

    clean_state_dict[key] = value


model.load_state_dict(
    clean_state_dict,

    strict=True
)

model = model.to(DEVICE)

model.eval()

print(
    "Model loaded successfully."
)

print()


# ============================================================
# EVALUATION
# ============================================================

all_true = []
all_pred = []


print("=" * 70)
print("EVALUATING")
print("=" * 70)


with torch.no_grad():

    for images, labels in tqdm(
        loader,
        desc="PlantVillage Test"
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_true.extend(
            labels.numpy().tolist()
        )

        all_pred.extend(
            predictions
            .cpu()
            .numpy()
            .tolist()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_true,
    all_pred
)

precision = precision_score(
    all_true,
    all_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_true,
    all_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_true,
    all_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print()

print("=" * 70)
print("PLANTVILLAGE TEST RESULTS")
print("=" * 70)

print(
    f"Test images        : {len(all_true)}"
)

print(
    f"Accuracy           : "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Weighted Precision : "
    f"{precision:.4f}"
)

print(
    f"Weighted Recall    : "
    f"{recall:.4f}"
)

print(
    f"Weighted F1        : "
    f"{f1:.4f}"
)

print("=" * 70)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

class_names = [
    label_to_class[i]
    for i in range(num_classes)
]


report = classification_report(
    all_true,
    all_pred,

    labels=list(
        range(num_classes)
    ),

    target_names=class_names,

    digits=4,

    zero_division=0
)

print()
print(report)


# ============================================================
# SAVE RESULTS
# ============================================================

report_path = (
    RESULTS_DIR
    / "plantvillage_domain_adapted_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


summary = {

    "model":
        "tf_efficientnetv2_b0",

    "training":
        "domain_adaptation",

    "dataset":
        "PlantVillage",

    "split":
        "test",

    "test_images":
        len(all_true),

    "accuracy":
        float(accuracy),

    "weighted_precision":
        float(precision),

    "weighted_recall":
        float(recall),

    "weighted_f1":
        float(f1)
}


summary_path = (
    RESULTS_DIR
    / "plantvillage_domain_adapted_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


print()
print(
    "Results saved to:"
)

print(report_path)
print(summary_path)

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)