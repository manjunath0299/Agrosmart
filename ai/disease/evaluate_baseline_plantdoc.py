# ============================================================
# evaluate_baseline_plantdoc.py
#
# Matched baseline evaluation:
# Original EfficientNetV2-B0 on the SAME 134 PlantDoc images
# used for domain-adaptation evaluation.
# ============================================================

from pathlib import Path
import json
import random

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
    confusion_matrix,
)
from tqdm import tqdm


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

EXTERNAL_DIR = (
    BASE_DIR
    / "ai"
    / "disease"
    / "external_test"
)

IMAGES_DIR = EXTERNAL_DIR / "images"

MAPPING_FILE = (
    EXTERNAL_DIR
    / "class_mapping.json"
)

SPLITS_FILE = (
    BASE_DIR
    / "ai"
    / "disease"
    / "dataset"
    / "splits.csv"
)

# ORIGINAL baseline checkpoint
MODEL_PATH = (
    BASE_DIR
    / "ai"
    / "disease"
    / "models"
    / "best_disease_model.pth"
)

RESULTS_DIR = (
    BASE_DIR
    / "ai"
    / "disease"
    / "results"
    / "baseline_plantdoc"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 4. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("ORIGINAL MODEL - MATCHED PLANTDOC EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()


# ============================================================
# 5. CHECK FILES
# ============================================================

required_paths = {
    "PlantDoc images": IMAGES_DIR,
    "PlantDoc mapping": MAPPING_FILE,
    "PlantVillage splits": SPLITS_FILE,
    "Original model": MODEL_PATH,
}

for name, path in required_paths.items():

    if not path.exists():

        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )


# ============================================================
# 6. LOAD PLANTVILLAGE CLASS MAPPING
# ============================================================

splits_df = pd.read_csv(
    SPLITS_FILE
)

required_columns = {
    "class_name",
    "label",
}

missing_columns = (
    required_columns
    - set(splits_df.columns)
)

if missing_columns:

    raise RuntimeError(
        f"Missing columns in splits.csv: "
        f"{missing_columns}"
    )


class_mapping_df = (
    splits_df[
        ["class_name", "label"]
    ]
    .drop_duplicates()
    .sort_values("label")
)


class_to_label = dict(
    zip(
        class_mapping_df["class_name"],
        class_mapping_df["label"],
    )
)


label_to_class = {
    int(label): class_name
    for class_name, label
    in class_to_label.items()
}


NUM_CLASSES = len(
    class_to_label
)

print(
    f"Number of PlantVillage classes: "
    f"{NUM_CLASSES}"
)

print()


# ============================================================
# 7. LOAD PLANTDOC MAPPING
# ============================================================

with open(
    MAPPING_FILE,
    "r",
    encoding="utf-8",
) as f:

    plantdoc_to_plantvillage = json.load(f)


# ============================================================
# 8. BUILD EXACT SAME 134-IMAGE TEST SET
# ============================================================

test_records = []

print(
    "Scanning PlantDoc external test set..."
)

print("-" * 70)


for (
    plantdoc_class,
    plantvillage_class,
) in sorted(
    plantdoc_to_plantvillage.items()
):

    class_dir = (
        IMAGES_DIR
        / plantdoc_class
    )


    if not class_dir.exists():

        raise RuntimeError(
            f"PlantDoc folder not found:\n"
            f"{class_dir}"
        )


    if (
        plantvillage_class
        not in class_to_label
    ):

        raise RuntimeError(
            "PlantVillage class not found:\n"
            f"{plantvillage_class}"
        )


    true_label = class_to_label[
        plantvillage_class
    ]


    image_files = sorted(
        [
            p.resolve()
            for p in class_dir.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )


    print(
        f"{plantdoc_class}: "
        f"{len(image_files)} images "
        f"-> {plantvillage_class}"
    )


    for image_path in image_files:

        test_records.append(
            {
                "image_path":
                    str(image_path),

                "plantdoc_class":
                    plantdoc_class,

                "true_class":
                    plantvillage_class,

                "true_label":
                    true_label,
            }
        )


test_df = pd.DataFrame(
    test_records
)


print()

print("=" * 70)

print(
    f"Mapped PlantDoc test images: "
    f"{len(test_df)}"
)

print(
    "Expected: 134"
)

print("=" * 70)


if len(test_df) != 134:

    raise RuntimeError(
        f"Expected exactly 134 images, "
        f"but found {len(test_df)}"
    )


# ============================================================
# 9. DATASET
# ============================================================

class PlantDocDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None,
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.transform = transform


    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = row[
            "image_path"
        ]

        label = int(
            row["true_label"]
        )


        image = (
            Image
            .open(image_path)
            .convert("RGB")
        )


        if self.transform:

            image = self.transform(
                image
            )


        return (
            image,
            label,
            image_path,
        )


# ============================================================
# 10. SAME PREPROCESSING AS ADAPTED MODEL
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
                0.406,
            ],

            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# 11. DATALOADER
# ============================================================

test_dataset = PlantDocDataset(
    test_df,
    transform=transform,
)

test_loader = DataLoader(
    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available(),
)


print()

print(
    f"Test dataset size: "
    f"{len(test_dataset)}"
)

print(
    f"Number of batches: "
    f"{len(test_loader)}"
)

print()


# ============================================================
# 12. CREATE ORIGINAL MODEL
# ============================================================

print(
    "Loading original EfficientNetV2-B0..."
)


model = timm.create_model(
    "tf_efficientnetv2_b0",

    pretrained=False,

    num_classes=NUM_CLASSES,
)


# ============================================================
# 13. LOAD ORIGINAL CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,

    map_location=DEVICE,

    weights_only=False,
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


# Remove DataParallel prefix
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
    "Original model loaded successfully."
)

print()


# ============================================================
# 14. EVALUATION
# ============================================================

print("=" * 70)
print("RUNNING MATCHED BASELINE EVALUATION")
print("=" * 70)


all_true = []
all_pred = []
all_confidence = []
all_paths = []


with torch.no_grad():

    for (
        images,
        labels,
        paths,
    ) in tqdm(
        test_loader,
        desc="Baseline PlantDoc",
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )


        outputs = model(
            images
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        confidence, predictions = (
            torch.max(
                probabilities,
                dim=1
            )
        )


        all_true.extend(
            labels
            .cpu()
            .numpy()
            .tolist()
        )


        all_pred.extend(
            predictions
            .cpu()
            .numpy()
            .tolist()
        )


        all_confidence.extend(
            confidence
            .cpu()
            .numpy()
            .tolist()
        )


        all_paths.extend(
            paths
        )


# ============================================================
# 15. METRICS
# ============================================================

accuracy = accuracy_score(
    all_true,
    all_pred,
)


weighted_precision = (
    precision_score(
        all_true,
        all_pred,
        average="weighted",
        zero_division=0,
    )
)


weighted_recall = (
    recall_score(
        all_true,
        all_pred,
        average="weighted",
        zero_division=0,
    )
)


weighted_f1 = (
    f1_score(
        all_true,
        all_pred,
        average="weighted",
        zero_division=0,
    )
)


# ============================================================
# 16. MAPPED 16-CLASS METRICS
# ============================================================

mapped_labels = sorted(
    test_df[
        "true_label"
    ].unique()
)


mapped_class_names = [
    label_to_class[label]
    for label in mapped_labels
]


macro_precision = (
    precision_score(
        all_true,
        all_pred,

        labels=mapped_labels,

        average="macro",

        zero_division=0,
    )
)


macro_recall = (
    recall_score(
        all_true,
        all_pred,

        labels=mapped_labels,

        average="macro",

        zero_division=0,
    )
)


macro_f1 = (
    f1_score(
        all_true,
        all_pred,

        labels=mapped_labels,

        average="macro",

        zero_division=0,
    )
)


# ============================================================
# 17. PRINT RESULTS
# ============================================================

print()

print("=" * 70)
print("ORIGINAL MODEL - MATCHED PLANTDOC RESULTS")
print("=" * 70)

print(
    f"Test Images          : "
    f"{len(all_true)}"
)

print(
    f"Accuracy             : "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Weighted Precision   : "
    f"{weighted_precision:.4f} "
    f"({weighted_precision * 100:.2f}%)"
)

print(
    f"Weighted Recall      : "
    f"{weighted_recall:.4f} "
    f"({weighted_recall * 100:.2f}%)"
)

print(
    f"Weighted F1          : "
    f"{weighted_f1:.4f} "
    f"({weighted_f1 * 100:.2f}%)"
)

print()

print(
    f"16-Class Macro Precision : "
    f"{macro_precision:.4f}"
)

print(
    f"16-Class Macro Recall    : "
    f"{macro_recall:.4f}"
)

print(
    f"16-Class Macro F1        : "
    f"{macro_f1:.4f}"
)

print("=" * 70)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_true,
    all_pred,

    labels=mapped_labels,

    target_names=mapped_class_names,

    digits=4,

    zero_division=0,
)


print()

print("=" * 70)
print("ORIGINAL MODEL CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# 19. SAVE REPORT
# ============================================================

report_txt_path = (
    RESULTS_DIR
    / "baseline_plantdoc_classification_report.txt"
)

with open(
    report_txt_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(report)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_true,
    all_pred,

    labels=mapped_labels,
)


cm_df = pd.DataFrame(
    cm,

    index=mapped_class_names,

    columns=mapped_class_names,
)


cm_path = (
    RESULTS_DIR
    / "baseline_plantdoc_confusion_matrix.csv"
)


cm_df.to_csv(
    cm_path
)


# ============================================================
# 21. SAVE PREDICTIONS
# ============================================================

prediction_records = []


for (
    path,
    true_label,
    pred_label,
    confidence,
) in zip(
    all_paths,
    all_true,
    all_pred,
    all_confidence,
):

    prediction_records.append(
        {
            "image_path": path,

            "true_label":
                int(true_label),

            "true_class":
                label_to_class[
                    int(true_label)
                ],

            "predicted_label":
                int(pred_label),

            "predicted_class":
                label_to_class[
                    int(pred_label)
                ],

            "confidence":
                float(confidence),

            "correct":
                (
                    int(true_label)
                    ==
                    int(pred_label)
                ),
        }
    )


predictions_df = pd.DataFrame(
    prediction_records
)


predictions_path = (
    RESULTS_DIR
    / "baseline_plantdoc_predictions.csv"
)


predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# 22. PER-CLASS ACCURACY
# ============================================================

true_array = np.array(
    all_true
)

pred_array = np.array(
    all_pred
)


per_class_rows = []


for label in mapped_labels:

    mask = (
        true_array
        == label
    )


    total = int(
        mask.sum()
    )


    correct = int(
        (
            pred_array[mask]
            == label
        ).sum()
    )


    per_class_rows.append(
        {
            "label":
                label,

            "class_name":
                label_to_class[
                    label
                ],

            "total_images":
                total,

            "correct":
                correct,

            "accuracy":
                (
                    correct / total
                    if total > 0
                    else 0.0
                ),
        }
    )


per_class_df = pd.DataFrame(
    per_class_rows
)


per_class_path = (
    RESULTS_DIR
    / "baseline_plantdoc_per_class_accuracy.csv"
)


per_class_df.to_csv(
    per_class_path,
    index=False
)


# ============================================================
# 23. SAVE SUMMARY
# ============================================================

summary = {

    "model":
        "Original EfficientNetV2-B0",

    "dataset":
        "PlantDoc",

    "test_images":
        len(all_true),

    "mapped_classes":
        len(mapped_labels),

    "accuracy":
        float(accuracy),

    "weighted_precision":
        float(weighted_precision),

    "weighted_recall":
        float(weighted_recall),

    "weighted_f1":
        float(weighted_f1),

    "macro_precision_16_classes":
        float(macro_precision),

    "macro_recall_16_classes":
        float(macro_recall),

    "macro_f1_16_classes":
        float(macro_f1),

    "model_path":
        str(MODEL_PATH),
}


summary_path = (
    RESULTS_DIR
    / "baseline_plantdoc_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# 24. FINAL
# ============================================================

print()

print("=" * 70)
print("MATCHED BASELINE EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Images evaluated : "
    f"{len(all_true)}"
)

print(
    f"Accuracy         : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Weighted F1      : "
    f"{weighted_f1 * 100:.2f}%"
)

print()

print(
    "Results directory:"
)

print(
    RESULTS_DIR
)

print("=" * 70)