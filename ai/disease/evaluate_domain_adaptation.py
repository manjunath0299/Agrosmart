# ============================================================
# evaluate_domain_adaptation.py
# Evaluate Domain-Adapted EfficientNetV2-B0 on PlantDoc
# External Test Set
# ============================================================

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import timm

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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

# IMPORTANT:
# PlantDoc images are inside external_test/images/
IMAGES_DIR = EXTERNAL_DIR / "images"

MODEL_PATH = (
    BASE_DIR
    / "ai"
    / "disease"
    / "models"
    / "domain_adaptation"
    / "best_domain_adapted_model.pth"
)

SPLITS_FILE = (
    BASE_DIR
    / "ai"
    / "disease"
    / "dataset"
    / "splits.csv"
)

MAPPING_FILE = (
    EXTERNAL_DIR
    / "class_mapping.json"
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
    exist_ok=True,
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
print("DOMAIN ADAPTATION - PLANTDOC EXTERNAL EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()


# ============================================================
# 5. CHECK REQUIRED FILES/FOLDERS
# ============================================================

required_paths = {
    "PlantDoc images directory": IMAGES_DIR,
    "Model checkpoint": MODEL_PATH,
    "PlantVillage splits": SPLITS_FILE,
    "PlantDoc class mapping": MAPPING_FILE,
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
        "Missing columns in splits.csv: "
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
# 7. LOAD PLANTDOC -> PLANTVILLAGE MAPPING
# ============================================================

with open(
    MAPPING_FILE,
    "r",
    encoding="utf-8",
) as f:

    plantdoc_to_plantvillage = json.load(f)


print("PlantDoc mapping:")
print("-" * 70)

for (
    plantdoc_class,
    plantvillage_class
) in sorted(
    plantdoc_to_plantvillage.items()
):

    print(
        f"{plantdoc_class} -> "
        f"{plantvillage_class}"
    )

print()


# ============================================================
# 8. BUILD EXACT PLANTDOC TEST SET
# ============================================================

test_records = []

print(
    "Scanning PlantDoc external test set..."
)

print("-" * 70)

for (
    plantdoc_class,
    plantvillage_class
) in sorted(
    plantdoc_to_plantvillage.items()
):

    # IMPORTANT:
    # Actual structure is:
    #
    # external_test/
    #     images/
    #         Apple Scab Leaf/
    #         Apple rust leaf/
    #         ...
    #
    class_dir = (
        IMAGES_DIR
        / plantdoc_class
    )

    if not class_dir.exists():

        print(
            f"WARNING: folder not found: "
            f"{class_dir}"
        )

        continue


    if (
        plantvillage_class
        not in class_to_label
    ):

        raise RuntimeError(
            "PlantVillage class not found "
            f"in splits.csv:\n"
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
                and
                p.suffix.lower()
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
                "image_path": str(
                    image_path
                ),

                "plantdoc_class":
                    plantdoc_class,

                "true_class":
                    plantvillage_class,

                "true_label":
                    true_label,
            }
        )


# ============================================================
# 9. CREATE DATAFRAME
# ============================================================

test_df = pd.DataFrame(
    test_records
)

print()

print("=" * 70)

print(
    "Mapped PlantDoc test images: "
    f"{len(test_df)}"
)

print(
    "Expected: 134"
)

print("=" * 70)


if len(test_df) != 134:

    raise RuntimeError(
        "\nExpected exactly 134 mapped "
        "PlantDoc images, but found "
        f"{len(test_df)}.\n\n"
        "Do not evaluate until this is "
        "resolved."
    )


# ============================================================
# 10. DATASET CLASS
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


        try:

            image = (
                Image
                .open(image_path)
                .convert("RGB")
            )

        except Exception as e:

            raise RuntimeError(
                f"Could not read image:\n"
                f"{image_path}\n"
                f"Error: {e}"
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
# 11. IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose(
    [

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
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
# 12. DATALOADER
# ============================================================

test_dataset = PlantDocDataset(
    dataframe=test_df,
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
# 13. CREATE MODEL
# ============================================================

print(
    "Loading domain-adapted model..."
)

model = timm.create_model(
    "tf_efficientnetv2_b0",

    pretrained=False,

    num_classes=NUM_CLASSES,
)


# ============================================================
# 14. LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)


if isinstance(
    checkpoint,
    dict
):

    if (
        "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        "state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


# Remove DataParallel prefix if present

clean_state_dict = {}

for key, value in (
    state_dict.items()
):

    if key.startswith(
        "module."
    ):

        key = key[
            len("module.") :
        ]

    clean_state_dict[
        key
    ] = value


model.load_state_dict(
    clean_state_dict,
    strict=True,
)

model = model.to(DEVICE)

model.eval()

print(
    "Model loaded successfully."
)

print()


# ============================================================
# 15. RUN EVALUATION
# ============================================================

print("=" * 70)
print("RUNNING EXTERNAL EVALUATION")
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
        desc="Evaluating",
    ):

        images = images.to(
            DEVICE,
            non_blocking=True,
        )


        outputs = model(
            images
        )


        probabilities = (
            torch.softmax(
                outputs,
                dim=1,
            )
        )


        confidence, predictions = (
            torch.max(
                probabilities,
                dim=1,
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
# 16. BASIC METRICS
# ============================================================

accuracy = accuracy_score(
    all_true,
    all_pred,
)


precision_weighted = (
    precision_score(
        all_true,
        all_pred,
        average="weighted",
        zero_division=0,
    )
)


recall_weighted = (
    recall_score(
        all_true,
        all_pred,
        average="weighted",
        zero_division=0,
    )
)


f1_weighted = (
    f1_score(
        all_true,
        all_pred,
        average="weighted",
        zero_division=0,
    )
)


# ============================================================
# 17. 38-CLASS MACRO METRICS
# ============================================================

all_labels = list(
    range(NUM_CLASSES)
)


precision_macro_38 = (
    precision_score(
        all_true,
        all_pred,
        labels=all_labels,
        average="macro",
        zero_division=0,
    )
)


recall_macro_38 = (
    recall_score(
        all_true,
        all_pred,
        labels=all_labels,
        average="macro",
        zero_division=0,
    )
)


f1_macro_38 = (
    f1_score(
        all_true,
        all_pred,
        labels=all_labels,
        average="macro",
        zero_division=0,
    )
)


# ============================================================
# 18. PRINT MAIN RESULTS
# ============================================================

print()

print("=" * 70)
print("PLANTDOC EXTERNAL TEST RESULTS")
print("=" * 70)

print(
    f"Test Images              : "
    f"{len(all_true)}"
)

print(
    f"Accuracy                 : "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Weighted Precision       : "
    f"{precision_weighted:.4f}"
)

print(
    f"Weighted Recall          : "
    f"{recall_weighted:.4f}"
)

print(
    f"Weighted F1              : "
    f"{f1_weighted:.4f}"
)

print()

print(
    f"Macro Precision (38 cls) : "
    f"{precision_macro_38:.4f}"
)

print(
    f"Macro Recall (38 cls)    : "
    f"{recall_macro_38:.4f}"
)

print(
    f"Macro F1 (38 cls)        : "
    f"{f1_macro_38:.4f}"
)

print("=" * 70)


# ============================================================
# 19. MAPPED 16-CLASS REPORT
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


print()

print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(
    all_true,
    all_pred,

    labels=mapped_labels,

    target_names=mapped_class_names,

    digits=4,

    zero_division=0,
)

print(report)


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
    / "plantdoc_confusion_matrix.csv"
)


cm_df.to_csv(
    cm_path
)


print(
    f"Confusion matrix saved to:\n"
    f"{cm_path}"
)


# ============================================================
# 21. CLASSIFICATION REPORT CSV
# ============================================================

report_dict = (
    classification_report(
        all_true,
        all_pred,

        labels=mapped_labels,

        target_names=mapped_class_names,

        output_dict=True,

        zero_division=0,
    )
)


report_df = (
    pd.DataFrame(
        report_dict
    )
    .transpose()
)


report_path = (
    RESULTS_DIR
    / "plantdoc_classification_report.csv"
)


report_df.to_csv(
    report_path
)


print(
    f"Classification report saved to:\n"
    f"{report_path}"
)


# ============================================================
# 22. SAVE INDIVIDUAL PREDICTIONS
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

    true_class = (
        label_to_class[
            int(true_label)
        ]
    )


    predicted_class = (
        label_to_class[
            int(pred_label)
        ]
    )


    prediction_records.append(
        {
            "image_path": path,

            "true_label":
                int(true_label),

            "true_class":
                true_class,

            "predicted_label":
                int(pred_label),

            "predicted_class":
                predicted_class,

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


predictions_df = (
    pd.DataFrame(
        prediction_records
    )
)


predictions_path = (
    RESULTS_DIR
    / "plantdoc_predictions.csv"
)


predictions_df.to_csv(
    predictions_path,
    index=False,
)


print(
    f"Predictions saved to:\n"
    f"{predictions_path}"
)


# ============================================================
# 23. PER-CLASS ACCURACY
# ============================================================

per_class_rows = []


true_array = np.array(
    all_true
)

pred_array = np.array(
    all_pred
)


for label in mapped_labels:

    true_mask = (
        true_array
        == label
    )


    total = int(
        true_mask.sum()
    )


    correct = int(
        (
            pred_array[
                true_mask
            ]
            == label
        ).sum()
    )


    class_accuracy = (
        correct / total
        if total > 0
        else 0.0
    )


    per_class_rows.append(
        {
            "label": label,

            "class_name":
                label_to_class[
                    label
                ],

            "total_images":
                total,

            "correct":
                correct,

            "accuracy":
                class_accuracy,
        }
    )


per_class_df = (
    pd.DataFrame(
        per_class_rows
    )
)


per_class_path = (
    RESULTS_DIR
    / "plantdoc_per_class_accuracy.csv"
)


per_class_df.to_csv(
    per_class_path,
    index=False,
)


print(
    f"Per-class accuracy saved to:\n"
    f"{per_class_path}"
)


# ============================================================
# 24. SAVE SUMMARY JSON
# ============================================================

summary = {

    "model":
        "tf_efficientnetv2_b0",

    "evaluation_dataset":
        "PlantDoc",

    "test_images":
        len(all_true),

    "num_classes_model":
        NUM_CLASSES,

    "mapped_test_classes":
        len(mapped_labels),

    "accuracy":
        float(accuracy),

    "weighted_precision":
        float(
            precision_weighted
        ),

    "weighted_recall":
        float(
            recall_weighted
        ),

    "weighted_f1":
        float(
            f1_weighted
        ),

    "macro_precision_38_classes":
        float(
            precision_macro_38
        ),

    "macro_recall_38_classes":
        float(
            recall_macro_38
        ),

    "macro_f1_38_classes":
        float(
            f1_macro_38
        ),

    "mapped_classes":
        mapped_class_names,

    "model_path":
        str(MODEL_PATH),
}


summary_path = (
    RESULTS_DIR
    / "plantdoc_evaluation_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=4,
    )


print(
    f"Summary saved to:\n"
    f"{summary_path}"
)


# ============================================================
# 25. FINAL
# ============================================================

print()

print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"PlantDoc images evaluated : "
    f"{len(all_true)}"
)

print(
    f"Accuracy                  : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Weighted F1               : "
    f"{f1_weighted:.4f}"
)

print()

print(
    "Output directory:"
)

print(
    RESULTS_DIR
)

print("=" * 70)