from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import timm
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = (
    PROJECT_ROOT
    / "ai"
    / "disease"
    / "dataset"
    / "splits.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "ai"
    / "disease"
    / "models"
    / "best_disease_model.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "ai"
    / "disease"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_NAME = "tf_efficientnetv2_b0"

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("PLANT DISEASE MODEL EVALUATION")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )


# ============================================================
# CHECK FILES
# ============================================================

if not CSV_FILE.exists():

    raise FileNotFoundError(
        f"\nSplit CSV not found:\n{CSV_FILE}"
    )


if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"\nTrained model not found:\n{MODEL_FILE}"
    )


print("\nCSV:")
print(CSV_FILE)

print("\nModel:")
print(MODEL_FILE)


# ============================================================
# LOAD DATAFRAME
# ============================================================

df = pd.read_csv(CSV_FILE)

test_df = df[
    df["split"] == "test"
].copy()

print("\n" + "=" * 70)
print("TEST DATASET")
print("=" * 70)

print(
    f"Total test images: {len(test_df)}"
)


# ============================================================
# LOAD MODEL CHECKPOINT
# ============================================================

print("\nLoading trained model...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)


# ============================================================
# LOAD CLASS INFORMATION
# ============================================================

classes = checkpoint["classes"]

num_classes = checkpoint["num_classes"]

image_size = checkpoint.get(
    "image_size",
    IMAGE_SIZE
)

print(
    f"Number of classes: {num_classes}"
)

print(
    f"Image size: {image_size}"
)


# ============================================================
# CREATE CLASS MAPPINGS
# ============================================================

class_to_idx = {
    class_name: index
    for index, class_name in enumerate(classes)
}

idx_to_class = {
    index: class_name
    for index, class_name in enumerate(classes)
}


# ============================================================
# DATASET
# ============================================================

class PlantDiseaseTestDataset(Dataset):

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

        image_path = row["path"]

        class_name = row["class_name"]

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as error:

            raise RuntimeError(
                f"\nCould not open image:\n"
                f"{image_path}\n"
                f"Error: {error}"
            )

        label = class_to_idx[
            class_name
        ]

        if self.transform:

            image = self.transform(
                image
            )

        return (
            image,
            label,
            image_path
        )


# ============================================================
# TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(
        (image_size, image_size)
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
# CREATE DATASET
# ============================================================

test_dataset = PlantDiseaseTestDataset(
    test_df,
    transform=test_transform
)


# ============================================================
# DATA LOADER
# ============================================================

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True
)


print(
    f"Test batches: {len(test_loader)}"
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating EfficientNetV2-B0...")

model = timm.create_model(

    MODEL_NAME,

    pretrained=False,

    num_classes=num_classes
)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print(
    "Trained weights loaded successfully."
)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING TEST EVALUATION")
print("=" * 70)

all_predictions = []

all_targets = []

all_probabilities = []

all_paths = []

start_time = time.time()


with torch.no_grad():

    for batch_index, (
        images,
        labels,
        paths
    ) in enumerate(test_loader):

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(
            images
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_targets.extend(
            labels.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        all_paths.extend(
            paths
        )

        if (
            batch_index + 1
        ) % 25 == 0:

            print(
                f"Processed "
                f"{batch_index + 1}/"
                f"{len(test_loader)} batches"
            )


elapsed_time = time.time() - start_time


# ============================================================
# CONVERT TO NUMPY
# ============================================================

y_true = np.array(
    all_targets
)

y_pred = np.array(
    all_predictions
)

probabilities = np.array(
    all_probabilities
)


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision_macro = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

precision_weighted = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"\nAccuracy          : {accuracy:.4f}"
    f"  ({accuracy * 100:.2f}%)"
)

print(
    f"Macro Precision   : {precision_macro:.4f}"
)

print(
    f"Macro Recall      : {recall_macro:.4f}"
)

print(
    f"Macro F1          : {f1_macro:.4f}"
)

print(
    f"Weighted Precision: {precision_weighted:.4f}"
)

print(
    f"Weighted Recall   : {recall_weighted:.4f}"
)

print(
    f"Weighted F1       : {f1_weighted:.4f}"
)

print(
    f"\nEvaluation time   : "
    f"{elapsed_time:.2f} seconds"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("PER-CLASS CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(

    y_true,

    y_pred,

    labels=list(range(num_classes)),

    target_names=classes,

    digits=4,

    zero_division=0
)

print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_file = (
    RESULTS_DIR
    / "classification_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PLANT DISEASE MODEL "
        "CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 70
        + "\n\n"
    )

    file.write(
        f"Model: {MODEL_NAME}\n"
    )

    file.write(
        f"Test images: {len(test_df)}\n"
    )

    file.write(
        f"Number of classes: "
        f"{num_classes}\n\n"
    )

    file.write(
        f"Accuracy: "
        f"{accuracy:.6f}\n"
    )

    file.write(
        f"Macro Precision: "
        f"{precision_macro:.6f}\n"
    )

    file.write(
        f"Macro Recall: "
        f"{recall_macro:.6f}\n"
    )

    file.write(
        f"Macro F1: "
        f"{f1_macro:.6f}\n"
    )

    file.write(
        f"Weighted Precision: "
        f"{precision_weighted:.6f}\n"
    )

    file.write(
        f"Weighted Recall: "
        f"{recall_weighted:.6f}\n"
    )

    file.write(
        f"Weighted F1: "
        f"{f1_weighted:.6f}\n\n"
    )

    file.write(
        report
    )


# ============================================================
# SAVE CSV CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(

    y_true,

    y_pred,

    labels=list(range(num_classes)),

    target_names=classes,

    output_dict=True,

    zero_division=0
)


report_df = pd.DataFrame(
    report_dict
).transpose()


report_csv = (
    RESULTS_DIR
    / "classification_report.csv"
)

report_df.to_csv(
    report_csv
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_true,

    y_pred,

    labels=list(range(num_classes))
)


# ============================================================
# SAVE CONFUSION MATRIX CSV
# ============================================================

cm_df = pd.DataFrame(

    cm,

    index=classes,

    columns=classes
)


cm_csv = (
    RESULTS_DIR
    / "confusion_matrix.csv"
)

cm_df.to_csv(
    cm_csv
)


# ============================================================
# CONFUSION MATRIX FIGURE
# ============================================================

plt.figure(
    figsize=(24, 20)
)

sns.heatmap(

    cm,

    annot=False,

    fmt="d",

    cmap="Blues",

    xticklabels=classes,

    yticklabels=classes,

    cbar=True
)


plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.title(
    "Plant Disease Classification "
    "Confusion Matrix"
)

plt.xticks(
    rotation=90,
    fontsize=8
)

plt.yticks(
    rotation=0,
    fontsize=8
)

plt.tight_layout()


cm_image = (
    RESULTS_DIR
    / "confusion_matrix.png"
)

plt.savefig(
    cm_image,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

cm_normalized = (
    cm.astype(float)
    /
    cm.sum(
        axis=1,
        keepdims=True
    )
)

# Avoid NaN if a class has zero samples
cm_normalized = np.nan_to_num(
    cm_normalized
)


plt.figure(
    figsize=(24, 20)
)

sns.heatmap(

    cm_normalized,

    annot=False,

    cmap="Blues",

    xticklabels=classes,

    yticklabels=classes,

    vmin=0,

    vmax=1,

    cbar=True
)


plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.title(
    "Normalized Plant Disease "
    "Confusion Matrix"
)

plt.xticks(
    rotation=90,
    fontsize=8
)

plt.yticks(
    rotation=0,
    fontsize=8
)

plt.tight_layout()


normalized_cm_image = (
    RESULTS_DIR
    / "confusion_matrix_normalized.png"
)

plt.savefig(
    normalized_cm_image,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE INDIVIDUAL PREDICTIONS
# ============================================================

predicted_class_names = [

    idx_to_class[
        prediction
    ]

    for prediction in y_pred

]


true_class_names = [

    idx_to_class[
        target
    ]

    for target in y_true

]


confidence_values = (
    probabilities.max(
        axis=1
    )
)


prediction_df = pd.DataFrame({

    "image_path":
        all_paths,

    "true_class":
        true_class_names,

    "predicted_class":
        predicted_class_names,

    "confidence":
        confidence_values,

    "correct":
        y_true == y_pred

})


prediction_csv = (
    RESULTS_DIR
    / "test_predictions.csv"
)

prediction_df.to_csv(
    prediction_csv,
    index=False
)


# ============================================================
# FIND MISCLASSIFIED IMAGES
# ============================================================

incorrect_df = prediction_df[
    prediction_df["correct"] == False
].copy()


incorrect_df = incorrect_df.sort_values(
    "confidence",
    ascending=False
)


incorrect_csv = (
    RESULTS_DIR
    / "misclassified_images.csv"
)

incorrect_df.to_csv(
    incorrect_csv,
    index=False
)


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

summary = {

    "model": MODEL_NAME,

    "test_images": int(
        len(test_df)
    ),

    "num_classes": int(
        num_classes
    ),

    "accuracy": float(
        accuracy
    ),

    "macro_precision": float(
        precision_macro
    ),

    "macro_recall": float(
        recall_macro
    ),

    "macro_f1": float(
        f1_macro
    ),

    "weighted_precision": float(
        precision_weighted
    ),

    "weighted_recall": float(
        recall_weighted
    ),

    "weighted_f1": float(
        f1_weighted
    ),

    "evaluation_time_seconds":
        float(elapsed_time),

    "incorrect_predictions":
        int(len(incorrect_df))

}


summary_file = (
    RESULTS_DIR
    / "evaluation_summary.json"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("\nFiles generated:")

print(
    f"1. {report_file}"
)

print(
    f"2. {report_csv}"
)

print(
    f"3. {cm_image}"
)

print(
    f"4. {normalized_cm_image}"
)

print(
    f"5. {cm_csv}"
)

print(
    f"6. {prediction_csv}"
)

print(
    f"7. {incorrect_csv}"
)

print(
    f"8. {summary_file}"
)

print("\nIncorrect predictions:")
print(
    len(incorrect_df)
)

print(
    f"\nFinal Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Final Test Macro F1: "
    f"{f1_macro:.4f}"
)

print("\nDone.")