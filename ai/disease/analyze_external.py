import json
from pathlib import Path

import torch
import timm
import numpy as np
import pandas as pd

from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "ai" / "disease" / "external_test" / "images"

MAPPING_FILE = PROJECT_ROOT / "ai" / "disease" / "external_test" / "class_mapping.json"

MODEL_FILE = PROJECT_ROOT / "ai" / "disease" / "models" / "best_disease_model.pth"

CLASSES_FILE = PROJECT_ROOT / "ai" / "disease" / "results" / "classes.json"

RESULT_DIR = PROJECT_ROOT / "ai" / "disease" / "results" / "external_test"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224

THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# LOAD MAPPING
# ============================================================

with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    mapping = json.load(f)

# ============================================================
# LOAD CLASSES
# ============================================================

with open(CLASSES_FILE, "r", encoding="utf-8") as f:
    classes_data = json.load(f)

if isinstance(classes_data, dict):
    if "classes" in classes_data:
        classes = classes_data["classes"]
    else:
        classes = list(classes_data.values())
else:
    classes = classes_data

class_to_index = {
    name: i
    for i, name in enumerate(classes)
}

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("EXTERNAL DATASET THRESHOLD ANALYSIS")
print("=" * 70)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print()
print("Loading model...")

model = timm.create_model(
    "tf_efficientnetv2_b0",
    pretrained=False,
    num_classes=len(classes)
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")

# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# COLLECT PREDICTIONS
# ============================================================

records = []

print()
print("=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

with torch.no_grad():

    for plantdoc_class, model_class in mapping.items():

        class_dir = IMAGE_DIR / plantdoc_class

        if not class_dir.exists():
            print(f"Missing: {class_dir}")
            continue

        true_index = class_to_index[model_class]

        image_files = sorted({
            p.resolve()
            for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            }
        })

        for image_path in image_files:

            try:

                image = Image.open(image_path).convert("RGB")

                x = transform(image).unsqueeze(0).to(DEVICE)

                output = model(x)

                probabilities = torch.softmax(
                    output,
                    dim=1
                )[0]

                confidence, prediction = torch.max(
                    probabilities,
                    dim=0
                )

                confidence = float(confidence.item())
                prediction = int(prediction.item())

                # A prediction is correct only if the predicted
                # model class exactly matches the mapped class.
                correct = prediction == true_index

                records.append({
                    "image": str(image_path),
                    "plantdoc_class": plantdoc_class,
                    "true_class": model_class,
                    "true_index": true_index,
                    "predicted_class": classes[prediction],
                    "predicted_index": prediction,
                    "confidence": confidence,
                    "correct": correct
                })

            except Exception as e:

                print(f"ERROR: {image_path}")
                print(e)

print()
print(f"Images analyzed: {len(records)}")

# ============================================================
# SAVE RAW PREDICTIONS
# ============================================================

df = pd.DataFrame(records)

prediction_file = RESULT_DIR / "external_predictions.csv"

df.to_csv(
    prediction_file,
    index=False
)

print(f"Predictions saved: {prediction_file}")

# ============================================================
# OVERALL METRICS
# ============================================================

true = df["true_index"].to_numpy()
pred = df["predicted_index"].to_numpy()

accuracy = accuracy_score(true, pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    true,
    pred,
    labels=sorted(df["true_index"].unique()),
    average="macro",
    zero_division=0
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        true,
        pred,
        labels=sorted(df["true_index"].unique()),
        average="weighted",
        zero_division=0
    )
)

print()
print("=" * 70)
print("CORRECTED EXTERNAL METRICS")
print("=" * 70)

print(f"Images                 : {len(df)}")
print(f"Accuracy               : {accuracy * 100:.2f}%")
print(f"Macro Precision        : {precision * 100:.2f}%")
print(f"Macro Recall           : {recall * 100:.2f}%")
print(f"Macro F1               : {f1 * 100:.2f}%")
print(f"Weighted Precision     : {weighted_precision * 100:.2f}%")
print(f"Weighted Recall        : {weighted_recall * 100:.2f}%")
print(f"Weighted F1            : {weighted_f1 * 100:.2f}%")

# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

threshold_results = []

print()
print("=" * 70)
print("CONFIDENCE THRESHOLD ANALYSIS")
print("=" * 70)

for threshold in THRESHOLDS:

    accepted = df[
        df["confidence"] >= threshold
    ]

    accepted_count = len(accepted)

    if accepted_count > 0:

        accepted_accuracy = accepted["correct"].mean()

    else:

        accepted_accuracy = np.nan

    acceptance_rate = (
        accepted_count / len(df)
        if len(df) > 0
        else 0
    )

    threshold_results.append({
        "threshold": threshold,
        "accepted_images": accepted_count,
        "rejected_images": len(df) - accepted_count,
        "acceptance_rate": acceptance_rate,
        "accepted_accuracy": accepted_accuracy
    })

    print(
        f"{threshold * 100:5.0f}% | "
        f"Accepted: {accepted_count:3d} | "
        f"Rejected: {len(df) - accepted_count:3d} | "
        f"Acceptance: {acceptance_rate * 100:6.2f}% | "
        f"Accepted accuracy: "
        f"{accepted_accuracy * 100 if not np.isnan(accepted_accuracy) else float('nan'):6.2f}%"
    )

threshold_df = pd.DataFrame(threshold_results)

threshold_file = (
    RESULT_DIR /
    "confidence_threshold_analysis.csv"
)

threshold_df.to_csv(
    threshold_file,
    index=False
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

mapped_indices = sorted(
    df["true_index"].unique()
)

cm = confusion_matrix(
    true,
    pred,
    labels=mapped_indices
)

cm_df = pd.DataFrame(
    cm,
    index=[classes[i] for i in mapped_indices],
    columns=[classes[i] for i in mapped_indices]
)

cm_file = (
    RESULT_DIR /
    "corrected_external_confusion_matrix.csv"
)

cm_df.to_csv(cm_file)

# ============================================================
# PER-CLASS METRICS
# ============================================================

precision_cls, recall_cls, f1_cls, support_cls = (
    precision_recall_fscore_support(
        true,
        pred,
        labels=mapped_indices,
        zero_division=0
    )
)

class_results = []

for i, class_index in enumerate(mapped_indices):

    class_results.append({
        "class": classes[class_index],
        "precision": precision_cls[i],
        "recall": recall_cls[i],
        "f1": f1_cls[i],
        "support": support_cls[i]
    })

class_df = pd.DataFrame(class_results)

class_file = (
    RESULT_DIR /
    "corrected_external_class_metrics.csv"
)

class_df.to_csv(
    class_file,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

summary = {
    "dataset": "PlantDoc",
    "images": len(df),
    "accuracy": float(accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(f1),
    "weighted_precision": float(weighted_precision),
    "weighted_recall": float(weighted_recall),
    "weighted_f1": float(weighted_f1),
    "thresholds": threshold_results
}

with open(
    RESULT_DIR / "corrected_external_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(prediction_file)
print(threshold_file)
print(cm_file)
print(class_file)
print(RESULT_DIR / "corrected_external_summary.json")

print("=" * 70)