import json
import sys
from pathlib import Path

import torch
import timm
import numpy as np

from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT_ROOT
    / "ai"
    / "disease"
    / "external_test"
    / "images"
)

MAPPING_FILE = (
    PROJECT_ROOT
    / "ai"
    / "disease"
    / "external_test"
    / "class_mapping.json"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "ai"
    / "disease"
    / "models"
    / "best_disease_model.pth"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "disease"
    / "results"
    / "external_test"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PLANTDOC EXTERNAL VALIDATION")
print("=" * 70)

print(f"Device       : {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU          : {torch.cuda.get_device_name(0)}")

print(f"Image folder : {IMAGE_DIR}")
print(f"Model        : {MODEL_FILE}")
print()

# ============================================================
# LOAD CLASS MAPPING
# ============================================================

with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    class_mapping = json.load(f)

print(f"Mapped PlantDoc classes: {len(class_mapping)}")

# ============================================================
# LOAD MODEL CLASSES
# ============================================================

classes_file = (
    PROJECT_ROOT
    / "ai"
    / "disease"
    / "results"
    / "classes.json"
)

with open(classes_file, "r", encoding="utf-8") as f:
    classes = json.load(f)

# Handle either list or dictionary format.
if isinstance(classes, dict):
    if "classes" in classes:
        classes = classes["classes"]
    else:
        classes = list(classes.values())

print(f"Model classes           : {len(classes)}")

# ============================================================
# LOAD MODEL
# ============================================================

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
# MODEL CLASS INDEX
# ============================================================

class_to_index = {
    name: index
    for index, name in enumerate(classes)
}

# ============================================================
# COLLECT DATA
# ============================================================

true_labels = []
pred_labels = []
confidences = []

accepted_true = []
accepted_pred = []

rejected = []

processed = 0
errors = 0

print()
print("=" * 70)
print("RUNNING EXTERNAL TEST")
print("=" * 70)

# ============================================================
# EVALUATION
# ============================================================

with torch.no_grad():

    for plantdoc_class, model_class in class_mapping.items():

        class_dir = IMAGE_DIR / plantdoc_class

        if not class_dir.exists():
            print(
                f"WARNING: Missing directory: {plantdoc_class}"
            )
            continue

        if model_class not in class_to_index:
            print(
                f"WARNING: Model class not found: {model_class}"
            )
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
                ".webp",
            }
        })
        for image_path in image_files:

            try:

                image = Image.open(image_path).convert("RGB")

                tensor = transform(image).unsqueeze(0)
                tensor = tensor.to(DEVICE)

                output = model(tensor)

                probabilities = torch.softmax(
                    output,
                    dim=1
                )[0]

                confidence, prediction = torch.max(
                    probabilities,
                    dim=0
                )

                confidence = confidence.item()
                prediction = prediction.item()

                true_labels.append(true_index)
                pred_labels.append(prediction)
                confidences.append(confidence)

                processed += 1

                # ------------------------------------------------
                # Confidence gate
                # ------------------------------------------------

                if confidence >= CONFIDENCE_THRESHOLD:

                    accepted_true.append(true_index)
                    accepted_pred.append(prediction)

                else:

                    rejected.append({
                        "image": str(image_path),
                        "true_class": model_class,
                        "predicted_class": classes[prediction],
                        "confidence": confidence
                    })

            except Exception as e:

                errors += 1

                print(
                    f"ERROR: {image_path}"
                )

                print(
                    f"       {e}"
                )

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    pred_labels
)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    pred_labels,
    average="macro",
    zero_division=0
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        pred_labels,
        average="weighted",
        zero_division=0
    )
)

# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("EXTERNAL VALIDATION RESULTS")
print("=" * 70)

print(f"Images processed       : {processed}")
print(f"Errors                 : {errors}")

print()
print(f"Accuracy               : {accuracy * 100:.2f}%")
print(f"Macro Precision        : {precision * 100:.2f}%")
print(f"Macro Recall           : {recall * 100:.2f}%")
print(f"Macro F1               : {f1 * 100:.2f}%")

print()
print(f"Weighted Precision     : {weighted_precision * 100:.2f}%")
print(f"Weighted Recall        : {weighted_recall * 100:.2f}%")
print(f"Weighted F1            : {weighted_f1 * 100:.2f}%")

# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

accepted_count = sum(
    c >= CONFIDENCE_THRESHOLD
    for c in confidences
)

rejected_count = sum(
    c < CONFIDENCE_THRESHOLD
    for c in confidences
)

print()
print("=" * 70)
print("CONFIDENCE GATE")
print("=" * 70)

print(
    f"Threshold              : "
    f"{CONFIDENCE_THRESHOLD * 100:.1f}%"
)

print(f"Accepted predictions   : {accepted_count}")
print(f"Rejected predictions   : {rejected_count}")

if processed > 0:

    print(
        f"Acceptance rate        : "
        f"{accepted_count / processed * 100:.2f}%"
    )

# ============================================================
# ACCEPTED-ONLY METRICS
# ============================================================

if len(accepted_true) > 0:

    accepted_accuracy = accuracy_score(
        accepted_true,
        accepted_pred
    )

    print()
    print("=" * 70)
    print("ACCEPTED-ONLY PERFORMANCE")
    print("=" * 70)

    print(
        f"Accepted images        : "
        f"{len(accepted_true)}"
    )

    print(
        f"Accuracy among accepted: "
        f"{accepted_accuracy * 100:.2f}%"
    )

else:

    print()
    print("No images passed the confidence threshold.")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

labels_used = sorted(set(true_labels))

target_names = [
    classes[index]
    for index in labels_used
]

report = classification_report(
    true_labels,
    pred_labels,
    labels=labels_used,
    target_names=target_names,
    zero_division=0
)

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    pred_labels,
    labels=labels_used
)

np.savetxt(
    RESULT_DIR / "external_confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)

# ============================================================
# SAVE REJECTED IMAGES
# ============================================================

with open(
    RESULT_DIR / "external_rejected.csv",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "image,true_class,predicted_class,confidence\n"
    )

    for item in rejected:

        f.write(
            f'"{item["image"]}",'
            f'"{item["true_class"]}",'
            f'"{item["predicted_class"]}",'
            f'{item["confidence"]:.6f}\n'
        )

# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "dataset": "PlantDoc",
    "images_processed": processed,
    "errors": errors,
    "accuracy": accuracy,
    "macro_precision": precision,
    "macro_recall": recall,
    "macro_f1": f1,
    "weighted_precision": weighted_precision,
    "weighted_recall": weighted_recall,
    "weighted_f1": weighted_f1,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "accepted_predictions": accepted_count,
    "rejected_predictions": rejected_count,
    "acceptance_rate": (
        accepted_count / processed
        if processed > 0 else 0
    )
}

with open(
    RESULT_DIR / "external_summary.json",
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
print("FILES SAVED")
print("=" * 70)

print(
    RESULT_DIR / "external_confusion_matrix.csv"
)

print(
    RESULT_DIR / "external_rejected.csv"
)

print(
    RESULT_DIR / "external_summary.json"
)

print("=" * 70)