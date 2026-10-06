from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"E:\smart_agriculture\smart_agriculture")

IMAGE_DIR = (
    ROOT / "ai" / "severity" / "dataset"
    / "external_test"
)

MODEL_PATH = (
    ROOT / "ai" / "severity" / "results"
    / "severity_model_250_final"
    / "best_severity_model_250.pth"
)

OUT_DIR = (
    ROOT / "ai" / "severity" / "results"
    / "external_test_predictions"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("EXTERNAL SEVERITY TEST")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# MODEL
# ============================================================

model = efficientnet_b0(
    weights=None
)

in_features = (
    model.classifier[1].in_features
)

model.classifier = nn.Sequential(
    nn.Dropout(0.30),
    nn.Linear(in_features, 1),
    nn.Sigmoid()
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


print(
    f"Loaded model from:\n{MODEL_PATH}"
)


# ============================================================
# FIND IMAGES
# ============================================================

extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.JPG",
    "*.JPEG",
    "*.PNG"
]

image_paths = []

for ext in extensions:
    image_paths.extend(
        IMAGE_DIR.glob(ext)
    )

# Remove duplicates
image_paths = sorted(
    set(image_paths)
)

print(
    f"\nExternal images found: {len(image_paths)}"
)

if len(image_paths) == 0:

    raise RuntimeError(
        f"No images found in:\n{IMAGE_DIR}"
    )


# ============================================================
# PREDICT
# ============================================================

results = []

with torch.no_grad():

    for image_path in image_paths:

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            tensor = transform(
                image
            ).unsqueeze(0)

            tensor = tensor.to(
                DEVICE
            )

            output = model(
                tensor
            ).item()

            severity = (
                float(output) * 100.0
            )

            # Safety clipping
            severity = max(
                0.0,
                min(100.0, severity)
            )

            results.append({
                "filename": image_path.name,
                "predicted_severity": severity
            })

            print(
                f"{image_path.name:60s} "
                f"-> {severity:6.2f}%"
            )

        except Exception as e:

            print(
                f"ERROR: {image_path.name}: {e}"
            )


# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(
    results
)

csv_path = (
    OUT_DIR /
    "external_severity_predictions.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# SEVERITY CATEGORY
# ============================================================

def severity_category(x):

    if x < 5:
        return "Very Low"

    elif x < 15:
        return "Low"

    elif x < 30:
        return "Moderate"

    elif x < 50:
        return "High"

    else:
        return "Very High"


df["severity_category"] = (
    df["predicted_severity"]
    .apply(severity_category)
)

df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("EXTERNAL TEST RESULTS")
print("=" * 70)

print(
    df[
        [
            "filename",
            "predicted_severity",
            "severity_category"
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(csv_path)