from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(r"C:\smart_agriculture")

IMAGE_DIR = (
    ROOT
    / "ai"
    / "severity"
    / "dataset"
    / "images"
    / "annotation_pool"
)

CHECKPOINT = (
    ROOT
    / "ai"
    / "severity"
    / "models"
    / "sam2_repo"
    / "checkpoints"
    / "sam2.1_hiera_tiny.pt"
)

CONFIG = (
    ROOT
    / "ai"
    / "severity"
    / "models"
    / "sam2_repo"
    / "sam2"
    / "configs"
    / "sam2.1"
    / "sam2.1_hiera_t.yaml"
)

OUTPUT_DIR = (
    ROOT
    / "ai"
    / "severity"
    / "results"
    / "sam2_test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SELECT ACTUAL LEAF IMAGE
# ============================================================

IMAGE_NAME = (
    "leaf_1_0_8d70db5b-7916-4397-836d-6a73a2faaead"
    "___RS_Erly.B 6330.JPG"
)

IMAGE_PATH = IMAGE_DIR / IMAGE_NAME


# ============================================================
# CHECK FILES
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"SAM2 checkpoint not found:\n{CHECKPOINT}"
    )

if not CONFIG.exists():
    raise FileNotFoundError(
        f"SAM2 config not found:\n{CONFIG}"
    )


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("SAM2.1 TINY TEST")
print("=" * 60)

print("\nImage:")
print(IMAGE_PATH.name)

print("\nDevice:")
print(DEVICE)

if DEVICE == "cuda":
    print("GPU:")
    print(torch.cuda.get_device_name(0))


# ============================================================
# LOAD SAM2
# ============================================================

print("\nLoading SAM2.1 Tiny...")

sam2_model = build_sam2(
    str(CONFIG),
    str(CHECKPOINT),
    device=DEVICE,
)

predictor = SAM2ImagePredictor(sam2_model)

print("SAM2 loaded successfully.")


# ============================================================
# LOAD IMAGE
# ============================================================

print("\nLoading image...")

image_bgr = cv2.imread(str(IMAGE_PATH))

if image_bgr is None:
    raise RuntimeError(
        f"Could not read image:\n{IMAGE_PATH}"
    )

image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB,
)

height, width = image_rgb.shape[:2]

print(f"Image size: {width} x {height}")


# ============================================================
# SET IMAGE FOR SAM2
# ============================================================

print("\nPreparing image for SAM2...")

predictor.set_image(image_rgb)

print("Image prepared.")


# ============================================================
# POINT PROMPT
# ============================================================
#
# For this first experiment we use one point near
# the center of the image.
#
# This is NOT disease detection.
#
# We are only testing how SAM2 responds to a
# generic point prompt.
# ============================================================

input_point = np.array(
    [[width // 2, height // 2]],
    dtype=np.float32,
)

input_label = np.array(
    [1],
    dtype=np.int32,
)


print("\nPrompt point:")
print(input_point)


# ============================================================
# RUN SAM2
# ============================================================

print("\nRunning SAM2...")

masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

print("\nSAM2 generated:")
print(f"{len(masks)} candidate masks")


# ============================================================
# PROCESS RESULTS
# ============================================================

for i, (mask, score) in enumerate(
    zip(masks, scores)
):

    # Convert SAM2 output to a clean Boolean mask
    mask = np.asarray(mask).squeeze().astype(bool)

    print(
        f"\nMask {i}"
        f"\n  Score: {float(score):.4f}"
        f"\n  Shape: {mask.shape}"
        f"\n  Pixels: {int(mask.sum())}"
    )

    # --------------------------------------------------------
    # SAVE BINARY MASK
    # --------------------------------------------------------

    mask_image = (
        mask.astype(np.uint8) * 255
    )

    mask_path = (
        OUTPUT_DIR
        / f"sam2_mask_{i}.png"
    )

    cv2.imwrite(
        str(mask_path),
        mask_image,
    )

    print(
        f"  Saved mask: {mask_path.name}"
    )

    # --------------------------------------------------------
    # CREATE OVERLAY
    # --------------------------------------------------------

    overlay = image_rgb.copy()

    # Highlight segmented area
    overlay[mask] = (
        255,
        0,
        0,
    )

    # Blend original image and segmentation
    result = cv2.addWeighted(
        image_rgb,
        0.65,
        overlay,
        0.35,
        0,
    )

    result_bgr = cv2.cvtColor(
        result,
        cv2.COLOR_RGB2BGR,
    )

    overlay_path = (
        OUTPUT_DIR
        / f"sam2_overlay_{i}.jpg"
    )

    cv2.imwrite(
        str(overlay_path),
        result_bgr,
    )

    print(
        f"  Saved overlay: {overlay_path.name}"
    )


# ============================================================
# SAVE ORIGINAL IMAGE
# ============================================================

original_path = (
    OUTPUT_DIR
    / "original.jpg"
)

cv2.imwrite(
    str(original_path),
    image_bgr,
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("SAM2 TEST COMPLETE")
print("=" * 60)

print("\nResults saved in:")

print(OUTPUT_DIR)

print("\nFiles created:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(" ", file.name)

print("\nNext step:")
print(
    "Open the three SAM2 overlay images and inspect "
    "what SAM2 segmented."
)