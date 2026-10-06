from pathlib import Path
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ORIGINAL_DIR = (
    PROJECT_ROOT
    / "PlantVillage-Dataset"
    / "raw"
    / "color"
    / "Tomato___Early_blight"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "leaf_mask_test_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND 9436 IMAGE
# ============================================================

matches = list(
    ORIGINAL_DIR.glob("*6873f362*9436*")
)

if not matches:
    raise FileNotFoundError(
        "Could not find the 9436 original image."
    )

image_path = matches[0]

print("=" * 70)
print("WHOLE-LEAF MASK V2 TEST")
print("=" * 70)
print()
print("Image:")
print(image_path)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(image_path)
)

if image is None:
    raise RuntimeError(
        "Could not read image."
    )

h, w = image.shape[:2]

print(
    f"Image size: {w} x {h}"
)


# ============================================================
# GRABCUT INITIALIZATION
# ============================================================
#
# The leaf is roughly the central foreground object.
#
# We mark:
#   - outer border = definite background
#   - central region = probable foreground
#
# GrabCut then estimates the actual foreground.
# ============================================================

mask = np.zeros(
    (h, w),
    np.uint8
)

# Definite background
mask[:] = cv2.GC_BGD

# Create a conservative probable-foreground rectangle.
#
# The leaf occupies the central portion of this image.
margin_x = int(w * 0.08)
margin_y = int(h * 0.08)

rect = (
    margin_x,
    margin_y,
    w - 2 * margin_x,
    h - 2 * margin_y
)

# Mark rectangle as probable foreground
mask[
    margin_y:h - margin_y,
    margin_x:w - margin_x
] = cv2.GC_PR_FGD


# ============================================================
# GRABCUT
# ============================================================

bgd_model = np.zeros(
    (1, 65),
    np.float64
)

fgd_model = np.zeros(
    (1, 65),
    np.float64
)

print()
print("Running GrabCut...")

cv2.grabCut(
    image,
    mask,
    rect,
    bgd_model,
    fgd_model,
    8,
    cv2.GC_INIT_WITH_RECT
)


# ============================================================
# CONVERT TO BINARY MASK
# ============================================================

leaf_mask = np.where(
    (
        (mask == cv2.GC_FGD)
        |
        (mask == cv2.GC_PR_FGD)
    ),
    255,
    0
).astype(np.uint8)


# ============================================================
# MORPHOLOGICAL CLEANUP
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (7, 7)
)

leaf_mask = cv2.morphologyEx(
    leaf_mask,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

leaf_mask = cv2.morphologyEx(
    leaf_mask,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)


# ============================================================
# KEEP LARGEST CONNECTED COMPONENT
# ============================================================

num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        leaf_mask,
        connectivity=8
    )
)

if num_labels > 1:

    largest_label = (
        1
        + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )
    )

    leaf_mask = np.where(
        labels == largest_label,
        255,
        0
    ).astype(np.uint8)


# ============================================================
# AREA STATISTICS
# ============================================================

leaf_pixels = int(
    np.sum(leaf_mask > 0)
)

total_pixels = h * w

leaf_percentage = (
    leaf_pixels
    / total_pixels
    * 100
)

print()
print(
    f"Detected leaf area: "
    f"{leaf_percentage:.2f}%"
)


# ============================================================
# SAVE BINARY MASK
# ============================================================

mask_path = (
    OUTPUT_DIR
    / "9436_leaf_mask_v2.png"
)

cv2.imwrite(
    str(mask_path),
    leaf_mask
)


# ============================================================
# CREATE VISUAL OVERLAY
# ============================================================

overlay = image.copy()

leaf_bool = (
    leaf_mask > 0
)

# Semi-transparent green overlay
green_layer = np.zeros_like(
    image
)

green_layer[:] = (
    0,
    255,
    0
)

overlay[leaf_bool] = cv2.addWeighted(
    image[leaf_bool],
    0.55,
    green_layer[leaf_bool],
    0.45,
    0
)


# Leaf boundary
contours, _ = cv2.findContours(
    leaf_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

cv2.drawContours(
    overlay,
    contours,
    -1,
    (0, 255, 0),
    2
)


# Text
text = (
    f"Leaf area: "
    f"{leaf_percentage:.2f}%"
)

cv2.rectangle(
    overlay,
    (5, 5),
    (300, 42),
    (0, 0, 0),
    -1
)

cv2.putText(
    overlay,
    text,
    (12, 31),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)


overlay_path = (
    OUTPUT_DIR
    / "9436_leaf_mask_v2_overlay.jpg"
)

cv2.imwrite(
    str(overlay_path),
    overlay,
    [cv2.IMWRITE_JPEG_QUALITY, 95]
)


# ============================================================
# SAVE SIDE-BY-SIDE COMPARISON
# ============================================================

mask_display = cv2.cvtColor(
    leaf_mask,
    cv2.COLOR_GRAY2BGR
)

comparison = np.hstack(
    [
        image,
        overlay,
        mask_display
    ]
)

comparison_path = (
    OUTPUT_DIR
    / "9436_leaf_mask_v2_comparison.jpg"
)

cv2.imwrite(
    str(comparison_path),
    comparison,
    [cv2.IMWRITE_JPEG_QUALITY, 95]
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)

print()
print("Binary mask:")
print(mask_path)

print()
print("Overlay:")
print(overlay_path)

print()
print("Comparison:")
print(comparison_path)

print()
print("=" * 70)