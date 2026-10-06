from pathlib import Path
import json

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\smart_agriculture")

ANNOTATION_DIR = (
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
    / "sam2_disease_test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SELECT ONE OF YOUR 11 ANNOTATED IMAGES
# ============================================================

JSON_NAME = (
    "leaf_1_0_8d70db5b-7916-4397-836d-6a73a2faaead"
    "___RS_Erly.B 6330.json"
)

JSON_PATH = ANNOTATION_DIR / JSON_NAME


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(json_path):

    stem = json_path.stem

    candidates = []

    for extension in [
        ".JPG",
        ".jpg",
        ".JPEG",
        ".jpeg",
        ".PNG",
        ".png",
    ]:
        candidates.append(
            json_path.parent / f"{stem}{extension}"
        )

    for path in candidates:
        if path.exists():
            return path

    # Fallback: search by stem
    matches = list(
        json_path.parent.glob(f"{stem}.*")
    )

    for path in matches:
        if path.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
        ]:
            return path

    raise FileNotFoundError(
        f"Image corresponding to {json_path.name} "
        f"was not found."
    )


# ============================================================
# CHECK ANNOTATION
# ============================================================

if not JSON_PATH.exists():
    raise FileNotFoundError(
        f"Annotation not found:\n{JSON_PATH}"
    )

IMAGE_PATH = find_image(JSON_PATH)


# ============================================================
# LOAD LABELME JSON
# ============================================================

print("=" * 65)
print("SAM2 DISEASE SEGMENTATION TEST")
print("=" * 65)

print("\nAnnotation:")
print(JSON_PATH.name)

print("\nImage:")
print(IMAGE_PATH.name)


with open(
    JSON_PATH,
    "r",
    encoding="utf-8",
) as f:
    annotation = json.load(f)


# ============================================================
# LOAD IMAGE
# ============================================================

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

print(
    f"\nImage size: {width} x {height}"
)


# ============================================================
# CREATE GROUND-TRUTH MASKS FROM LABELME
# ============================================================

leaf_mask = np.zeros(
    (height, width),
    dtype=np.uint8,
)

disease_mask = np.zeros(
    (height, width),
    dtype=np.uint8,
)


for shape in annotation["shapes"]:

    label = shape["label"]

    points = np.array(
        shape["points"],
        dtype=np.int32,
    )

    if label == "leaf":

        cv2.fillPoly(
            leaf_mask,
            [points],
            1,
        )

    elif label == "disease":

        cv2.fillPoly(
            disease_mask,
            [points],
            1,
        )


# Make sure disease is inside the leaf
disease_mask = (
    disease_mask & leaf_mask
).astype(np.uint8)


# ============================================================
# CHECK ANNOTATIONS
# ============================================================

leaf_pixels = int(
    np.sum(leaf_mask)
)

disease_pixels = int(
    np.sum(disease_mask)
)

ground_truth_severity = (
    disease_pixels / leaf_pixels * 100
    if leaf_pixels > 0
    else 0
)

print("\nGround-truth pixels:")
print("  Leaf:", leaf_pixels)
print("  Disease:", disease_pixels)

print(
    f"  Severity: {ground_truth_severity:.2f}%"
)


# ============================================================
# CREATE POSITIVE / NEGATIVE POINTS
# ============================================================
#
# Positive points:
#     inside manually annotated disease
#
# Negative points:
#     inside healthy leaf
#
# These points are used ONLY for this experiment.
# They allow us to test how well SAM2 can refine
# a disease region when given a few hints.
# ============================================================

rng = np.random.default_rng(42)


positive_yx = np.argwhere(
    disease_mask == 1
)

healthy_leaf_mask = (
    (leaf_mask == 1)
    & (disease_mask == 0)
)

negative_yx = np.argwhere(
    healthy_leaf_mask
)


if len(positive_yx) == 0:
    raise RuntimeError(
        "No disease pixels found in annotation."
    )

if len(negative_yx) == 0:
    raise RuntimeError(
        "No healthy leaf pixels found in annotation."
    )


# Number of prompts
NUM_POSITIVE = min(5, len(positive_yx))
NUM_NEGATIVE = min(10, len(negative_yx))


positive_indices = rng.choice(
    len(positive_yx),
    size=NUM_POSITIVE,
    replace=False,
)

negative_indices = rng.choice(
    len(negative_yx),
    size=NUM_NEGATIVE,
    replace=False,
)


positive_points_yx = (
    positive_yx[positive_indices]
)

negative_points_yx = (
    negative_yx[negative_indices]
)


# Convert (y, x) -> (x, y)
positive_points = np.array(
    [
        [x, y]
        for y, x in positive_points_yx
    ],
    dtype=np.float32,
)

negative_points = np.array(
    [
        [x, y]
        for y, x in negative_points_yx
    ],
    dtype=np.float32,
)


input_points = np.concatenate(
    [
        positive_points,
        negative_points,
    ],
    axis=0,
)

input_labels = np.concatenate(
    [
        np.ones(
            len(positive_points),
            dtype=np.int32,
        ),
        np.zeros(
            len(negative_points),
            dtype=np.int32,
        ),
    ]
)


print("\nSAM2 prompts:")
print(
    "  Positive disease points:",
    len(positive_points),
)

print(
    "  Negative healthy-leaf points:",
    len(negative_points),
)


# ============================================================
# LOAD SAM2
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)

if device == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


print("\nLoading SAM2.1 Tiny...")

sam2_model = build_sam2(
    str(CONFIG),
    str(CHECKPOINT),
    device=device,
)

predictor = SAM2ImagePredictor(
    sam2_model
)

print("SAM2 loaded.")


# ============================================================
# SET IMAGE
# ============================================================

predictor.set_image(image_rgb)


# ============================================================
# RUN SAM2
# ============================================================

print("\nRunning SAM2 disease segmentation...")

masks, scores, _ = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True,
)


print(
    "\nSAM2 generated",
    len(masks),
    "candidate masks."
)


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_iou(
    prediction,
    ground_truth,
):

    intersection = np.logical_and(
        prediction,
        ground_truth,
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth,
    ).sum()

    if union == 0:
        return 0.0

    return intersection / union


def calculate_dice(
    prediction,
    ground_truth,
):

    intersection = np.logical_and(
        prediction,
        ground_truth,
    ).sum()

    total = (
        prediction.sum()
        + ground_truth.sum()
    )

    if total == 0:
        return 0.0

    return (
        2 * intersection / total
    )


# ============================================================
# SAVE GROUND TRUTH
# ============================================================

cv2.imwrite(
    str(
        OUTPUT_DIR
        / "ground_truth_disease.png"
    ),
    disease_mask * 255,
)


# ============================================================
# SAVE PROMPT VISUALIZATION
# ============================================================

prompt_image = image_rgb.copy()


# Positive points = green
for x, y in positive_points.astype(int):

    cv2.circle(
        prompt_image,
        (x, y),
        5,
        (0, 255, 0),
        -1,
    )


# Negative points = red
for x, y in negative_points.astype(int):

    cv2.circle(
        prompt_image,
        (x, y),
        5,
        (255, 0, 0),
        -1,
    )


prompt_bgr = cv2.cvtColor(
    prompt_image,
    cv2.COLOR_RGB2BGR,
)

cv2.imwrite(
    str(
        OUTPUT_DIR
        / "prompt_points.jpg"
    ),
    prompt_bgr,
)


# ============================================================
# EVALUATE EACH SAM2 MASK
# ============================================================

results = []


for i, (mask, score) in enumerate(
    zip(masks, scores)
):

    mask = np.asarray(
        mask
    ).squeeze().astype(bool)

    ground_truth = (
        disease_mask.astype(bool)
    )

    iou = calculate_iou(
        mask,
        ground_truth,
    )

    dice = calculate_dice(
        mask,
        ground_truth,
    )

    predicted_disease_pixels = int(
        mask.sum()
    )

    predicted_severity = (
        predicted_disease_pixels
        / leaf_pixels
        * 100
        if leaf_pixels > 0
        else 0
    )

    print("\n" + "-" * 50)

    print(f"Mask {i}")

    print(
        f"  SAM2 score: {float(score):.4f}"
    )

    print(
        f"  IoU: {iou:.4f}"
    )

    print(
        f"  Dice: {dice:.4f}"
    )

    print(
        f"  Predicted disease pixels: "
        f"{predicted_disease_pixels}"
    )

    print(
        f"  Predicted severity: "
        f"{predicted_severity:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE MASK
    # --------------------------------------------------------

    mask_uint8 = (
        mask.astype(np.uint8)
        * 255
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"sam2_disease_mask_{i}.png"
        ),
        mask_uint8,
    )

    # --------------------------------------------------------
    # CREATE OVERLAY
    # --------------------------------------------------------

    overlay = image_rgb.copy()

    overlay[mask] = (
        255,
        0,
        0,
    )

    result_image = cv2.addWeighted(
        image_rgb,
        0.65,
        overlay,
        0.35,
        0,
    )

    result_bgr = cv2.cvtColor(
        result_image,
        cv2.COLOR_RGB2BGR,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"sam2_disease_overlay_{i}.jpg"
        ),
        result_bgr,
    )

    results.append(
        {
            "mask": i,
            "sam2_score": float(score),
            "iou": float(iou),
            "dice": float(dice),
            "predicted_severity": float(
                predicted_severity
            ),
        }
    )


# ============================================================
# FIND BEST MASK FOR THIS EXPERIMENT
# ============================================================

best = max(
    results,
    key=lambda x: x["iou"],
)


print("\n" + "=" * 65)
print("BEST SAM2 RESULT")
print("=" * 65)

print(
    f"Mask: {best['mask']}"
)

print(
    f"IoU: {best['iou']:.4f}"
)

print(
    f"Dice: {best['dice']:.4f}"
)

print(
    f"Predicted severity: "
    f"{best['predicted_severity']:.2f}%"
)

print(
    f"Ground-truth severity: "
    f"{ground_truth_severity:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    OUTPUT_DIR
    / "results.txt"
)

with open(
    results_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "SAM2 Disease Segmentation Test\n"
    )

    f.write(
        f"Image: {IMAGE_PATH.name}\n"
    )

    f.write(
        f"Ground truth severity: "
        f"{ground_truth_severity:.2f}%\n\n"
    )

    for result in results:

        f.write(
            f"Mask {result['mask']}: "
            f""
            f"SAM2 score="
            f"{result['sam2_score']:.4f}, "
            f"IoU="
            f"{result['iou']:.4f}, "
            f"Dice="
            f"{result['dice']:.4f}, "
            f"Severity="
            f"{result['predicted_severity']:.2f}%\n"
        )

    f.write("\n")

    f.write(
        f"Best mask: {best['mask']}\n"
    )

    f.write(
        f"Best IoU: {best['iou']:.4f}\n"
    )

    f.write(
        f"Best Dice: {best['dice']:.4f}\n"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 65)
print("TEST COMPLETE")
print("=" * 65)

print("\nResults saved to:")

print(OUTPUT_DIR)

print("\nOpen the results folder with:")

print(
    'explorer ".\\ai\\severity\\results\\sam2_disease_test"'
)