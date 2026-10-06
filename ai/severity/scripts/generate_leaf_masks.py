from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# PATHS
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

DISEASE_MASK_DIR = (
    ROOT
    / "ai"
    / "severity"
    / "results"
    / "sam2_annotations"
)

OUTPUT_DIR = (
    ROOT
    / "ai"
    / "severity"
    / "results"
    / "sam2_leaf_masks_v2"
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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("SAM2 LEAF SEGMENTATION V2")
print("=" * 70)

print("\nDevice:", DEVICE)

if DEVICE == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# FIND DISEASE MASKS
# ============================================================

disease_masks = sorted(
    DISEASE_MASK_DIR.glob(
        "*_disease_mask.png"
    )
)

if not disease_masks:
    raise RuntimeError(
        "No disease masks found."
    )

print(
    f"\nFound {len(disease_masks)} disease masks."
)


# ============================================================
# LOAD SAM2
# ============================================================

print("\nLoading SAM2.1 Tiny...")

model = build_sam2(
    str(CONFIG),
    str(CHECKPOINT),
    device=DEVICE,
)

predictor = SAM2ImagePredictor(
    model
)

print("SAM2 loaded successfully.")


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(mask_path):

    stem = mask_path.name.replace(
        "_disease_mask.png",
        ""
    )

    for extension in [
        ".JPG",
        ".jpg",
        ".JPEG",
        ".jpeg",
        ".PNG",
        ".png",
    ]:

        path = (
            IMAGE_DIR
            / f"{stem}{extension}"
        )

        if path.exists():
            return path

    return None


# ============================================================
# CREATE LEAF PROMPTS
# ============================================================

def create_leaf_prompts(
    width,
    height,
):

    # Positive point near the center.
    # PlantVillage leaf is generally centered.

    positive = np.array(
        [
            [width / 2, height / 2]
        ],
        dtype=np.float32,
    )

    # Negative points near corners.
    margin_x = max(
        5,
        int(width * 0.05)
    )

    margin_y = max(
        5,
        int(height * 0.05)
    )

    negative = np.array(
        [
            [margin_x, margin_y],
            [width - margin_x, margin_y],
            [margin_x, height - margin_y],
            [width - margin_x, height - margin_y],
        ],
        dtype=np.float32,
    )

    points = np.concatenate(
        [
            positive,
            negative,
        ],
        axis=0,
    )

    labels = np.array(
        [
            1,
            0,
            0,
            0,
            0,
        ],
        dtype=np.int32,
    )

    return points, labels


# ============================================================
# SELECT BEST LEAF MASK
# ============================================================

def select_leaf_mask(
    masks,
    scores,
    width,
    height,
):

    center_x = width // 2
    center_y = height // 2

    candidates = []

    image_area = width * height

    for i, (mask, score) in enumerate(
        zip(masks, scores)
    ):

        mask = (
            np.asarray(mask)
            .squeeze()
            .astype(bool)
        )

        area = int(mask.sum())

        if area == 0:
            continue

        area_ratio = (
            area / image_area
        )

        center_inside = bool(
            mask[
                min(center_y, height - 1),
                min(center_x, width - 1),
            ]
        )

        # Require the candidate to contain
        # the positive center point.
        if not center_inside:
            continue

        # Reject obviously huge/full-image masks.
        if area_ratio > 0.85:
            continue

        # Reject extremely tiny masks.
        if area_ratio < 0.05:
            continue

        candidates.append(
            (
                i,
                float(score),
                area,
                area_ratio,
            )
        )

    if not candidates:

        print(
            "WARNING: No ideal candidate found."
        )

        # Fallback: highest SAM score
        best_index = int(
            np.argmax(scores)
        )

        return (
            np.asarray(
                masks[best_index]
            )
            .squeeze()
            .astype(bool),
            best_index,
        )

    # Prefer highest SAM score among
    # geometrically reasonable candidates.

    best = max(
        candidates,
        key=lambda x: x[1],
    )

    best_index = best[0]

    return (
        np.asarray(
            masks[best_index]
        )
        .squeeze()
        .astype(bool),
        best_index,
    )


# ============================================================
# PROCESS
# ============================================================

for number, disease_mask_path in enumerate(
    disease_masks,
    start=1,
):

    print("\n")
    print("=" * 70)

    print(
        f"IMAGE {number}/{len(disease_masks)}"
    )

    print(
        disease_mask_path.name
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image_path = find_image(
        disease_mask_path
    )

    if image_path is None:

        print(
            "Image not found. Skipping."
        )

        continue

    image_bgr = cv2.imread(
        str(image_path)
    )

    if image_bgr is None:

        print(
            "Could not read image. Skipping."
        )

        continue

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    height, width = (
        image_rgb.shape[:2]
    )


    # --------------------------------------------------------
    # DISEASE MASK
    # --------------------------------------------------------

    disease_mask = cv2.imread(
        str(disease_mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if disease_mask is None:

        print(
            "Could not read disease mask."
        )

        continue

    disease_mask = (
        disease_mask > 0
    )


    # --------------------------------------------------------
    # SAM2
    # --------------------------------------------------------

    predictor.set_image(
        image_rgb
    )

    points, labels = (
        create_leaf_prompts(
            width,
            height,
        )
    )

    masks, scores, _ = (
        predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
    )


    # --------------------------------------------------------
    # PRINT CANDIDATES
    # --------------------------------------------------------

    print("\nCandidate leaf masks:")

    for i, (mask, score) in enumerate(
        zip(masks, scores)
    ):

        mask_bool = (
            np.asarray(mask)
            .squeeze()
            .astype(bool)
        )

        area = int(
            mask_bool.sum()
        )

        ratio = (
            area
            / (width * height)
            * 100
        )

        print(
            f"  Mask {i + 1}: "
            f"score={float(score):.4f}, "
            f"area={ratio:.2f}%"
        )


    # --------------------------------------------------------
    # SELECT MASK
    # --------------------------------------------------------

    leaf_mask, selected_index = (
        select_leaf_mask(
            masks,
            scores,
            width,
            height,
        )
    )


    print(
        f"\nSelected leaf mask: "
        f"{selected_index + 1}"
    )

    print(
        f"SAM2 score: "
        f"{float(scores[selected_index]):.4f}"
    )


    # --------------------------------------------------------
    # CLEAN LEAF MASK
    # --------------------------------------------------------

    leaf_uint8 = (
        leaf_mask.astype(np.uint8)
        * 255
    )

    # Morphological cleanup
    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    leaf_uint8 = cv2.morphologyEx(
        leaf_uint8,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    leaf_uint8 = cv2.morphologyEx(
        leaf_uint8,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )


    leaf_mask = (
        leaf_uint8 > 0
    )


    # --------------------------------------------------------
    # KEEP LARGEST CONNECTED COMPONENT
    # --------------------------------------------------------

    num_labels, labels_img, stats, _ = (
        cv2.connectedComponentsWithStats(
            leaf_uint8,
            connectivity=8,
        )
    )

    if num_labels > 1:

        largest_label = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        leaf_mask = (
            labels_img
            == largest_label
        )


    # --------------------------------------------------------
    # DISEASE INSIDE LEAF
    # --------------------------------------------------------

    disease_inside_leaf = (
        disease_mask
        & leaf_mask
    )


    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    leaf_pixels = int(
        leaf_mask.sum()
    )

    disease_pixels = int(
        disease_inside_leaf.sum()
    )

    if leaf_pixels > 0:

        severity = (
            disease_pixels
            / leaf_pixels
            * 100
        )

    else:

        severity = 0.0


    print(
        f"Leaf pixels: "
        f"{leaf_pixels}"
    )

    print(
        f"Disease pixels inside leaf: "
        f"{disease_pixels}"
    )

    print(
        f"Severity: "
        f"{severity:.2f}%"
    )


    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    if severity > 80:

        print(
            "\nWARNING:"
            "\nSeverity > 80%."
            "\nThis image should be visually checked."
        )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    stem = disease_mask_path.name.replace(
        "_disease_mask.png",
        ""
    )


    leaf_path = (
        OUTPUT_DIR
        / f"{stem}_leaf_mask.png"
    )

    disease_path = (
        OUTPUT_DIR
        / f"{stem}_disease_inside_leaf.png"
    )

    overlay_path = (
        OUTPUT_DIR
        / f"{stem}_overlay.jpg"
    )


    cv2.imwrite(
        str(leaf_path),
        (
            leaf_mask.astype(np.uint8)
            * 255
        ),
    )


    cv2.imwrite(
        str(disease_path),
        (
            disease_inside_leaf.astype(
                np.uint8
            )
            * 255
        ),
    )


    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    overlay = image_rgb.copy()


    # Disease = red
    overlay[
        disease_inside_leaf
    ] = (
        255,
        0,
        0,
    )


    # Leaf contour = green
    contours, _ = cv2.findContours(
        (
            leaf_mask.astype(np.uint8)
            * 255
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )


    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 0),
        2,
    )


    result = cv2.addWeighted(
        image_rgb,
        0.70,
        overlay,
        0.30,
        0,
    )


    result_bgr = cv2.cvtColor(
        result,
        cv2.COLOR_RGB2BGR,
    )


    cv2.imwrite(
        str(overlay_path),
        result_bgr,
    )


    print(
        "Saved:"
    )

    print(
        leaf_path.name
    )

    print(
        disease_path.name
    )

    print(
        overlay_path.name
    )


# ============================================================
# DONE
# ============================================================

print("\n")
print("=" * 70)
print("SAM2 LEAF SEGMENTATION V2 COMPLETE")
print("=" * 70)

print(
    "\nResults:"
)

print(
    OUTPUT_DIR
)

print(
    "\nOpen with:"
)

print(
    'explorer ".\\ai\\severity\\results\\sam2_leaf_masks_v2"'
)