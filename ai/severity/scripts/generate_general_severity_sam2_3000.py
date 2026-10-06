"""
generate_general_severity_sam2_3000.py

Automatic severity pseudo-label generation for the 3,000-image
general PlantVillage dataset using SAM2 Hiera-Tiny.

IMPORTANT:
The generated severity percentages are AUTOMATIC PSEUDO-LABELS.
They are not independently verified ground-truth measurements.
"""

# ============================================================
# 1. WINDOWS DLL SETUP
#    MUST COME BEFORE SAM2 IMPORTS
# ============================================================

import os
from pathlib import Path

if os.name == "nt":

    DLL_DIRS = [
        r"E:\smart_agriculture\smart_agriculture\.venv\Lib\site-packages\torch\lib",
        r"E:\bin\x64",
        r"E:\bin",
    ]

    for dll_dir in DLL_DIRS:
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)


# ============================================================
# 2. IMPORTS
# ============================================================

import csv
import json
import time
import warnings
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# ============================================================
# 3. PROJECT PATHS
# ============================================================

ROOT = Path(r"E:\smart_agriculture\smart_agriculture")

DATASET_DIR = (
    ROOT
    / "ai"
    / "severity"
    / "dataset"
    / "general_3000"
)

RESULTS_DIR = (
    ROOT
    / "ai"
    / "severity"
    / "results"
    / "general_severity_sam2_3000"
)

MASK_DIR = RESULTS_DIR / "masks"

OVERLAY_DIR = RESULTS_DIR / "overlays"

METADATA_CSV = DATASET_DIR / "metadata.csv"

CHECKPOINT_CSV = (
    RESULTS_DIR
    / "severity_labels_sam2_3000_checkpoint.csv"
)

FINAL_CSV = (
    RESULTS_DIR
    / "severity_labels_sam2_3000.csv"
)

SUMMARY_JSON = RESULTS_DIR / "summary.json"

SAM2_CHECKPOINT = (
    ROOT
    / "ai"
    / "severity"
    / "models"
    / "sam2_repo"
    / "checkpoints"
    / "sam2.1_hiera_tiny.pt"
)

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


# ============================================================
# 4. SAM2 SETTINGS
# ============================================================

MAX_IMAGE_SIDE = 1024

POINTS_PER_SIDE = 20

PRED_IOU_THRESHOLD = 0.72

STABILITY_SCORE_THRESHOLD = 0.72

MIN_MASK_REGION_AREA = 80

MIN_DISEASE_COMPONENT_AREA = 20

MAX_FINAL_SEVERITY = 70.0

MAX_SELECTED_PROPOSALS = 12

MASK_OVERLAP_THRESHOLD = 0.85


# ============================================================
# 5. OUTPUT COLUMNS
# ============================================================

CSV_COLUMNS = [
    "image_path",
    "filename",
    "plant",
    "disease",
    "severity",
    "severity_level",
    "leaf_pixels",
    "disease_pixels",
    "leaf_area_fraction",
    "disease_area_fraction_of_image",
    "num_sam_proposals",
    "num_ranked_proposals",
    "num_selected_proposals",
    "quality_flags",
    "status",
]


# ============================================================
# 6. BASIC FUNCTIONS
# ============================================================

def ensure_directories():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MASK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OVERLAY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def normalize_name(name):

    name = str(name)

    name = name.replace("___", " ")

    name = name.replace("_", " ")

    name = name.replace("/", " ")

    return " ".join(
        name.lower().split()
    )


def resize_image(image):

    h, w = image.shape[:2]

    longest = max(h, w)

    if longest <= MAX_IMAGE_SIDE:
        return image

    scale = MAX_IMAGE_SIDE / float(longest)

    new_w = int(w * scale)

    new_h = int(h * scale)

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


def binary_mask(mask):

    return (
        mask.astype(np.uint8) > 0
    ).astype(np.uint8) * 255


def clean_mask(
    mask,
    open_size=3,
    close_size=5
):

    mask = binary_mask(mask)

    if open_size > 1:

        kernel = np.ones(
            (open_size, open_size),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

    if close_size > 1:

        kernel = np.ones(
            (close_size, close_size),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

    return mask > 0


def normalize01(x):

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    low = np.percentile(x, 2)

    high = np.percentile(x, 98)

    if high <= low + 1e-6:

        return np.zeros_like(
            x,
            dtype=np.float32
        )

    result = (
        (x - low)
        / (high - low)
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# 7. LEAF MASK
# ============================================================

def estimate_leaf_mask(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    h = hsv[:, :, 0]

    s = hsv[:, :, 1]

    v = hsv[:, :, 2]

    L = lab[:, :, 0]

    a = lab[:, :, 1]

    b = lab[:, :, 2]

    green = (
        (h >= 25)
        & (h <= 100)
        & (s >= 35)
        & (v >= 35)
    )

    yellow = (
        (h >= 12)
        & (h <= 42)
        & (s >= 35)
        & (v >= 35)
    )

    brown = (
        (h >= 5)
        & (h <= 35)
        & (s >= 45)
        & (v >= 25)
        & (L < 210)
    )

    lab_leaf = (
        (a >= 105)
        & (a <= 155)
        & (b >= 105)
        & (b <= 175)
        & (L >= 35)
    )

    leaf = (
        green
        | yellow
        | brown
        | lab_leaf
    )

    leaf &= v > 25

    leaf &= ~(
        (s < 18)
        & (v > 215)
    )

    leaf = clean_mask(
        leaf,
        open_size=5,
        close_size=11
    )

    # Keep largest useful regions.
    mask = binary_mask(leaf)

    number, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    if number > 1:

        components = []

        for label in range(1, number):

            area = stats[
                label,
                cv2.CC_STAT_AREA
            ]

            if area >= 300:

                components.append(
                    (area, label)
                )

        components.sort(
            reverse=True
        )

        new_mask = np.zeros_like(mask)

        for _, label in components[:4]:

            new_mask[
                labels == label
            ] = 255

        leaf = new_mask > 0

    # Fallback when leaf mask is clearly too small.
    ratio = leaf.mean()

    if ratio < 0.10:

        B, G, R = cv2.split(image)

        foreground = (
            (
                np.maximum.reduce(
                    [B, G, R]
                ) < 245
            )
            |
            (
                np.max(
                    image,
                    axis=2
                )
                -
                np.min(
                    image,
                    axis=2
                )
                > 18
            )
        )

        foreground = clean_mask(
            foreground,
            open_size=5,
            close_size=9
        )

        if foreground.mean() > leaf.mean():

            leaf = foreground

    return leaf


# ============================================================
# 8. VISUAL DISEASE FEATURES
# ============================================================

def local_darkness(gray):

    blur = cv2.GaussianBlur(
        gray,
        (0, 0),
        9
    )

    darkness = (
        blur.astype(np.float32)
        -
        gray.astype(np.float32)
    )

    return normalize01(
        darkness
    )


def local_color_difference(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    blur = cv2.GaussianBlur(
        lab,
        (0, 0),
        9
    )

    difference = np.linalg.norm(
        lab - blur,
        axis=2
    )

    return normalize01(
        difference
    )


def texture_map(gray):

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    return normalize01(
        magnitude
    )


# ============================================================
# 9. DISEASE-SPECIFIC EVIDENCE
# ============================================================

def disease_evidence(
    image,
    disease,
    leaf_mask
):

    disease = normalize_name(
        disease
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    h = hsv[:, :, 0].astype(
        np.float32
    )

    s = hsv[:, :, 1].astype(
        np.float32
    )

    v = hsv[:, :, 2].astype(
        np.float32
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    darkness = local_darkness(
        gray
    )

    color_difference = (
        local_color_difference(
            image
        )
    )

    texture = texture_map(
        gray
    )

    brown = (
        (h >= 5)
        & (h <= 35)
        & (s > 40)
        & (v < 210)
    ).astype(np.float32)

    dark = (
        (v < 130)
        & (s > 25)
    ).astype(np.float32)

    yellow = (
        (h >= 18)
        & (h <= 42)
        & (s > 35)
        & (v > 45)
    ).astype(np.float32)

    red_brown = (
        ((h <= 15) | (h >= 170))
        & (s > 35)
        & (v < 215)
    ).astype(np.float32)

    pale = (
        (v > 135)
        & (s < 90)
    ).astype(np.float32)

    score = (
        0.28 * darkness
        + 0.25 * color_difference
        + 0.18 * texture
    )

    # --------------------------------------------------------
    # Tomato
    # --------------------------------------------------------

    if "bacterial spot" in disease:

        score += (
            0.24 * dark
            + 0.18 * brown
            + 0.10 * color_difference
        )

    elif "early blight" in disease:

        score += (
            0.30 * brown
            + 0.24 * darkness
            + 0.18 * texture
        )

    elif "late blight" in disease:

        score += (
            0.34 * brown
            + 0.26 * darkness
            + 0.16 * color_difference
        )

    elif "leaf mold" in disease:

        score += (
            0.25 * yellow
            + 0.22 * brown
            + 0.22 * color_difference
        )

    # --------------------------------------------------------
    # Corn
    # --------------------------------------------------------

    elif (
        "cercospora" in disease
        or "gray leaf spot" in disease
    ):

        score += (
            0.34 * darkness
            + 0.28 * color_difference
            + 0.20 * texture
        )

    elif "common rust" in disease:

        score += (
            0.32 * red_brown
            + 0.26 * brown
            + 0.18 * color_difference
        )

    elif "northern leaf blight" in disease:

        score += (
            0.32 * brown
            + 0.25 * darkness
            + 0.20 * texture
        )

    # --------------------------------------------------------
    # Potato
    # --------------------------------------------------------

    elif "potato" in disease:

        if "early blight" in disease:

            score += (
                0.32 * brown
                + 0.25 * darkness
                + 0.20 * texture
            )

        elif "late blight" in disease:

            score += (
                0.35 * brown
                + 0.25 * darkness
                + 0.18 * color_difference
            )

    # --------------------------------------------------------
    # Apple
    # --------------------------------------------------------

    elif "apple scab" in disease:

        score += (
            0.30 * brown
            + 0.25 * darkness
            + 0.20 * color_difference
        )

    elif "black rot" in disease:

        score += (
            0.38 * darkness
            + 0.25 * brown
            + 0.18 * color_difference
        )

    elif "cedar apple rust" in disease:

        score += (
            0.34 * yellow
            + 0.22 * brown
            + 0.20 * color_difference
        )

    # --------------------------------------------------------
    # Other possible PlantVillage classes
    # --------------------------------------------------------

    elif "septoria" in disease:

        score += (
            0.28 * dark
            + 0.22 * brown
            + 0.20 * texture
        )

    elif "spider mites" in disease:

        score += (
            0.30 * color_difference
            + 0.25 * pale
            + 0.20 * texture
        )

    elif "target spot" in disease:

        score += (
            0.28 * brown
            + 0.25 * darkness
            + 0.22 * texture
        )

    elif "mosaic" in disease:

        score += (
            0.32 * color_difference
            + 0.25 * yellow
            + 0.15 * texture
        )

    elif "yellow leaf curl" in disease:

        score += (
            0.32 * yellow
            + 0.25 * color_difference
            + 0.15 * pale
        )

    score = normalize01(
        score
    )

    # Remove obvious white background.
    background = (
        (v > 225)
        & (s < 20)
    )

    score[background] *= 0.10

    score[~leaf_mask] = 0.0

    score = cv2.GaussianBlur(
        score,
        (3, 3),
        0
    )

    return np.clip(
        score,
        0.0,
        1.0
    )


# ============================================================
# 10. MASK IOU
# ============================================================

def mask_iou(a, b):

    intersection = np.logical_and(
        a,
        b
    ).sum()

    union = np.logical_or(
        a,
        b
    ).sum()

    if union == 0:

        return 0.0

    return (
        intersection
        / float(union)
    )


# ============================================================
# 11. SCORE SAM2 PROPOSALS
# ============================================================

def proposal_score(
    mask,
    evidence,
    leaf_mask
):

    mask = (
        mask.astype(bool)
        & leaf_mask
    )

    area = int(
        mask.sum()
    )

    if area < MIN_DISEASE_COMPONENT_AREA:

        return -1.0, {}

    values = evidence[mask]

    if len(values) == 0:

        return -1.0, {}

    leaf_area = max(
        int(leaf_mask.sum()),
        1
    )

    leaf_fraction = (
        area
        / float(leaf_area)
    )

    mean_evidence = float(
        values.mean()
    )

    p75 = float(
        np.percentile(
            values,
            75
        )
    )

    p95 = float(
        np.percentile(
            values,
            95
        )
    )

    leaf_values = evidence[
        leaf_mask
    ]

    baseline = float(
        np.percentile(
            leaf_values,
            55
        )
    )

    relative = max(
        0.0,
        mean_evidence - baseline
    )

    # Penalize whole-leaf proposals.
    if leaf_fraction > 0.55:

        size_penalty = 0.60

    elif leaf_fraction > 0.35:

        size_penalty = 0.78

    elif leaf_fraction > 0.20:

        size_penalty = 0.92

    else:

        size_penalty = 1.0

    score = (
        0.35 * mean_evidence
        + 0.30 * p75
        + 0.20 * relative
        + 0.15 * p95
    )

    score *= size_penalty

    info = {
        "area": area,
        "leaf_fraction": leaf_fraction,
        "mean_evidence": mean_evidence,
        "p75": p75,
        "p95": p95,
        "relative": relative,
        "quality": score,
    }

    return float(score), info


# ============================================================
# 12. BUILD FINAL DISEASE MASK
# ============================================================

def build_disease_mask(
    image,
    disease,
    leaf_mask,
    sam_masks
):

    evidence = disease_evidence(
        image,
        disease,
        leaf_mask
    )

    leaf_values = evidence[
        leaf_mask
    ]

    if len(leaf_values):

        p65 = float(
            np.percentile(
                leaf_values,
                65
            )
        )

        p75 = float(
            np.percentile(
                leaf_values,
                75
            )
        )

        evidence_threshold = max(
            0.22,
            0.55 * p65
            + 0.45 * p75
        )

    else:

        evidence_threshold = 0.30

    candidates = []

    for proposal in sam_masks:

        mask = np.asarray(
            proposal["segmentation"]
        ).astype(bool)

        mask &= leaf_mask

        if mask.sum() < MIN_DISEASE_COMPONENT_AREA:

            continue

        values = evidence[
            mask
        ]

        if len(values) == 0:

            continue

        strong_fraction = float(
            np.mean(
                values
                >= evidence_threshold
            )
        )

        mean_value = float(
            values.mean()
        )

        if (
            strong_fraction < 0.08
            and mean_value < evidence_threshold
        ):

            continue

        quality, info = proposal_score(
            mask,
            evidence,
            leaf_mask
        )

        if quality <= 0:

            continue

        predicted_iou = float(
            proposal.get(
                "predicted_iou",
                0.0
            )
        )

        stability = float(
            proposal.get(
                "stability_score",
                0.0
            )
        )

        combined = (
            0.72 * quality
            + 0.14 * predicted_iou
            + 0.14 * stability
        )

        candidates.append(
            (
                combined,
                mask,
                info
            )
        )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    selected = []

    final_mask = np.zeros(
        leaf_mask.shape,
        dtype=bool
    )

    for score, mask, info in candidates:

        duplicate = False

        for _, old_mask, _ in selected:

            if (
                mask_iou(
                    mask,
                    old_mask
                )
                >= MASK_OVERLAP_THRESHOLD
            ):

                duplicate = True

                break

        if duplicate:

            continue

        selected.append(
            (
                score,
                mask,
                info
            )
        )

        final_mask |= mask

        if (
            len(selected)
            >= MAX_SELECTED_PROPOSALS
        ):

            break

    # Conservative fallback.
    if not selected:

        final_mask = (
            (evidence >= max(
                0.45,
                evidence_threshold
            ))
            & leaf_mask
        )

    # Pixel-level evidence gate.
    if final_mask.any():

        selected_values = evidence[
            final_mask
        ]

        gate = max(
            evidence_threshold * 0.65,
            float(
                np.percentile(
                    selected_values,
                    20
                )
            )
        )

        final_mask &= (
            evidence >= gate
        )

    final_mask = clean_mask(
        final_mask,
        open_size=3,
        close_size=5
    )

    final_mask &= leaf_mask

    return (
        final_mask,
        {
            "num_ranked_proposals":
                len(candidates),

            "num_selected_proposals":
                len(selected),

            "evidence_threshold":
                evidence_threshold,
        }
    )


# ============================================================
# 13. SEVERITY
# ============================================================

def calculate_severity(
    leaf_mask,
    disease_mask
):

    leaf_pixels = int(
        leaf_mask.sum()
    )

    if leaf_pixels == 0:

        return 0.0, 0, 0

    disease_mask = (
        disease_mask
        & leaf_mask
    )

    disease_pixels = int(
        disease_mask.sum()
    )

    severity = (
        disease_pixels
        / float(leaf_pixels)
        * 100.0
    )

    severity = float(
        np.clip(
            severity,
            0.0,
            MAX_FINAL_SEVERITY
        )
    )

    return (
        severity,
        disease_pixels,
        leaf_pixels
    )


def severity_level(severity):

    if severity <= 10:

        return "Mild"

    elif severity <= 25:

        return "Moderate"

    elif severity <= 50:

        return "Severe"

    return "Very Severe"


# ============================================================
# 14. QUALITY FLAGS
# ============================================================

def quality_flags(
    severity,
    leaf_pixels,
    disease_pixels,
    sam_count
):

    flags = []

    if leaf_pixels < 1000:

        flags.append(
            "small_leaf_area"
        )

    if sam_count == 0:

        flags.append(
            "no_sam_proposals"
        )

    if disease_pixels == 0:

        flags.append(
            "zero_disease_pixels"
        )

    if severity >= 45:

        flags.append(
            "high_severity_review"
        )

    if severity <= 0.2:

        flags.append(
            "very_low_severity"
        )

    if not flags:

        return "OK"

    return "|".join(flags)


# ============================================================
# 15. OVERLAY
# ============================================================

def create_overlay(
    image,
    leaf_mask,
    disease_mask,
    severity,
    disease
):

    overlay = image.copy()

    # Leaf contour.
    leaf_u8 = binary_mask(
        leaf_mask
    )

    contours, _ = cv2.findContours(
        leaf_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 0),
        1
    )

    # Disease region.
    disease_u8 = binary_mask(
        disease_mask
    )

    red = np.zeros_like(
        overlay
    )

    red[:, :, 2] = 255

    indices = disease_u8 > 0

    overlay[indices] = cv2.addWeighted(
        overlay[indices],
        0.45,
        red[indices],
        0.55,
        0
    )

    # Text box.
    cv2.rectangle(
        overlay,
        (5, 5),
        (500, 65),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        overlay,
        f"Severity: {severity:.2f}%",
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    label = normalize_name(
        disease
    ).title()

    cv2.putText(
        overlay,
        label[:55],
        (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return overlay


# ============================================================
# 16. CHECKPOINT
# ============================================================

def load_completed():

    completed = {}

    if not CHECKPOINT_CSV.exists():

        return completed

    try:

        df = pd.read_csv(
            CHECKPOINT_CSV
        )

        for _, row in df.iterrows():

            key = (
                str(row["filename"]),
                str(row["disease"])
            )

            completed[key] = (
                row.to_dict()
            )

    except Exception as e:

        print(
            f"[WARNING] "
            f"Could not read checkpoint: {e}"
        )

    return completed


def append_checkpoint(row):

    exists = CHECKPOINT_CSV.exists()

    with open(
        CHECKPOINT_CSV,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore"
        )

        if not exists:

            writer.writeheader()

        writer.writerow(row)


# ============================================================
# 17. LOAD SAM2
# ============================================================

def load_sam2():

    if not SAM2_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"SAM2 checkpoint not found:\n"
            f"{SAM2_CHECKPOINT}"
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)

    print(
        "GENERAL SEVERITY - SAM2 3000"
    )

    print("=" * 70)

    print(
        f"Dataset : {DATASET_DIR}"
    )

    print(
        f"Results : {RESULTS_DIR}"
    )

    print(
        f"Device  : {device}"
    )

    if device == "cuda":

        print(
            "GPU     : "
            + torch.cuda.get_device_name(0)
        )

        print(
            "CUDA    : "
            + str(torch.version.cuda)
        )

    print()

    print(
        "Loading SAM2..."
    )

    model = build_sam2(
        SAM2_CONFIG,
        str(SAM2_CHECKPOINT),
        device=device,
        apply_postprocessing=False
    )

    mask_generator = (
        SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=POINTS_PER_SIDE,
            pred_iou_thresh=PRED_IOU_THRESHOLD,
            stability_score_thresh=(
                STABILITY_SCORE_THRESHOLD
            ),
            min_mask_region_area=(
                MIN_MASK_REGION_AREA
            ),
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            box_nms_thresh=0.7,
            crop_nms_thresh=0.7
        )
    )

    print(
        "SAM2 loaded successfully."
    )

    print()

    return mask_generator


# ============================================================
# 18. PROCESS ONE IMAGE
# ============================================================

def process_image(
    row,
    mask_generator
):

    image_path = Path(
        str(row["image_path"])
    )

    filename = str(
        row["filename"]
    )

    plant = str(
        row["plant"]
    )

    disease = str(
        row["disease"]
    )

    if not image_path.exists():

        raise FileNotFoundError(
            str(image_path)
        )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"Could not read: "
            f"{image_path}"
        )

    image = resize_image(
        image
    )

    # --------------------------------------------------------
    # Leaf mask
    # --------------------------------------------------------

    leaf_mask = estimate_leaf_mask(
        image
    )

    if leaf_mask.sum() < 100:

        raise RuntimeError(
            "Leaf mask too small"
        )

    # --------------------------------------------------------
    # SAM2
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    sam_masks = (
        mask_generator.generate(
            rgb
        )
    )

    # --------------------------------------------------------
    # Disease mask
    # --------------------------------------------------------

    disease_mask, info = (
        build_disease_mask(
            image,
            disease,
            leaf_mask,
            sam_masks
        )
    )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    severity, disease_pixels, leaf_pixels = (
        calculate_severity(
            leaf_mask,
            disease_mask
        )
    )

    level = severity_level(
        severity
    )

    leaf_area_fraction = (
        leaf_pixels
        / float(
            image.shape[0]
            * image.shape[1]
        )
    )

    disease_image_fraction = (
        disease_pixels
        / float(
            image.shape[0]
            * image.shape[1]
        )
    )

    flags = quality_flags(
        severity,
        leaf_pixels,
        disease_pixels,
        len(sam_masks)
    )

    # --------------------------------------------------------
    # Save mask
    # --------------------------------------------------------

    clean_disease = (
        normalize_name(
            disease
        )
        .replace(" ", "_")
    )

    stem = Path(
        filename
    ).stem

    mask_name = (
        f"{plant}_"
        f"{clean_disease}_"
        f"{stem}.png"
    )

    overlay_name = (
        f"{plant}_"
        f"{clean_disease}_"
        f"{stem}.jpg"
    )

    cv2.imwrite(
        str(
            MASK_DIR / mask_name
        ),
        binary_mask(
            disease_mask
        )
    )

    # --------------------------------------------------------
    # Save overlay
    # --------------------------------------------------------

    overlay = create_overlay(
        image,
        leaf_mask,
        disease_mask,
        severity,
        disease
    )

    cv2.imwrite(
        str(
            OVERLAY_DIR / overlay_name
        ),
        overlay,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            92
        ]
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {

        "image_path":
            str(image_path),

        "filename":
            filename,

        "plant":
            plant,

        "disease":
            disease,

        "severity":
            round(
                severity,
                4
            ),

        "severity_level":
            level,

        "leaf_pixels":
            leaf_pixels,

        "disease_pixels":
            disease_pixels,

        "leaf_area_fraction":
            round(
                leaf_area_fraction,
                6
            ),

        "disease_area_fraction_of_image":
            round(
                disease_image_fraction,
                6
            ),

        "num_sam_proposals":
            len(sam_masks),

        "num_ranked_proposals":
            info[
                "num_ranked_proposals"
            ],

        "num_selected_proposals":
            info[
                "num_selected_proposals"
            ],

        "quality_flags":
            flags,

        "status":
            "ok"
    }


# ============================================================
# 19. SUMMARY
# ============================================================

def create_summary(
    dataframe,
    total,
    runtime
):

    successful = dataframe[
        dataframe["status"]
        .astype(str)
        .str.startswith("ok")
    ].copy()

    summary = {

        "total_dataset_images":
            int(total),

        "results_available":
            int(len(dataframe)),

        "successful":
            int(len(successful)),

        "runtime_minutes":
            round(
                runtime / 60,
                2
            ),

        "device":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU"
            ),

        "severity_definition":
            (
                "diseased leaf pixels / "
                "total leaf pixels * 100"
            ),

        "warning":
            (
                "Automatically generated "
                "SAM2 pseudo-labels; not "
                "independently verified "
                "ground truth."
            )
    }

    if len(successful):

        values = (
            successful[
                "severity"
            ]
            .astype(float)
            .to_numpy()
        )

        summary[
            "severity_statistics"
        ] = {

            "mean":
                round(
                    float(
                        values.mean()
                    ),
                    4
                ),

            "median":
                round(
                    float(
                        np.median(values)
                    ),
                    4
                ),

            "std":
                round(
                    float(
                        values.std()
                    ),
                    4
                ),

            "min":
                round(
                    float(
                        values.min()
                    ),
                    4
                ),

            "max":
                round(
                    float(
                        values.max()
                    ),
                    4
                ),

            "p25":
                round(
                    float(
                        np.percentile(
                            values,
                            25
                        )
                    ),
                    4
                ),

            "p75":
                round(
                    float(
                        np.percentile(
                            values,
                            75
                        )
                    ),
                    4
                )
        }

        summary[
            "severity_distribution"
        ] = {

            "0-5":
                int(
                    (
                        (values >= 0)
                        &
                        (values < 5)
                    ).sum()
                ),

            "5-10":
                int(
                    (
                        (values >= 5)
                        &
                        (values < 10)
                    ).sum()
                ),

            "10-20":
                int(
                    (
                        (values >= 10)
                        &
                        (values < 20)
                    ).sum()
                ),

            "20-30":
                int(
                    (
                        (values >= 20)
                        &
                        (values < 30)
                    ).sum()
                ),

            "30-50":
                int(
                    (
                        (values >= 30)
                        &
                        (values < 50)
                    ).sum()
                ),

            "50+":
                int(
                    (
                        values >= 50
                    ).sum()
                )
        }

        disease_summary = {}

        for disease, group in (
            successful.groupby(
                "disease"
            )
        ):

            vals = (
                group["severity"]
                .astype(float)
                .to_numpy()
            )

            disease_summary[
                disease
            ] = {

                "count":
                    int(len(vals)),

                "mean":
                    round(
                        float(
                            vals.mean()
                        ),
                        4
                    ),

                "median":
                    round(
                        float(
                            np.median(vals)
                        ),
                        4
                    ),

                "min":
                    round(
                        float(
                            vals.min()
                        ),
                        4
                    ),

                "max":
                    round(
                        float(
                            vals.max()
                        ),
                        4
                    )
            }

        summary[
            "disease_summary"
        ] = disease_summary

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )


# ============================================================
# 20. MAIN
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        message=".*cannot import name '_C'.*"
    )

    ensure_directories()

    if not METADATA_CSV.exists():

        raise FileNotFoundError(
            f"Metadata not found:\n"
            f"{METADATA_CSV}"
        )

    metadata = pd.read_csv(
        METADATA_CSV
    )

    required = {
        "image_path",
        "filename",
        "plant",
        "disease"
    }

    missing = (
        required
        - set(metadata.columns)
    )

    if missing:

        raise ValueError(
            "Missing metadata columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    metadata = (
        metadata
        .drop_duplicates(
            subset=[
                "filename",
                "disease"
            ]
        )
        .reset_index(drop=True)
    )

    total_images = len(
        metadata
    )

    print(
        f"Images in dataset: "
        f"{total_images}"
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    completed = (
        load_completed()
    )

    print(
        f"Existing checkpoint entries: "
        f"{len(completed)}"
    )

    print()

    # --------------------------------------------------------
    # SAM2
    # --------------------------------------------------------

    mask_generator = load_sam2()

    start_time = time.time()

    success = 0

    errors = 0

    skipped = 0

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------

    progress = tqdm(
        metadata.itertuples(
            index=False
        ),
        total=len(metadata),
        desc="Generating severity",
        unit="img"
    )

    for item in progress:

        row = item._asdict()

        key = (
            str(row["filename"]),
            str(row["disease"])
        )

        # Already processed.
        if key in completed:

            skipped += 1

            continue

        try:

            result = process_image(
                pd.Series(row),
                mask_generator
            )

            append_checkpoint(
                result
            )

            success += 1

            progress.set_postfix(
                success=success,
                errors=errors,
                skipped=skipped,
                severity=(
                    f"{result['severity']:.1f}%"
                )
            )

        except KeyboardInterrupt:

            print()

            print(
                "Process interrupted."
            )

            print(
                "Completed results are "
                "already stored in the "
                "checkpoint CSV."
            )

            break

        except Exception as e:

            errors += 1

            error_result = {

                "image_path":
                    str(row["image_path"]),

                "filename":
                    str(row["filename"]),

                "plant":
                    str(row["plant"]),

                "disease":
                    str(row["disease"]),

                "severity":
                    "",

                "severity_level":
                    "",

                "leaf_pixels":
                    "",

                "disease_pixels":
                    "",

                "leaf_area_fraction":
                    "",

                "disease_area_fraction_of_image":
                    "",

                "num_sam_proposals":
                    "",

                "num_ranked_proposals":
                    "",

                "num_selected_proposals":
                    "",

                "quality_flags":
                    "processing_error",

                "status":
                    (
                        "error: "
                        + str(e)
                    )
            }

            append_checkpoint(
                error_result
            )

            print(
                f"\n[ERROR] "
                f"{row['filename']}: "
                f"{e}"
            )

    runtime = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Build final CSV
    # --------------------------------------------------------

    if CHECKPOINT_CSV.exists():

        final_df = pd.read_csv(
            CHECKPOINT_CSV
        )

        final_df = (
            final_df
            .drop_duplicates(
                subset=[
                    "filename",
                    "disease"
                ],
                keep="last"
            )
        )

        final_df.to_csv(
            FINAL_CSV,
            index=False
        )

        create_summary(
            final_df,
            total_images,
            runtime
        )

    print()

    print("=" * 70)

    print(
        "SEVERITY GENERATION COMPLETE"
    )

    print("=" * 70)

    print(
        f"Total dataset images : "
        f"{total_images}"
    )

    print(
        f"Processed this run   : "
        f"{success}"
    )

    print(
        f"Skipped/resumed      : "
        f"{skipped}"
    )

    print(
        f"Errors               : "
        f"{errors}"
    )

    print(
        f"Runtime              : "
        f"{runtime / 60:.2f} minutes"
    )

    print()

    print(
        f"Checkpoint CSV:\n"
        f"{CHECKPOINT_CSV}"
    )

    print()

    print(
        f"Final CSV:\n"
        f"{FINAL_CSV}"
    )

    print()

    print(
        f"Masks:\n"
        f"{MASK_DIR}"
    )

    print()

    print(
        f"Overlays:\n"
        f"{OVERLAY_DIR}"
    )

    print()

    print(
        f"Summary:\n"
        f"{SUMMARY_JSON}"
    )

    print("=" * 70)


# ============================================================
# 21. RUN
# ============================================================

if __name__ == "__main__":

    main()