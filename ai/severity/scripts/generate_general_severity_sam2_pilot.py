"""
generate_general_severity_sam2_pilot.py
=======================================

SAM2-based automatic severity pseudo-label generation.

CURRENT PILOT:
    12 disease classes
    5 images per class
    Total = 60 images

Pipeline:

    Plant image
          |
          v
    Leaf estimation
          |
          v
    SAM2 automatic mask proposals
          |
          v
    Disease-specific visual scoring
          |
          v
    Select disease-like regions
          |
          v
    Disease pixels / leaf pixels
          |
          v
    Severity percentage


IMPORTANT
---------
The generated severity values are AUTOMATIC PSEUDO-LABELS.

They are NOT manually verified ground-truth severity labels.

SAM2 is a general segmentation model. It does not inherently
understand Early Blight, Late Blight, Rust, Scab, etc.

Therefore:
    SAM2 proposals + disease-specific visual scoring
are used to estimate candidate disease regions.

DO NOT scale to 3000 images until the 60-image pilot has been
visually inspected.
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import sys
import json
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(
    r"E:\smart_agriculture\smart_agriculture"
)

# ------------------------------------------------------------
# CURRENT PILOT DATASET
# ------------------------------------------------------------

DATASET_DIR = (
    ROOT
    / "ai"
    / "severity"
    / "dataset"
    / "general_60"
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_DIR = (
    ROOT
    / "ai"
    / "severity"
    / "results"
    / "general_severity_sam2_pilot"
)

# ------------------------------------------------------------
# SAM2
# ------------------------------------------------------------

SAM2_REPO = (
    ROOT
    / "ai"
    / "severity"
    / "models"
    / "sam2_repo"
)

SAM2_CHECKPOINT = (
    SAM2_REPO
    / "checkpoints"
    / "sam2.1_hiera_tiny.pt"
)

SAM2_CONFIG = (
    "configs/sam2.1/sam2.1_hiera_t.yaml"
)


# ============================================================
# SAM2 SETTINGS
# ============================================================

# Maximum image dimension sent to SAM2.
# Lower values use less GPU memory.
MAX_IMAGE_SIDE = 1024

# Number of automatically generated points per image side.
POINTS_PER_SIDE = 16

# SAM2 mask quality threshold.
PRED_IOU_THRESHOLD = 0.70

# SAM2 mask stability threshold.
STABILITY_SCORE_THRESHOLD = 0.70

# Remove very small SAM2 regions.
MIN_MASK_REGION_AREA = 100

# Minimum connected disease component.
MIN_DISEASE_COMPONENT_AREA = 20

# Never allow automatic pseudo-label to exceed this.
MAX_SEVERITY_PERCENT = 85.0


# ============================================================
# IMPORT SAM2
# ============================================================

sys.path.insert(
    0,
    str(SAM2_REPO)
)

try:

    from sam2.build_sam import build_sam2

    from sam2.automatic_mask_generator import (
        SAM2AutomaticMaskGenerator
    )

except Exception as error:

    print("\n" + "=" * 70)
    print("ERROR: SAM2 IMPORT FAILED")
    print("=" * 70)

    print("\nMake sure SAM2 is installed.")

    print("\nOriginal error:")
    print(error)

    sys.exit(1)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# BASIC IMAGE FUNCTIONS
# ============================================================

def read_image(image_path):
    """
    Read image using OpenCV and convert BGR -> RGB.
    """

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            f"Could not read image:\n{image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


def resize_image(
    image,
    max_side=MAX_IMAGE_SIDE
):
    """
    Resize large images while preserving aspect ratio.
    """

    height, width = image.shape[:2]

    largest_side = max(
        height,
        width
    )

    if largest_side <= max_side:

        return image

    scale = (
        max_side
        / largest_side
    )

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    resized = cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )

    return resized


# ============================================================
# LEAF MASK
# ============================================================

def estimate_leaf_mask(image):
    """
    Estimate visible leaf region.

    This is NOT a trained leaf segmentation model.

    It is a conservative color-based foreground estimate
    used to prevent disease masks from covering obvious
    background pixels.
    """

    rgb = (
        image.astype(
            np.float32
        )
        / 255.0
    )

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # --------------------------------------------------------
    # Green vegetation score
    # --------------------------------------------------------

    green_score = (
        g
        - (r + b) / 2.0
    )

    leaf_mask = np.zeros(
        green_score.shape,
        dtype=np.uint8
    )

    green_region = (
        (green_score > 0.015)
        &
        (g > 0.18)
    )

    leaf_mask[
        green_region
    ] = 255

    # --------------------------------------------------------
    # Include yellow/brown leaf regions
    # because diseased leaf tissue may not be green.
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    yellow_brown_region = (
        (h >= 5)
        &
        (h <= 40)
        &
        (s > 35)
        &
        (v > 35)
    )

    leaf_mask[
        yellow_brown_region
    ] = 255

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    kernel = np.ones(
        (7, 7),
        dtype=np.uint8
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

    # --------------------------------------------------------
    # Keep large connected regions
    # --------------------------------------------------------

    number_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            leaf_mask
        )
    )

    if number_labels > 1:

        component_areas = (
            stats[
                1:,
                cv2.CC_STAT_AREA
            ]
        )

        order = np.argsort(
            component_areas
        )[::-1]

        clean_mask = np.zeros_like(
            leaf_mask
        )

        for index in order[:5]:

            label = index + 1

            area = component_areas[
                index
            ]

            if area > (
                0.02
                * leaf_mask.size
            ):

                clean_mask[
                    labels == label
                ] = 255

        leaf_mask = clean_mask

    # --------------------------------------------------------
    # Fallback if segmentation is too small
    # --------------------------------------------------------

    if np.sum(
        leaf_mask > 0
    ) < (
        0.05
        * leaf_mask.size
    ):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        _, leaf_mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU
        )

    return leaf_mask > 0


# ============================================================
# DISEASE NAME NORMALIZATION
# ============================================================

def normalize_disease_name(
    disease
):
    """
    Convert PlantVillage names such as:

        Tomato___Early_blight

    into:

        tomato early blight
    """

    normalized = (
        str(disease)
        .lower()
        .replace(
            "___",
            " "
        )
        .replace(
            "_",
            " "
        )
        .strip()
    )

    return normalized


# ============================================================
# DISEASE-SPECIFIC VISUAL SCORING
# ============================================================

def disease_score_map(
    image,
    leaf_mask,
    disease
):
    """
    Create a disease-likelihood map.

    Values:
        0.0 = weak disease evidence
        1.0 = strong disease evidence

    IMPORTANT:
        This is a heuristic visual score.

        It is NOT a trained disease segmentation model.
    """

    disease_name = normalize_disease_name(
        disease
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
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

    # --------------------------------------------------------
    # Basic visual maps
    # --------------------------------------------------------

    darkness = (
        1.0
        - v / 255.0
    )

    saturation = (
        s / 255.0
    )

    score = np.zeros_like(
        darkness,
        dtype=np.float32
    )

    # ========================================================
    # TOMATO
    # ========================================================

    if (
        "tomato" in disease_name
        and
        "bacterial spot" in disease_name
    ):

        dark_spot = (
            darkness
        )

        brown_region = (
            (h >= 5)
            &
            (h <= 35)
            &
            (s > 45)
            &
            (v < 190)
        )

        score = (
            0.45
            * dark_spot
            +
            0.55
            * brown_region.astype(
                np.float32
            )
        )

    elif (
        "tomato" in disease_name
        and
        "early blight" in disease_name
    ):

        brown_region = (
            (h >= 5)
            &
            (h <= 35)
            &
            (s > 40)
        )

        # Texture response
        texture = cv2.Laplacian(
            gray,
            cv2.CV_32F
        )

        texture = np.abs(
            texture
        )

        if texture.max() > 0:

            texture = (
                texture
                / texture.max()
            )

        score = (
            0.40
            * darkness
            +
            0.40
            * brown_region.astype(
                np.float32
            )
            +
            0.20
            * texture
        )

    elif (
        "tomato" in disease_name
        and
        "late blight" in disease_name
    ):

        brown_region = (
            (h >= 5)
            &
            (h <= 40)
            &
            (s > 35)
        )

        score = (
            0.55
            * darkness
            +
            0.45
            * brown_region.astype(
                np.float32
            )
        )

    elif (
        "tomato" in disease_name
        and
        "leaf mold" in disease_name
    ):

        yellow_brown = (
            (h >= 10)
            &
            (h <= 45)
            &
            (s > 30)
        )

        score = (
            0.50
            * darkness
            +
            0.50
            * yellow_brown.astype(
                np.float32
            )
        )

    # ========================================================
    # POTATO
    # ========================================================

    elif (
        "potato" in disease_name
        and
        "early blight" in disease_name
    ):

        brown_region = (
            (h >= 5)
            &
            (h <= 40)
            &
            (s > 35)
        )

        score = (
            0.55
            * darkness
            +
            0.45
            * brown_region.astype(
                np.float32
            )
        )

    elif (
        "potato" in disease_name
        and
        "late blight" in disease_name
    ):

        brown_region = (
            (h >= 5)
            &
            (h <= 45)
            &
            (s > 30)
        )

        score = (
            0.60
            * darkness
            +
            0.40
            * brown_region.astype(
                np.float32
            )
        )

    # ========================================================
    # CORN
    # ========================================================

    elif (
        "corn" in disease_name
        and
        "gray leaf spot" in disease_name
    ):

        low_saturation = (
            1.0
            - saturation
        )

        score = (
            0.60
            * darkness
            +
            0.40
            * low_saturation
        )

    elif (
        "corn" in disease_name
        and
        "common rust" in disease_name
    ):

        rust_region = (
            (h >= 0)
            &
            (h <= 25)
            &
            (s > 70)
            &
            (v > 40)
        )

        score = (
            rust_region.astype(
                np.float32
            )
        )

    elif (
        "corn" in disease_name
        and
        "northern leaf blight" in disease_name
    ):

        yellow_brown = (
            (h >= 10)
            &
            (h <= 50)
            &
            (s > 30)
        )

        score = (
            0.60
            * darkness
            +
            0.40
            * yellow_brown.astype(
                np.float32
            )
        )

    # ========================================================
    # APPLE
    # ========================================================

    elif (
        "apple" in disease_name
        and
        "scab" in disease_name
    ):

        brown_region = (
            (h >= 5)
            &
            (h <= 40)
            &
            (s > 35)
        )

        score = (
            0.55
            * darkness
            +
            0.45
            * brown_region.astype(
                np.float32
            )
        )

    elif (
        "apple" in disease_name
        and
        "black rot" in disease_name
    ):

        score = darkness

    elif (
        "apple" in disease_name
        and
        "cedar apple rust" in disease_name
    ):

        rust_region = (
            (h >= 0)
            &
            (h <= 30)
            &
            (s > 50)
        )

        score = (
            rust_region.astype(
                np.float32
            )
        )

    # ========================================================
    # FALLBACK
    # ========================================================

    else:

        score = darkness

    # --------------------------------------------------------
    # Restrict to leaf
    # --------------------------------------------------------

    score *= leaf_mask.astype(
        np.float32
    )

    score = np.clip(
        score,
        0.0,
        1.0
    )

    return score


# ============================================================
# SAM2 PROPOSAL QUALITY
# ============================================================

def proposal_quality(
    proposal_mask,
    leaf_mask,
    disease_scores
):
    """
    Calculate quality score for a SAM2 proposal.
    """

    area = int(
        np.sum(
            proposal_mask
        )
    )

    if area < MIN_DISEASE_COMPONENT_AREA:

        return -1.0

    # --------------------------------------------------------
    # Leaf overlap
    # --------------------------------------------------------

    leaf_overlap = int(
        np.sum(
            proposal_mask
            &
            leaf_mask
        )
    )

    if leaf_overlap < (
        MIN_DISEASE_COMPONENT_AREA
    ):

        return -1.0

    leaf_overlap_ratio = (
        leaf_overlap
        /
        max(area, 1)
    )

    if leaf_overlap_ratio < 0.50:

        return -1.0

    # --------------------------------------------------------
    # Disease evidence
    # --------------------------------------------------------

    values = disease_scores[
        proposal_mask
    ]

    if values.size == 0:

        return -1.0

    mean_disease_score = float(
        np.mean(values)
    )

    # --------------------------------------------------------
    # Relative size
    # --------------------------------------------------------

    leaf_area = max(
        int(np.sum(leaf_mask)),
        1
    )

    relative_area = (
        area
        /
        leaf_area
    )

    # Huge regions are suspicious.
    size_penalty = 0.0

    if relative_area > 0.40:

        size_penalty = (
            relative_area
            - 0.40
        ) * 1.5

    # --------------------------------------------------------
    # Final proposal score
    # --------------------------------------------------------

    quality = (
        0.75
        * mean_disease_score
        +
        0.25
        * leaf_overlap_ratio
        -
        size_penalty
    )

    return float(
        quality
    )


# ============================================================
# BUILD DISEASE MASK
# ============================================================

def build_disease_mask(
    image,
    leaf_mask,
    sam_proposals,
    disease
):
    """
    Select disease-like SAM2 proposals.
    """

    scores = disease_score_map(
        image=image,
        leaf_mask=leaf_mask,
        disease=disease
    )

    candidates = []

    for proposal in sam_proposals:

        mask = proposal[
            "segmentation"
        ].astype(bool)

        quality = proposal_quality(
            proposal_mask=mask,
            leaf_mask=leaf_mask,
            disease_scores=scores
        )

        if quality < 0:

            continue

        candidates.append(
            (
                quality,
                mask
            )
        )

    # --------------------------------------------------------
    # No valid proposals
    # --------------------------------------------------------

    if not candidates:

        return np.zeros_like(
            leaf_mask,
            dtype=bool
        )

    # --------------------------------------------------------
    # Strongest first
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    final_mask = np.zeros_like(
        leaf_mask,
        dtype=bool
    )

    # --------------------------------------------------------
    # Select strongest proposals
    # --------------------------------------------------------

    for quality, mask in candidates[:25]:

        area = int(
            np.sum(mask)
        )

        leaf_area = max(
            int(np.sum(leaf_mask)),
            1
        )

        relative_area = (
            area
            /
            leaf_area
        )

        # Avoid enormous SAM2 regions.
        if relative_area > 0.35:

            continue

        local_disease_score = float(
            np.mean(
                scores[mask]
            )
        )

        # Require reasonable disease evidence.
        if local_disease_score < 0.25:

            continue

        final_mask |= mask

    # --------------------------------------------------------
    # Restrict to leaf
    # --------------------------------------------------------

    final_mask &= leaf_mask

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    mask_u8 = (
        final_mask.astype(
            np.uint8
        )
        * 255
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_OPEN,
        kernel
    )

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # Remove tiny connected components
    # --------------------------------------------------------

    number_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask_u8
        )
    )

    cleaned = np.zeros_like(
        mask_u8
    )

    for label in range(
        1,
        number_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= MIN_DISEASE_COMPONENT_AREA:

            cleaned[
                labels == label
            ] = 255

    return cleaned > 0


# ============================================================
# SEVERITY CALCULATION
# ============================================================

def calculate_severity(
    disease_mask,
    leaf_mask
):
    """
    Severity percentage:

        diseased leaf pixels
        -------------------- × 100
          total leaf pixels
    """

    leaf_pixels = int(
        np.sum(leaf_mask)
    )

    if leaf_pixels == 0:

        return 0.0

    disease_pixels = int(
        np.sum(
            disease_mask
            &
            leaf_mask
        )
    )

    severity = (
        disease_pixels
        /
        leaf_pixels
    ) * 100.0

    severity = np.clip(
        severity,
        0.0,
        MAX_SEVERITY_PERCENT
    )

    return float(
        severity
    )


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    image,
    leaf_mask,
    disease_mask
):
    """
    Create visual QC overlay.

    Background -> darkened
    Disease    -> red
    """

    overlay = image.copy()

    # --------------------------------------------------------
    # Darken background
    # --------------------------------------------------------

    background = ~leaf_mask

    overlay[
        background
    ] = (
        overlay[
            background
        ].astype(
            np.float32
        )
        * 0.35
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Highlight disease in red
    # --------------------------------------------------------

    disease_pixels = disease_mask

    overlay[
        disease_pixels,
        0
    ] = 255

    overlay[
        disease_pixels,
        1
    ] = (
        overlay[
            disease_pixels,
            1
        ].astype(
            np.float32
        )
        * 0.25
    ).astype(
        np.uint8
    )

    overlay[
        disease_pixels,
        2
    ] = (
        overlay[
            disease_pixels,
            2
        ].astype(
            np.float32
        )
        * 0.25
    ).astype(
        np.uint8
    )

    return overlay


# ============================================================
# SAVE MASK
# ============================================================

def save_mask(
    output_path,
    mask
):
    """
    Save binary mask.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(output_path),
        (
            mask.astype(
                np.uint8
            )
            * 255
        )
    )


# ============================================================
# SAVE OVERLAY
# ============================================================

def save_overlay(
    output_path,
    overlay
):
    """
    Save RGB overlay as JPG.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    bgr = cv2.cvtColor(
        overlay,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(output_path),
        bgr
    )


# ============================================================
# INITIALIZE SAM2
# ============================================================

def initialize_sam2():
    """
    Load SAM2.1 Hiera-Tiny.
    """

    print("\n" + "=" * 70)
    print("LOADING SAM2")
    print("=" * 70)

    if not SAM2_CHECKPOINT.exists():

        raise FileNotFoundError(
            "SAM2 checkpoint not found:\n"
            f"{SAM2_CHECKPOINT}"
        )

    print(
        f"\nCheckpoint:\n"
        f"{SAM2_CHECKPOINT}"
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    sam2_model = build_sam2(
        SAM2_CONFIG,
        str(SAM2_CHECKPOINT),
        device=DEVICE
    )

    mask_generator = (
        SAM2AutomaticMaskGenerator(
            sam2_model,

            points_per_side=(
                POINTS_PER_SIDE
            ),

            pred_iou_thresh=(
                PRED_IOU_THRESHOLD
            ),

            stability_score_thresh=(
                STABILITY_SCORE_THRESHOLD
            ),

            min_mask_region_area=(
                MIN_MASK_REGION_AREA
            )
        )
    )

    print(
        "\nSAM2 loaded successfully."
    )

    return mask_generator


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    plant,
    disease,
    mask_generator
):
    """
    Process one image.
    """

    # --------------------------------------------------------
    # Read
    # --------------------------------------------------------

    image = read_image(
        image_path
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = resize_image(
        image
    )

    # --------------------------------------------------------
    # Leaf mask
    # --------------------------------------------------------

    leaf_mask = estimate_leaf_mask(
        image
    )

    # --------------------------------------------------------
    # SAM2 automatic proposals
    # --------------------------------------------------------

    sam_proposals = (
        mask_generator.generate(
            image
        )
    )

    # --------------------------------------------------------
    # Disease mask
    # --------------------------------------------------------

    disease_mask = build_disease_mask(
        image=image,
        leaf_mask=leaf_mask,
        sam_proposals=sam_proposals,
        disease=disease
    )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    severity = calculate_severity(
        disease_mask=disease_mask,
        leaf_mask=leaf_mask
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = create_overlay(
        image=image,
        leaf_mask=leaf_mask,
        disease_mask=disease_mask
    )

    return {
        "image": image_path.name,

        "plant": plant,

        "disease": disease,

        "leaf_pixels": int(
            np.sum(leaf_mask)
        ),

        "disease_pixels": int(
            np.sum(disease_mask)
        ),

        "severity_percent": float(
            severity
        ),

        "sam_proposals": int(
            len(sam_proposals)
        ),

        "overlay": overlay,

        "disease_mask": disease_mask
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print(
        "GENERAL SEVERITY — SAM2 AUTOMATIC PILOT"
    )
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "These severity values are automatic "
        "pseudo-labels, not verified ground truth."
    )

    print(
        f"\nDataset:\n{DATASET_DIR}"
    )

    print(
        f"\nOutput:\n{OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    metadata_path = (
        DATASET_DIR
        / "metadata.csv"
    )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"\nMetadata not found:\n"
            f"{metadata_path}"
        )

    metadata = pd.read_csv(
        metadata_path
    )

    print(
        f"\nImages in metadata: "
        f"{len(metadata)}"
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = {
        "filename",
        "plant",
        "disease"
    }

    missing_columns = (
        required_columns
        -
        set(metadata.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing metadata columns: "
            f"{sorted(missing_columns)}"
        )

    # --------------------------------------------------------
    # Initialize SAM2
    # --------------------------------------------------------

    mask_generator = (
        initialize_sam2()
    )

    # --------------------------------------------------------
    # Output folders
    # --------------------------------------------------------

    mask_root = (
        OUTPUT_DIR
        / "masks"
    )

    overlay_root = (
        OUTPUT_DIR
        / "overlays"
    )

    mask_root.mkdir(
        parents=True,
        exist_ok=True
    )

    overlay_root.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    results = []

    for _, row in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Generating SAM2 pseudo-labels"
    ):

        plant = str(
            row["plant"]
        )

        disease = str(
            row["disease"]
        )

        filename = str(
            row["filename"]
        )

        # ----------------------------------------------------
        # Actual pilot image path
        #
        # general_60/
        #     Tomato/
        #         Tomato___Early_blight/
        #             image.jpg
        # ----------------------------------------------------

        image_path = (
            DATASET_DIR
            / plant
            / disease
            / filename
        )

        if not image_path.exists():

            print(
                f"\nWARNING: image not found:"
                f"\n{image_path}"
            )

            continue

        try:

            result = process_image(
                image_path=image_path,
                plant=plant,
                disease=disease,
                mask_generator=mask_generator
            )

            # ------------------------------------------------
            # Safe folder names
            # ------------------------------------------------

            safe_plant = (
                plant
                .replace(
                    "/",
                    "_"
                )
                .replace(
                    "\\",
                    "_"
                )
            )

            safe_disease = (
                disease
                .replace(
                    "/",
                    "_"
                )
                .replace(
                    "\\",
                    "_"
                )
            )

            # ------------------------------------------------
            # Save mask
            # ------------------------------------------------

            mask_path = (
                mask_root
                / safe_plant
                / safe_disease
                / f"{Path(filename).stem}.png"
            )

            save_mask(
                mask_path,
                result[
                    "disease_mask"
                ]
            )

            # ------------------------------------------------
            # Save overlay
            # ------------------------------------------------

            overlay_path = (
                overlay_root
                / safe_plant
                / safe_disease
                / f"{Path(filename).stem}.jpg"
            )

            save_overlay(
                overlay_path,
                result[
                    "overlay"
                ]
            )

            # ------------------------------------------------
            # Store CSV result
            # ------------------------------------------------

            results.append(
                {
                    "image": filename,

                    "plant": plant,

                    "disease": disease,

                    "leaf_pixels": (
                        result[
                            "leaf_pixels"
                        ]
                    ),

                    "disease_pixels": (
                        result[
                            "disease_pixels"
                        ]
                    ),

                    "severity_percent": (
                        result[
                            "severity_percent"
                        ]
                    ),

                    "sam_proposals": (
                        result[
                            "sam_proposals"
                        ]
                    )
                }
            )

        except Exception as error:

            print(
                f"\nERROR processing:"
                f"\n{image_path}"
            )

            print(
                f"Error: {error}"
            )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:

        raise RuntimeError(
            "No images were successfully processed."
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / "severity_labels_sam2_pilot.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    print("\n" + "=" * 70)
    print("SAM2 PILOT RESULTS")
    print("=" * 70)

    print(
        f"\nSuccessfully processed:"
        f" {len(results_df)}"
    )

    print(
        f"\nMean severity:"
        f" {results_df['severity_percent'].mean():.2f}%"
    )

    print(
        f"Median severity:"
        f" {results_df['severity_percent'].median():.2f}%"
    )

    print(
        f"Std deviation:"
        f" {results_df['severity_percent'].std():.2f}%"
    )

    print(
        f"Minimum:"
        f" {results_df['severity_percent'].min():.2f}%"
    )

    print(
        f"Maximum:"
        f" {results_df['severity_percent'].max():.2f}%"
    )

    # ========================================================
    # DISEASE-WISE STATISTICS
    # ========================================================

    print(
        "\nDisease-wise statistics:"
    )

    disease_statistics = (
        results_df
        .groupby(
            [
                "plant",
                "disease"
            ]
        )[
            "severity_percent"
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max"
            ]
        )
    )

    print(
        disease_statistics.to_string()
    )

    # ========================================================
    # SEVERITY BINS
    # ========================================================

    bins = [
        0,
        5,
        10,
        20,
        30,
        50,
        100
    ]

    labels = [
        "0-5",
        "5-10",
        "10-20",
        "20-30",
        "30-50",
        "50+"
    ]

    results_df[
        "severity_range"
    ] = pd.cut(
        results_df[
            "severity_percent"
        ],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    print(
        "\nSeverity distribution:"
    )

    print(
        results_df[
            "severity_range"
        ].value_counts(
            sort=False
        )
    )

    # Save CSV again with severity range.
    results_df.to_csv(
        csv_path,
        index=False
    )

    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {

        "method":
            "SAM2 automatic mask proposals "
            "+ disease-specific visual scoring",

        "dataset":
            "general_60",

        "total_images":
            int(len(metadata)),

        "processed_images":
            int(len(results_df)),

        "num_disease_classes":
            int(
                metadata[
                    "disease"
                ].nunique()
            ),

        "images_per_class":
            5,

        "device":
            DEVICE,

        "sam2_checkpoint":
            str(
                SAM2_CHECKPOINT
            ),

        "points_per_side":
            POINTS_PER_SIDE,

        "pred_iou_threshold":
            PRED_IOU_THRESHOLD,

        "stability_score_threshold":
            STABILITY_SCORE_THRESHOLD,

        "min_mask_region_area":
            MIN_MASK_REGION_AREA,

        "mean_severity_percent":
            float(
                results_df[
                    "severity_percent"
                ].mean()
            ),

        "median_severity_percent":
            float(
                results_df[
                    "severity_percent"
                ].median()
            ),

        "std_severity_percent":
            float(
                results_df[
                    "severity_percent"
                ].std()
            ),

        "min_severity_percent":
            float(
                results_df[
                    "severity_percent"
                ].min()
            ),

        "max_severity_percent":
            float(
                results_df[
                    "severity_percent"
                ].max()
            ),

        "pseudo_label_warning":
            "These are automatic pseudo-labels "
            "and are not independently verified "
            "ground-truth severity measurements.",

        "csv":
            str(
                csv_path
            )
    }

    summary_path = (
        OUTPUT_DIR
        / "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("PILOT COMPLETE")
    print("=" * 70)

    print(
        f"\nCSV:"
        f"\n{csv_path}"
    )

    print(
        f"\nMasks:"
        f"\n{mask_root}"
    )

    print(
        f"\nOverlays:"
        f"\n{overlay_root}"
    )

    print(
        f"\nSummary:"
        f"\n{summary_path}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do NOT generate labels for all 3000 images yet."
    )

    print(
        "First inspect the 60-image masks and overlays."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()