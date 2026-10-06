"""
GENERAL SEVERITY V3 — 60 IMAGE AUTOMATIC PILOT

Purpose:
    Generate automatic leaf masks, disease masks and pseudo-severity
    labels for the 60-image pilot dataset.

IMPORTANT:
    These are PSEUDO-LABELS, NOT manually verified ground truth.

V3:
    - Keeps V2 pipeline for the 9 better-performing disease classes
    - Improved Tomato Bacterial Spot detection
    - Improved Tomato Early Blight detection
    - Improved Tomato Leaf Mold detection
    - Adds local-contrast lesion detection
    - Adds adaptive local darkness/color deviation
    - Keeps connected-component filtering
    - Keeps strict leaf masking
    - Safe overlay generation

Dataset:
    12 disease classes
    5 images per class
    Total = 60 images
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"E:\smart_agriculture\smart_agriculture")

DATASET_DIR = (
    ROOT
    / "ai"
    / "severity"
    / "dataset"
    / "general_3000"
)

METADATA_PATH = DATASET_DIR / "metadata.csv"

OUTPUT_DIR = (
    ROOT
    / "ai"
    / "severity"
    / "results"
    / "general_severity_pilot_v3"
)

LEAF_DIR = OUTPUT_DIR / "leaf_masks"
DISEASE_DIR = OUTPUT_DIR / "disease_masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

for directory in [
    OUTPUT_DIR,
    LEAF_DIR,
    DISEASE_DIR,
    OVERLAY_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# CONFIGURATION
# ============================================================

IMAGES_PER_DISEASE = 5

MAX_PROCESSING_SIZE = 900

MIN_SEVERITY = 0.1
MAX_SEVERITY = 85.0


# ============================================================
# DISEASE PARAMETERS
# ============================================================

DISEASE_PARAMS = {

    # --------------------------------------------------------
    # APPLE
    # --------------------------------------------------------

    "Apple___Apple_scab": {
        "features": [
            "brown",
            "dark",
        ],
        "min_component": 25,
        "kernel": 3,
        "local_strength": 0.0,
    },

    "Apple___Black_rot": {
        "features": [
            "dark",
            "brown",
        ],
        "min_component": 30,
        "kernel": 3,
        "local_strength": 0.0,
    },

    "Apple___Cedar_apple_rust": {
        "features": [
            "orange",
            "brown",
            "yellow",
        ],
        "min_component": 20,
        "kernel": 3,
        "local_strength": 0.0,
    },

    # --------------------------------------------------------
    # CORN
    # --------------------------------------------------------

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "features": [
            "brown",
            "dark",
        ],
        "min_component": 25,
        "kernel": 5,
        "local_strength": 0.0,
    },

    "Corn_(maize)___Common_rust_": {
        "features": [
            "rust",
            "brown",
        ],
        "min_component": 15,
        "kernel": 3,
        "local_strength": 0.0,
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "features": [
            "brown",
            "yellow",
            "dark",
        ],
        "min_component": 40,
        "kernel": 5,
        "local_strength": 0.0,
    },

    # --------------------------------------------------------
    # POTATO
    # --------------------------------------------------------

    "Potato___Early_blight": {
        "features": [
            "brown",
            "dark",
        ],
        "min_component": 25,
        "kernel": 3,
        "local_strength": 0.0,
    },

    "Potato___Late_blight": {
        "features": [
            "dark",
            "brown",
        ],
        "min_component": 35,
        "kernel": 5,
        "local_strength": 0.0,
    },

    # --------------------------------------------------------
    # TOMATO
    # --------------------------------------------------------

    "Tomato___Bacterial_spot": {
        "features": [
            "dark",
            "brown",
            "spot",
        ],
        "min_component": 8,
        "kernel": 3,
        "local_strength": 1.0,
    },

    "Tomato___Early_blight": {
        "features": [
            "brown",
            "dark",
            "yellow",
        ],
        "min_component": 12,
        "kernel": 3,
        "local_strength": 0.85,
    },

    "Tomato___Late_blight": {
        "features": [
            "dark",
            "brown",
        ],
        "min_component": 30,
        "kernel": 5,
        "local_strength": 0.0,
    },

    "Tomato___Leaf_Mold": {
        "features": [
            "yellow",
            "brown",
            "dark",
        ],
        "min_component": 12,
        "kernel": 3,
        "local_strength": 0.85,
    },
}


# ============================================================
# RESIZE
# ============================================================

def resize_for_processing(
    image,
    max_size=MAX_PROCESSING_SIZE,
):
    """
    Resize while preserving aspect ratio.
    """

    height, width = image.shape[:2]

    largest = max(
        height,
        width,
    )

    if largest <= max_size:
        return image.copy()

    scale = (
        max_size
        / largest
    )

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# LEAF SEGMENTATION
# ============================================================

def create_leaf_mask(image):
    """
    Estimate visible leaf region using HSV + LAB
    followed by morphology and connected components.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    h, s, v = cv2.split(hsv)

    l, a, b = cv2.split(lab)

    # --------------------------------------------------------
    # Green vegetation
    # --------------------------------------------------------

    green = (
        (h >= 25)
        & (h <= 100)
        & (s >= 35)
        & (v >= 30)
    )

    # --------------------------------------------------------
    # LAB vegetation
    # --------------------------------------------------------

    lab_green = (
        (a < 145)
        & (b > 100)
        & (l > 30)
    )

    # --------------------------------------------------------
    # Saturated tissue
    # --------------------------------------------------------

    saturated = (
        (s > 45)
        & (v > 35)
    )

    mask = (
        green
        | (lab_green & saturated)
    ).astype(
        np.uint8
    ) * 255

    # --------------------------------------------------------
    # Morphology
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    if count > 1:

        areas = stats[
            1:,
            cv2.CC_STAT_AREA,
        ]

        largest_area = np.max(
            areas
        )

        cleaned = np.zeros_like(
            mask
        )

        for component_id in range(
            1,
            count,
        ):

            area = stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]

            if (
                area
                >= largest_area * 0.10
            ):

                cleaned[
                    labels == component_id
                ] = 255

        mask = cleaned

    # --------------------------------------------------------
    # GrabCut fallback
    # --------------------------------------------------------

    ratio = (
        np.count_nonzero(mask)
        / mask.size
    )

    if ratio < 0.20:

        try:

            gc_mask = np.full(
                image.shape[:2],
                cv2.GC_PR_BGD,
                dtype=np.uint8,
            )

            gc_mask[
                mask > 0
            ] = cv2.GC_PR_FGD

            erosion_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 9),
            )

            definite_fg = cv2.erode(
                mask,
                erosion_kernel,
                iterations=1,
            )

            gc_mask[
                definite_fg > 0
            ] = cv2.GC_FGD

            gc_mask[
                :5,
                :
            ] = cv2.GC_BGD

            gc_mask[
                -5:,
                :
            ] = cv2.GC_BGD

            gc_mask[
                :,
                :5
            ] = cv2.GC_BGD

            gc_mask[
                :,
                -5:
            ] = cv2.GC_BGD

            bg_model = np.zeros(
                (1, 65),
                np.float64,
            )

            fg_model = np.zeros(
                (1, 65),
                np.float64,
            )

            cv2.grabCut(
                image,
                gc_mask,
                None,
                bg_model,
                fg_model,
                3,
                cv2.GC_INIT_WITH_MASK,
            )

            mask = np.where(
                (
                    (gc_mask == cv2.GC_FGD)
                    |
                    (gc_mask == cv2.GC_PR_FGD)
                ),
                255,
                0,
            ).astype(
                np.uint8
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    final_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        final_kernel,
        iterations=1,
    )

    return mask


# ============================================================
# DISEASE COLOR FEATURES
# ============================================================

def get_disease_features(image):
    """
    Generate basic disease color masks.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    h, s, v = cv2.split(hsv)

    l, a, b = cv2.split(lab)

    blue = image[:, :, 0].astype(
        np.int16
    )

    green = image[:, :, 1].astype(
        np.int16
    )

    red = image[:, :, 2].astype(
        np.int16
    )

    # --------------------------------------------------------
    # Brown / necrotic
    # --------------------------------------------------------

    brown = (
        (h >= 5)
        & (h <= 30)
        & (s >= 30)
        & (v >= 35)
        & (red > green + 3)
    )

    # --------------------------------------------------------
    # Dark lesion
    # --------------------------------------------------------

    dark = (
        (v < 120)
        & (s > 20)
    )

    # --------------------------------------------------------
    # Yellow / chlorotic
    # --------------------------------------------------------

    yellow = (
        (h >= 20)
        & (h <= 45)
        & (s >= 25)
        & (v >= 75)
        & (red > green - 20)
    )

    # --------------------------------------------------------
    # Orange
    # --------------------------------------------------------

    orange = (
        (h >= 5)
        & (h <= 25)
        & (s >= 55)
        & (v >= 65)
        & (red > green + 12)
    )

    # --------------------------------------------------------
    # Rust
    # --------------------------------------------------------

    rust = (
        (h <= 20)
        & (s >= 45)
        & (v >= 40)
        & (red > green + 15)
    )

    # --------------------------------------------------------
    # Small spots
    # --------------------------------------------------------

    spot = (
        (v < 145)
        & (s > 20)
    )

    return {
        "brown": brown,
        "dark": dark,
        "yellow": yellow,
        "orange": orange,
        "rust": rust,
        "spot": spot,
    }


# ============================================================
# LOCAL CONTRAST FEATURES
# ============================================================

def get_local_contrast_masks(
    image,
    leaf_mask,
):
    """
    Detect pixels that are locally darker or chromatically
    different from their surrounding leaf tissue.

    This is particularly useful for:
        - Tomato Bacterial Spot
        - Tomato Early Blight
        - Tomato Leaf Mold
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    ).astype(
        np.float32
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    h, s, v = cv2.split(
        hsv
    )

    # --------------------------------------------------------
    # Local mean intensity
    # --------------------------------------------------------

    local_mean = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=9,
    )

    # Positive value means pixel is darker than surroundings
    darkness_difference = (
        local_mean
        - gray
    )

    locally_dark = (
        (darkness_difference > 12)
        & (v < 150)
    )

    # --------------------------------------------------------
    # Local saturation difference
    # --------------------------------------------------------

    sat = s.astype(
        np.float32
    )

    local_sat = cv2.GaussianBlur(
        sat,
        (0, 0),
        sigmaX=9,
    )

    saturation_difference = (
        np.abs(
            sat
            - local_sat
        )
    )

    local_color_change = (
        saturation_difference > 18
    )

    # --------------------------------------------------------
    # Local hue/color deviation
    # --------------------------------------------------------

    # Detect pixels that differ strongly from
    # surrounding saturation/color characteristics.
    local_color = (
        local_color_change
        & (s > 25)
        & (v > 35)
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    local_dark = locally_dark.astype(
        np.uint8
    )

    local_color = local_color.astype(
        np.uint8
    )

    # Restrict to leaf
    leaf_bool = (
        leaf_mask > 0
    )

    local_dark[
        ~leaf_bool
    ] = 0

    local_color[
        ~leaf_bool
    ] = 0

    return (
        local_dark > 0,
        local_color > 0,
    )


# ============================================================
# TOMATO-SPECIFIC DISEASE MASK
# ============================================================

def create_tomato_disease_mask(
    image,
    leaf_mask,
    disease_name,
):
    """
    Specialized segmentation for the three problematic
    Tomato disease classes.

    Uses:
        - color features
        - local darkness
        - local color deviation
        - adaptive thresholds
        - connected components
    """

    features = get_disease_features(
        image
    )

    local_dark, local_color = (
        get_local_contrast_masks(
            image,
            leaf_mask,
        )
    )

    candidate = np.zeros(
        leaf_mask.shape,
        dtype=np.uint8,
    )

    # ========================================================
    # TOMATO BACTERIAL SPOT
    # ========================================================

    if disease_name == "Tomato___Bacterial_spot":

        # Small dark/brown lesions
        basic = (
            (
                features["dark"]
                & (
                    features["spot"]
                    | features["brown"]
                )
            )
            |
            (
                features["brown"]
                & (features["spot"])
            )
        )

        # Local dark spots
        local_component = (
            local_dark
            & (features["spot"])
        )

        # Combine
        candidate[
            basic
            | local_component
        ] = 255

    # ========================================================
    # TOMATO EARLY BLIGHT
    # ========================================================

    elif disease_name == "Tomato___Early_blight":

        # Brown/dark lesions
        brown_dark = (
            features["brown"]
            | (
                features["dark"]
                & features["spot"]
            )
        )

        # Local lesion contrast
        local_component = (
            local_dark
            & (
                features["brown"]
                | features["spot"]
            )
        )

        # Chlorotic surroundings
        chlorotic = (
            features["yellow"]
            & local_color
        )

        candidate[
            brown_dark
            | local_component
            | chlorotic
        ] = 255

    # ========================================================
    # TOMATO LEAF MOLD
    # ========================================================

    elif disease_name == "Tomato___Leaf_Mold":

        # Yellow/brown/dark regions
        base = (
            features["yellow"]
            | features["brown"]
            | (
                features["dark"]
                & features["spot"]
            )
        )

        # Local color/darkness deviation
        local_component = (
            local_color
            | (
                local_dark
                & features["spot"]
            )
        )

        candidate[
            base
            & (
                local_component
                | features["brown"]
                | features["dark"]
            )
        ] = 255

    # --------------------------------------------------------
    # Restrict to leaf
    # --------------------------------------------------------

    candidate[
        leaf_mask == 0
    ] = 0

    return candidate


# ============================================================
# GENERAL DISEASE MASK
# ============================================================

def create_general_disease_mask(
    image,
    leaf_mask,
    disease_name,
):
    """
    V2-style segmentation for the nine disease classes
    that were already behaving reasonably.
    """

    parameters = DISEASE_PARAMS[
        disease_name
    ]

    features = get_disease_features(
        image
    )

    candidate = np.zeros(
        leaf_mask.shape,
        dtype=np.uint8,
    )

    for feature_name in parameters[
        "features"
    ]:

        candidate[
            features[feature_name]
        ] = 255

    candidate[
        leaf_mask == 0
    ] = 0

    # --------------------------------------------------------
    # Morphology
    # --------------------------------------------------------

    kernel_size = parameters[
        "kernel"
    ]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    return candidate


# ============================================================
# CONNECTED COMPONENT FILTER
# ============================================================

def clean_disease_components(
    disease_mask,
    leaf_mask,
    min_component,
):
    """
    Remove tiny components and pathological giant components.
    """

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            disease_mask,
            connectivity=8,
        )
    )

    cleaned = np.zeros_like(
        disease_mask
    )

    leaf_area = np.count_nonzero(
        leaf_mask
    )

    # Don't allow one component to consume
    # more than 35% of the leaf.
    max_component_area = (
        leaf_area * 0.35
    )

    for component_id in range(
        1,
        count,
    ):

        area = stats[
            component_id,
            cv2.CC_STAT_AREA,
        ]

        if area < min_component:
            continue

        if area > max_component_area:
            continue

        cleaned[
            labels == component_id
        ] = 255

    # --------------------------------------------------------
    # Small closing
    # --------------------------------------------------------

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
        iterations=1,
    )

    cleaned[
        leaf_mask == 0
    ] = 0

    return cleaned


# ============================================================
# DISEASE MASK
# ============================================================

def create_disease_mask(
    image,
    leaf_mask,
    disease_name,
):
    """
    Select specialized Tomato pipeline or
    standard V2 pipeline.
    """

    tomato_special_cases = {
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Leaf_Mold",
    }

    if disease_name in tomato_special_cases:

        candidate = create_tomato_disease_mask(
            image,
            leaf_mask,
            disease_name,
        )

    else:

        candidate = create_general_disease_mask(
            image,
            leaf_mask,
            disease_name,
        )

    minimum_component = DISEASE_PARAMS[
        disease_name
    ][
        "min_component"
    ]

    return clean_disease_components(
        candidate,
        leaf_mask,
        minimum_component,
    )


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    image,
    leaf_mask,
    disease_mask,
):
    """
    Disease regions = red
    Leaf boundary = cyan
    """

    overlay = image.copy()

    # --------------------------------------------------------
    # Leaf boundary
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        leaf_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (255, 255, 0),
        2,
    )

    # --------------------------------------------------------
    # Disease overlay
    # --------------------------------------------------------

    disease_pixels = (
        disease_mask > 0
    )

    if np.any(
        disease_pixels
    ):

        red_layer = np.zeros_like(
            image
        )

        red_layer[:, :, 2] = 255

        blended = cv2.addWeighted(
            image,
            0.35,
            red_layer,
            0.65,
            0,
        )

        overlay[
            disease_pixels
        ] = blended[
            disease_pixels
        ]

    return overlay


# ============================================================
# SEVERITY
# ============================================================

def calculate_severity(
    leaf_mask,
    disease_mask,
):
    """
    Severity =
        disease pixels / leaf pixels * 100
    """

    leaf_pixels = np.count_nonzero(
        leaf_mask
    )

    disease_pixels = np.count_nonzero(
        disease_mask
    )

    if leaf_pixels == 0:
        return (
            0.0,
            0,
            0,
        )

    severity = (
        disease_pixels
        / leaf_pixels
        * 100.0
    )

    severity = float(
        np.clip(
            severity,
            MIN_SEVERITY,
            MAX_SEVERITY,
        )
    )

    return (
        severity,
        leaf_pixels,
        disease_pixels,
    )


# ============================================================
# IMAGE PATH
# ============================================================

def resolve_image_path(row):
    """
    Resolve image path from metadata.
    """

    if "image_path" in row:

        path = Path(
            str(
                row["image_path"]
            )
        )

        if path.exists():
            return path

    if "relative_path" in row:

        path = (
            DATASET_DIR
            / str(
                row["relative_path"]
            )
        )

        if path.exists():
            return path

    if "image" in row:

        filename = str(
            row["image"]
        )

        matches = list(
            DATASET_DIR.rglob(
                filename
            )
        )

        if matches:
            return matches[0]

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GENERAL SEVERITY V3 — 60 IMAGE AUTOMATIC PILOT")
    print("=" * 70)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Metadata not found:\n"
            f"{METADATA_PATH}"
        )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    # --------------------------------------------------------
    # Select 5 images per disease
    # --------------------------------------------------------

    groups = []

    for disease, group in metadata.groupby(
        "disease"
    ):

        selected = group.sample(
            n=min(
                IMAGES_PER_DISEASE,
                len(group),
            ),
            random_state=42,
        )

        groups.append(
            selected
        )

    pilot = pd.concat(
        groups,
        ignore_index=True,
    )

    print(
        f"Total dataset records: "
        f"{len(metadata)}"
    )

    print(
        f"Pilot images: "
        f"{len(pilot)}"
    )

    print("\nImages per disease:")

    print(
        pilot[
            "disease"
        ].value_counts().sort_index()
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    successful = 0

    # ========================================================
    # PROCESS
    # ========================================================

    for index, row in pilot.iterrows():

        disease = str(
            row["disease"]
        )

        image_path = resolve_image_path(
            row
        )

        print(
            "\n"
            + "-"
            * 70
        )

        print(
            f"[{index + 1:02d}/{len(pilot)}] "
            f"{disease}"
        )

        if image_path is None:

            print(
                "ERROR: Image not found."
            )

            continue

        print(
            f"Image: {image_path.name}"
        )

        # ----------------------------------------------------
        # Read
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                "ERROR: Could not read image."
            )

            continue

        original_height, original_width = (
            image.shape[:2]
        )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        processed = resize_for_processing(
            image
        )

        # ----------------------------------------------------
        # Leaf
        # ----------------------------------------------------

        leaf_mask = create_leaf_mask(
            processed
        )

        # ----------------------------------------------------
        # Disease
        # ----------------------------------------------------

        disease_mask = create_disease_mask(
            processed,
            leaf_mask,
            disease,
        )

        # ----------------------------------------------------
        # Severity
        # ----------------------------------------------------

        (
            severity,
            leaf_pixels,
            disease_pixels,
        ) = calculate_severity(
            leaf_mask,
            disease_mask,
        )

        leaf_area_percent = (
            leaf_pixels
            / leaf_mask.size
            * 100.0
        )

        disease_area_percent = (
            disease_pixels
            / disease_mask.size
            * 100.0
        )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        overlay = create_overlay(
            processed,
            leaf_mask,
            disease_mask,
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        stem = image_path.stem

        cv2.imwrite(
            str(
                LEAF_DIR
                / f"{stem}_leaf.png"
            ),
            leaf_mask,
        )

        cv2.imwrite(
            str(
                DISEASE_DIR
                / f"{stem}_disease.png"
            ),
            disease_mask,
        )

        cv2.imwrite(
            str(
                OVERLAY_DIR
                / f"{stem}_overlay.jpg"
            ),
            overlay,
        )

        # ----------------------------------------------------
        # Console
        # ----------------------------------------------------

        print(
            f"Leaf area     : "
            f"{leaf_area_percent:.2f}%"
        )

        print(
            f"Disease area  : "
            f"{disease_area_percent:.2f}%"
        )

        print(
            f"Severity      : "
            f"{severity:.2f}%"
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        results.append(
            {
                "image": image_path.name,
                "disease": disease,
                "plant": row.get(
                    "plant",
                    "",
                ),
                "leaf_pixels": int(
                    leaf_pixels
                ),
                "disease_pixels": int(
                    disease_pixels
                ),
                "leaf_area_percent": float(
                    leaf_area_percent
                ),
                "disease_area_image_percent": float(
                    disease_area_percent
                ),
                "severity_percent": float(
                    severity
                ),
                "image_width": int(
                    original_width
                ),
                "image_height": int(
                    original_height
                ),
            }
        )

        successful += 1

    # ========================================================
    # SAVE CSV
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    csv_path = (
        OUTPUT_DIR
        / "pilot_severity_labels_v3.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")

    print("=" * 70)
    print("V3 PILOT COMPLETE")
    print("=" * 70)

    print(
        f"Successful: {successful}"
    )

    print(
        f"CSV: {csv_path}"
    )

    if results_df.empty:

        print(
            "No successful results."
        )

        return

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    print(
        "\nSeverity statistics:"
    )

    print(
        results_df[
            "severity_percent"
        ].describe()
    )

    # --------------------------------------------------------
    # Per disease
    # --------------------------------------------------------

    print(
        "\nSeverity by disease:"
    )

    summary = (
        results_df
        .groupby(
            "disease"
        )[
            "severity_percent"
        ]
        .agg(
            [
                "count",
                "min",
                "mean",
                "max",
            ]
        )
        .round(2)
    )

    print(
        summary
    )

    summary_path = (
        OUTPUT_DIR
        / "severity_summary_by_disease_v3.csv"
    )

    summary.to_csv(
        summary_path
    )

    # ========================================================
    # OUTPUTS
    # ========================================================

    print(
        "\nOutput folders:"
    )

    print(
        f"Leaf masks   : "
        f"{LEAF_DIR}"
    )

    print(
        f"Disease masks: "
        f"{DISEASE_DIR}"
    )

    print(
        f"Overlays     : "
        f"{OVERLAY_DIR}"
    )

    print(
        f"Labels CSV   : "
        f"{csv_path}"
    )

    print(
        f"Summary CSV  : "
        f"{summary_path}"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "NEXT STEP:"
    )

    print(
        "Inspect the V3 overlays before generating "
        "the full 3000-image dataset."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()