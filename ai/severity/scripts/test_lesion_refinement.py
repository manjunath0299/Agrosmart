from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import re


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"E:\smart_agriculture\smart_agriculture")

IMAGE_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "dataset"
    / "images"
    / "annotation_pool"
)

SAM2_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "sam2_annotations"
)

V2_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "severity_250_v2"
)

V2_LEAF_DIR = V2_DIR / "leaf_masks"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "lesion_refinement_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEST IMAGES
# ============================================================

TARGETS = [
    "*9436*",
    "*8220*",
    "*7597*",
    "*8328*",
    "*6401*",
    "*9411*",
]


# ============================================================
# FIND IMAGES
# ============================================================

image_paths = []

for pattern in TARGETS:

    matches = list(
        IMAGE_DIR.glob(pattern + ".JPG")
    )

    if not matches:
        matches = list(
            IMAGE_DIR.glob(pattern + ".jpg")
        )

    if not matches:
        raise FileNotFoundError(
            f"Could not find image: {pattern}"
        )

    image_paths.append(matches[0])


# ============================================================
# REFINEMENT FUNCTION
# ============================================================

def refine_disease_mask(
    image,
    leaf_mask,
    sam2_mask
):
    """
    Refine the SAM2 disease proposal.

    Goal:
        Remove large healthy-green areas from the SAM2 mask.

    We use:
        - HSV
        - LAB
        - relative RGB relationships
        - darkness
        - morphology
        - connected components

    This is an experimental refinement and must be
    visually validated before being used for all 250 images.
    """

    # --------------------------------------------------------
    # Binary masks
    # --------------------------------------------------------

    leaf = leaf_mask > 127
    sam2 = sam2_mask > 127

    # Work only where SAM2 selected disease
    roi = leaf & sam2

    if roi.sum() == 0:
        return np.zeros_like(sam2_mask)

    # --------------------------------------------------------
    # Color spaces
    # --------------------------------------------------------

    b, g, r = cv2.split(image)

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(hsv)

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    L, A, B = cv2.split(lab)

    # --------------------------------------------------------
    # Relative color
    # --------------------------------------------------------

    # Red-vs-green:
    # Brown/red disease tissue generally has
    # more red relative to green than healthy leaf tissue.

    red_green = (
        r.astype(np.int16)
        -
        g.astype(np.int16)
    )

    # Blue-vs-green
    blue_green = (
        b.astype(np.int16)
        -
        g.astype(np.int16)
    )

    # --------------------------------------------------------
    # Candidate disease conditions
    # --------------------------------------------------------

    # Dark tissue
    dark = (
        v < 105
    )

    # Brown / reddish tissue
    brown = (
        (red_green > 8)
        &
        (r > b + 5)
        &
        (s > 25)
    )

    # Strongly non-green tissue
    non_green = (
        red_green > 15
    )

    # Yellow / chlorotic tissue.
    #
    # This is deliberately conservative because yellow
    # healthy tissue must not automatically become disease.
    yellow = (
        (r > 100)
        &
        (g > 90)
        &
        (b < 100)
        &
        (s > 35)
        &
        (red_green > -20)
    )

    # Dark + colored tissue
    dark_colored = (
        dark
        &
        (s > 20)
    )

    # --------------------------------------------------------
    # Combine disease candidates
    # --------------------------------------------------------

    candidate = (
        brown
        |
        non_green
        |
        dark_colored
        |
        yellow
    )

    candidate = (
        candidate
        &
        roi
    )

    # --------------------------------------------------------
    # Remove very weak isolated pixels
    # --------------------------------------------------------

    candidate_uint8 = (
        candidate.astype(np.uint8)
        * 255
    )

    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    candidate_uint8 = cv2.morphologyEx(
        candidate_uint8,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    # Connect nearby lesion pixels
    kernel_medium = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    candidate_uint8 = cv2.morphologyEx(
        candidate_uint8,
        cv2.MORPH_CLOSE,
        kernel_medium,
        iterations=1
    )

    # --------------------------------------------------------
    # Remove tiny components
    # --------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            candidate_uint8,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(
        candidate_uint8
    )

    min_component_area = max(
        10,
        int(image.shape[0] * image.shape[1] * 0.00015)
    )

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= min_component_area:

            cleaned[
                labels == label
            ] = 255

    # --------------------------------------------------------
    # Final clip to leaf + SAM2
    # --------------------------------------------------------

    cleaned = (
        (
            cleaned > 127
        )
        &
        leaf
        &
        sam2
    ).astype(np.uint8) * 255

    return cleaned


# ============================================================
# PROCESS
# ============================================================

summary = []

print("=" * 75)
print("LESION REFINEMENT TEST")
print("=" * 75)

for index, image_path in enumerate(
    image_paths,
    start=1
):

    print(
        f"\n[{index}/6] {image_path.name}"
    )

    stem = image_path.stem

    # --------------------------------------------------------
    # Load original
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print("ERROR: image")
        continue

    # --------------------------------------------------------
    # SAM2 mask
    # --------------------------------------------------------

    sam2_path = (
        SAM2_DIR
        / f"{stem}_disease_mask.png"
    )

    sam2_mask = cv2.imread(
        str(sam2_path),
        cv2.IMREAD_GRAYSCALE
    )

    if sam2_mask is None:
        print("ERROR: SAM2 mask")
        continue

    # --------------------------------------------------------
    # Leaf mask
    # --------------------------------------------------------

    leaf_path = (
        V2_LEAF_DIR
        / f"{stem}_leaf_mask.png"
    )

    leaf_mask = cv2.imread(
        str(leaf_path),
        cv2.IMREAD_GRAYSCALE
    )

    if leaf_mask is None:
        print("ERROR: leaf mask")
        continue

    # --------------------------------------------------------
    # Resize if necessary
    # --------------------------------------------------------

    h, w = image.shape[:2]

    if sam2_mask.shape != (h, w):

        sam2_mask = cv2.resize(
            sam2_mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

    if leaf_mask.shape != (h, w):

        leaf_mask = cv2.resize(
            leaf_mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

    # --------------------------------------------------------
    # Refine
    # --------------------------------------------------------

    refined = refine_disease_mask(
        image,
        leaf_mask,
        sam2_mask
    )

    # --------------------------------------------------------
    # Calculate values
    # --------------------------------------------------------

    leaf_binary = (
        leaf_mask > 127
    )

    sam2_binary = (
        sam2_mask > 127
    )

    refined_binary = (
        refined > 127
    )

    leaf_pixels = int(
        leaf_binary.sum()
    )

    sam2_pixels = int(
        (
            sam2_binary
            &
            leaf_binary
        ).sum()
    )

    refined_pixels = int(
        (
            refined_binary
            &
            leaf_binary
        ).sum()
    )

    sam2_severity = (
        sam2_pixels
        /
        leaf_pixels
        *
        100
    )

    refined_severity = (
        refined_pixels
        /
        leaf_pixels
        *
        100
    )

    # --------------------------------------------------------
    # Create visualization
    # --------------------------------------------------------

    overlay = image.copy()

    # --------------------------------------------------------
    # SAM2 area = green
    # --------------------------------------------------------

    sam2_only = (
        sam2_binary
        &
        leaf_binary
    )

    green = np.zeros_like(
        image
    )

    green[:, :] = (
        0,
        255,
        0
    )

    overlay[sam2_only] = (
        0.20 * overlay[sam2_only]
        +
        0.20 * green[sam2_only]
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Refined disease = red
    # --------------------------------------------------------

    red = np.zeros_like(
        image
    )

    red[:, :] = (
        0,
        0,
        255
    )

    overlay[refined_binary] = (
        0.55 * overlay[refined_binary]
        +
        0.45 * red[refined_binary]
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Refined boundary
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        refined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 0, 255),
        2
    )

    # --------------------------------------------------------
    # Leaf boundary = blue
    # --------------------------------------------------------

    leaf_contours, _ = cv2.findContours(
        leaf_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        leaf_contours,
        -1,
        (255, 0, 0),
        2
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    cv2.rectangle(
        overlay,
        (5, 5),
        (450, 75),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        overlay,
        f"SAM2: {sam2_severity:.2f}%",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        overlay,
        f"Refined: {refined_severity:.2f}%",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    mask_output = (
        OUTPUT_DIR
        / f"{stem}_refined_mask.png"
    )

    overlay_output = (
        OUTPUT_DIR
        / f"{stem}_refined_overlay.png"
    )

    cv2.imwrite(
        str(mask_output),
        refined
    )

    cv2.imwrite(
        str(overlay_output),
        overlay
    )

    summary.append({
        "image": image_path.name,
        "leaf_pixels": leaf_pixels,
        "sam2_pixels": sam2_pixels,
        "refined_pixels": refined_pixels,
        "sam2_severity": sam2_severity,
        "refined_severity": refined_severity
    })

    print(
        f"SAM2 severity     : {sam2_severity:.2f}%"
    )

    print(
        f"Refined severity  : {refined_severity:.2f}%"
    )


# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(
    summary
)

csv_path = (
    OUTPUT_DIR
    / "refinement_comparison.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("REFINEMENT TEST COMPLETE")
print("=" * 75)

print(
    df[
        [
            "image",
            "sam2_severity",
            "refined_severity"
        ]
    ].to_string(index=False)
)

print("\nOutput:")
print(OUTPUT_DIR)

print("\nCSV:")
print(csv_path)