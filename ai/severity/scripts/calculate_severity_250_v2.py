from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
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

DISEASE_MASK_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "sam2_annotations"
)

SEGMENTED_DIR = (
    PROJECT_ROOT
    / "PlantVillage-Dataset"
    / "raw"
    / "segmented"
    / "Tomato___Early_blight"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "severity_250_v2"
)

LEAF_MASK_DIR = OUTPUT_DIR / "leaf_masks"
DISEASE_MASK_OUTPUT_DIR = OUTPUT_DIR / "disease_masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LEAF_MASK_DIR.mkdir(parents=True, exist_ok=True)
DISEASE_MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)


EXPECTED_MASKS = 250


# ============================================================
# UUID
# ============================================================

UUID_PATTERN = re.compile(
    r"([0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12})"
)


def extract_uuid(filename):

    match = UUID_PATTERN.search(filename)

    if match:
        return match.group(1).lower()

    return None


# ============================================================
# INDEX SEGMENTED IMAGES
# ============================================================

print("=" * 75)
print("FINAL 250-IMAGE SEVERITY CALCULATION V2")
print("=" * 75)

print("\nIndexing PlantVillage segmented images...")

segmented_index = {}

for path in SEGMENTED_DIR.glob("*"):

    if not path.is_file():
        continue

    uid = extract_uuid(path.name)

    if uid:
        segmented_index[uid] = path

print(
    f"Segmented images indexed: {len(segmented_index)}"
)


# ============================================================
# DISEASE MASKS
# ============================================================

mask_files = sorted(
    DISEASE_MASK_DIR.glob("*_disease_mask.png")
)

print(
    f"SAM2 disease masks found: {len(mask_files)}"
)

if len(mask_files) != EXPECTED_MASKS:

    raise RuntimeError(
        f"Expected {EXPECTED_MASKS} masks "
        f"but found {len(mask_files)}."
    )


# ============================================================
# ROBUST LEAF MASK
# ============================================================

def create_leaf_mask(original, segmented):
    """
    Estimate the complete visible leaf area.

    Uses:
      1. PlantVillage segmented image as a foreground hint.
      2. Original-image color information.
      3. Border pixels as background.
      4. GrabCut refinement.
      5. Largest connected component.

    This is designed to be more robust for dark leaves
    than simply thresholding grayscale intensity.
    """

    h, w = original.shape[:2]

    # --------------------------------------------------------
    # Convert to HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2HSV
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # --------------------------------------------------------
    # Initial mask from segmented image
    # --------------------------------------------------------

    seg_gray = cv2.cvtColor(
        segmented,
        cv2.COLOR_BGR2GRAY
    )

    # PlantVillage segmented foreground hint.
    seg_hint = (
        seg_gray > 5
    ).astype(np.uint8)

    # Clean the hint.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    seg_hint = cv2.morphologyEx(
        seg_hint,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # --------------------------------------------------------
    # Color-based foreground hints
    # --------------------------------------------------------

    # Green / colored leaf pixels.
    color_hint = (
        saturation > 25
    )

    # Very dark pixels can still be leaf tissue.
    dark_hint = (
        value < 100
    )

    # Combine.
    foreground_hint = (
        seg_hint.astype(bool)
        | color_hint
        | dark_hint
    )

    foreground_hint = (
        foreground_hint.astype(np.uint8)
    )

    # --------------------------------------------------------
    # Remove obvious border background
    # --------------------------------------------------------

    border = max(
        5,
        int(min(h, w) * 0.04)
    )

    foreground_hint[
        :border,
        :
    ] = 0

    foreground_hint[
        h-border:h,
        :
    ] = 0

    foreground_hint[
        :,
        :border
    ] = 0

    foreground_hint[
        :,
        w-border:w
    ] = 0

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    foreground_hint = cv2.morphologyEx(
        foreground_hint,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    foreground_hint = cv2.morphologyEx(
        foreground_hint,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=3
    )

    # --------------------------------------------------------
    # Keep useful connected components
    # --------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            foreground_hint,
            connectivity=8
        )
    )

    cleaned_hint = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    if num_labels > 1:

        areas = stats[
            1:,
            cv2.CC_STAT_AREA
        ]

        largest = (
            1 + np.argmax(areas)
        )

        # Keep largest component.
        cleaned_hint[
            labels == largest
        ] = 1

        # Also keep sufficiently large components
        # close to the main component.
        largest_area = areas.max()

        for label in range(
            1,
            num_labels
        ):

            area = stats[
                label,
                cv2.CC_STAT_AREA
            ]

            if area >= 0.08 * largest_area:

                cleaned_hint[
                    labels == label
                ] = 1

    else:

        cleaned_hint = foreground_hint

    # --------------------------------------------------------
    # GrabCut initialization
    # --------------------------------------------------------

    grab_mask = np.full(
        (h, w),
        cv2.GC_PR_BGD,
        dtype=np.uint8
    )

    # Definite background: image borders.
    grab_mask[
        :border,
        :
    ] = cv2.GC_BGD

    grab_mask[
        h-border:h,
        :
    ] = cv2.GC_BGD

    grab_mask[
        :,
        :border
    ] = cv2.GC_BGD

    grab_mask[
        :,
        w-border:w
    ] = cv2.GC_BGD

    # Foreground hints.
    grab_mask[
        cleaned_hint > 0
    ] = cv2.GC_PR_FGD

    # --------------------------------------------------------
    # Strong foreground seeds
    # --------------------------------------------------------

    # Saturated colored pixels inside the hint.
    strong_color = (
        (saturation > 40)
        & (cleaned_hint > 0)
    )

    grab_mask[
        strong_color
    ] = cv2.GC_FGD

    # Segmented-image pixels with clearly visible
    # foreground information.
    strong_segmented = (
        (seg_gray > 20)
        & (cleaned_hint > 0)
    )

    grab_mask[
        strong_segmented
    ] = cv2.GC_FGD

    # --------------------------------------------------------
    # Ensure enough foreground
    # --------------------------------------------------------

    fg_pixels = np.sum(
        (grab_mask == cv2.GC_FGD)
        | (grab_mask == cv2.GC_PR_FGD)
    )

    if fg_pixels < 0.01 * h * w:

        # Fallback: central rectangle as probable foreground.
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)

        grab_mask[
            margin_y:h-margin_y,
            margin_x:w-margin_x
        ] = cv2.GC_PR_FGD

    # --------------------------------------------------------
    # Run GrabCut
    # --------------------------------------------------------

    bgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    fgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    try:

        cv2.grabCut(
            original,
            grab_mask,
            None,
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_MASK
        )

    except cv2.error:

        # Fallback to cleaned hint.
        final_mask = (
            cleaned_hint * 255
        )

        return final_mask

    # --------------------------------------------------------
    # Extract foreground
    # --------------------------------------------------------

    final_mask = np.where(
        (
            (grab_mask == cv2.GC_FGD)
            |
            (grab_mask == cv2.GC_PR_FGD)
        ),
        255,
        0
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            final_mask,
            connectivity=8
        )
    )

    if num_labels <= 1:

        return final_mask

    areas = stats[
        1:,
        cv2.CC_STAT_AREA
    ]

    largest_label = (
        1 + np.argmax(areas)
    )

    largest_mask = (
        labels == largest_label
    ).astype(np.uint8) * 255

    # --------------------------------------------------------
    # Fill outer contour
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        largest_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        filled = np.zeros_like(
            largest_mask
        )

        cv2.drawContours(
            filled,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

        largest_mask = filled

    return largest_mask


# ============================================================
# RESULTS
# ============================================================

results = []

successful = 0
failed = 0


# ============================================================
# PROCESS
# ============================================================

for index, disease_mask_path in enumerate(
    mask_files,
    start=1
):

    mask_name = disease_mask_path.name

    image_stem = mask_name.replace(
        "_disease_mask.png",
        ""
    )

    print("\n" + "-" * 75)
    print(
        f"[{index:03d}/{len(mask_files):03d}] "
        f"{image_stem}"
    )

    try:

        # ====================================================
        # ORIGINAL IMAGE
        # ====================================================

        candidates = list(
            IMAGE_DIR.glob(
                image_stem + ".*"
            )
        )

        if not candidates:

            raise FileNotFoundError(
                "Original image not found."
            )

        image_path = candidates[0]

        original = cv2.imread(
            str(image_path)
        )

        if original is None:

            raise RuntimeError(
                "Could not read original image."
            )

        height, width = (
            original.shape[:2]
        )


        # ====================================================
        # UUID
        # ====================================================

        uid = extract_uuid(
            image_path.name
        )

        if uid is None:

            raise RuntimeError(
                "Could not extract UUID."
            )


        # ====================================================
        # SEGMENTED IMAGE
        # ====================================================

        segmented_path = (
            segmented_index.get(uid)
        )

        if segmented_path is None:

            raise FileNotFoundError(
                f"Segmented image missing for UUID {uid}"
            )

        segmented = cv2.imread(
            str(segmented_path)
        )

        if segmented is None:

            raise RuntimeError(
                "Could not read segmented image."
            )


        # ====================================================
        # LEAF MASK
        # ====================================================

        leaf_mask = create_leaf_mask(
            original,
            segmented
        )

        if (
            leaf_mask.shape[0] != height
            or
            leaf_mask.shape[1] != width
        ):

            leaf_mask = cv2.resize(
                leaf_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )


        # ====================================================
        # DISEASE MASK
        # ====================================================

        disease_mask = cv2.imread(
            str(disease_mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if disease_mask is None:

            raise RuntimeError(
                "Could not read disease mask."
            )

        if (
            disease_mask.shape[0] != height
            or
            disease_mask.shape[1] != width
        ):

            disease_mask = cv2.resize(
                disease_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )


        # ====================================================
        # BINARY
        # ====================================================

        leaf_binary = (
            leaf_mask > 127
        )

        disease_binary = (
            disease_mask > 127
        )


        # ====================================================
        # CLIP DISEASE TO LEAF
        # ====================================================

        disease_inside_leaf = (
            disease_binary
            &
            leaf_binary
        )


        # ====================================================
        # PIXEL COUNTS
        # ====================================================

        leaf_pixels = int(
            leaf_binary.sum()
        )

        disease_pixels = int(
            disease_inside_leaf.sum()
        )

        if leaf_pixels == 0:

            raise RuntimeError(
                "Leaf mask has zero pixels."
            )


        # ====================================================
        # SEVERITY
        # ====================================================

        severity = (
            disease_pixels
            /
            leaf_pixels
            *
            100.0
        )

        severity = float(
            np.clip(
                severity,
                0.0,
                100.0
            )
        )


        # ====================================================
        # SAVE LEAF MASK
        # ====================================================

        leaf_output = (
            LEAF_MASK_DIR
            / f"{image_stem}_leaf_mask.png"
        )

        cv2.imwrite(
            str(leaf_output),
            leaf_mask
        )


        # ====================================================
        # SAVE CLIPPED DISEASE MASK
        # ====================================================

        disease_output = (
            DISEASE_MASK_OUTPUT_DIR
            / f"{image_stem}_disease_mask.png"
        )

        clipped_disease = (
            disease_inside_leaf.astype(
                np.uint8
            ) * 255
        )

        cv2.imwrite(
            str(disease_output),
            clipped_disease
        )


        # ====================================================
        # OVERLAY
        # ====================================================

        overlay = original.copy()

        # Disease = green
        green = np.zeros_like(
            original
        )

        green[:, :] = (
            0,
            255,
            0
        )

        disease_indices = (
            disease_inside_leaf
        )

        alpha = 0.40

        overlay[disease_indices] = (
            (1 - alpha)
            * overlay[disease_indices]
            +
            alpha
            * green[disease_indices]
        ).astype(np.uint8)


        # Disease boundary
        contours, _ = cv2.findContours(
            clipped_disease,
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


        # Leaf boundary = blue
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


        # ----------------------------------------------------
        # Severity text
        # ----------------------------------------------------

        cv2.rectangle(
            overlay,
            (5, 5),
            (300, 48),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            overlay,
            f"Severity: {severity:.2f}%",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )


        overlay_path = (
            OVERLAY_DIR
            / f"{image_stem}_overlay.png"
        )

        cv2.imwrite(
            str(overlay_path),
            overlay
        )


        # ====================================================
        # STORE
        # ====================================================

        results.append({
            "image": image_path.name,
            "image_stem": image_stem,
            "uuid": uid,
            "segmented_image": segmented_path.name,
            "leaf_pixels": leaf_pixels,
            "disease_pixels": disease_pixels,
            "leaf_area_percent": (
                leaf_pixels
                /
                (height * width)
                *
                100.0
            ),
            "severity_percent": severity
        })

        successful += 1

        print(
            f"Leaf pixels    : {leaf_pixels:,}"
        )

        print(
            f"Leaf area      : "
            f"{leaf_pixels/(height*width)*100:.2f}%"
        )

        print(
            f"Disease pixels : {disease_pixels:,}"
        )

        print(
            f"Severity       : {severity:.2f}%"
        )


    except Exception as e:

        failed += 1

        print(
            f"ERROR: {e}"
        )

        results.append({
            "image": image_stem,
            "image_stem": image_stem,
            "uuid": "",
            "segmented_image": "",
            "leaf_pixels": 0,
            "disease_pixels": 0,
            "leaf_area_percent": np.nan,
            "severity_percent": np.nan
        })


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "severity_labels_250_v2.csv"
)

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

valid = results_df[
    results_df["severity_percent"].notna()
].copy()

print("\n")
print("=" * 75)
print("FINAL 250-IMAGE SEVERITY V2 SUMMARY")
print("=" * 75)

print(
    f"Expected masks : {EXPECTED_MASKS}"
)

print(
    f"Found masks    : {len(mask_files)}"
)

print(
    f"Successful     : {successful}"
)

print(
    f"Failed         : {failed}"
)

if len(valid) > 0:

    severity_values = (
        valid["severity_percent"]
        .astype(float)
    )

    leaf_area_values = (
        valid["leaf_area_percent"]
        .astype(float)
    )

    print(
        f"Minimum severity : "
        f"{severity_values.min():.2f}%"
    )

    print(
        f"Maximum severity : "
        f"{severity_values.max():.2f}%"
    )

    print(
        f"Mean severity    : "
        f"{severity_values.mean():.2f}%"
    )

    print(
        f"Median severity  : "
        f"{severity_values.median():.2f}%"
    )

    print(
        f"Std deviation    : "
        f"{severity_values.std():.2f}%"
    )

    print(
        f"Minimum leaf area: "
        f"{leaf_area_values.min():.2f}%"
    )

    print(
        f"Maximum leaf area: "
        f"{leaf_area_values.max():.2f}%"
    )

    print(
        f"Mean leaf area   : "
        f"{leaf_area_values.mean():.2f}%"
    )

    print(
        f"Valid labels     : "
        f"{len(valid)}"
    )


# ============================================================
# DISTRIBUTION
# ============================================================

if len(valid) > 0:

    print("\nSeverity distribution:")

    bins = [
        (0, 5, "0-5%"),
        (5, 10, "5-10%"),
        (10, 20, "10-20%"),
        (20, 30, "20-30%"),
        (30, 40, "30-40%"),
        (40, 50, "40-50%"),
        (50, 60, "50-60%"),
        (60, 70, "60-70%"),
        (70, 80, "70-80%"),
        (80, 90, "80-90%"),
        (90, 100.1, "90-100%")
    ]

    for low, high, label in bins:

        count = int(
            (
                (severity_values >= low)
                &
                (severity_values < high)
            ).sum()
        )

        print(
            f"  {label:10s}: {count}"
        )


# ============================================================
# EXTREME VALUES
# ============================================================

if len(valid) > 0:

    print("\nTop 10 highest severity:")

    print(
        valid.nlargest(
            10,
            "severity_percent"
        )[
            [
                "image",
                "leaf_pixels",
                "disease_pixels",
                "leaf_area_percent",
                "severity_percent"
            ]
        ].to_string(index=False)
    )

    print("\nTop 10 lowest severity:")

    print(
        valid.nsmallest(
            10,
            "severity_percent"
        )[
            [
                "image",
                "leaf_pixels",
                "disease_pixels",
                "leaf_area_percent",
                "severity_percent"
            ]
        ].to_string(index=False)
    )


# ============================================================
# OUTPUT
# ============================================================

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nCSV:")
print(csv_path)

print("\n" + "=" * 75)
print("SEVERITY V2 CALCULATION COMPLETE")
print("=" * 75)