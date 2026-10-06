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
    / "severity_250_v1"
)

LEAF_MASK_DIR = OUTPUT_DIR / "leaf_masks"
DISEASE_MASK_OUTPUT_DIR = OUTPUT_DIR / "disease_masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LEAF_MASK_DIR.mkdir(parents=True, exist_ok=True)
DISEASE_MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_MASKS = 250


# ============================================================
# UUID EXTRACTION
# ============================================================

UUID_PATTERN = re.compile(
    r"([0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12})"
)


def extract_uuid(filename):
    """
    Extract PlantVillage UUID from a filename.
    """

    match = UUID_PATTERN.search(filename)

    if match:
        return match.group(1).lower()

    return None


# ============================================================
# INDEX SEGMENTED IMAGES
# ============================================================

print("=" * 75)
print("FINAL 250-IMAGE SEVERITY CALCULATION")
print("=" * 75)

print("\nIndexing PlantVillage segmented images...")

segmented_index = {}

segmented_files = list(
    SEGMENTED_DIR.glob("*")
)

for path in segmented_files:

    if not path.is_file():
        continue

    uid = extract_uuid(path.name)

    if uid is not None:
        segmented_index[uid] = path

print(
    f"Segmented images indexed: {len(segmented_index)}"
)


# ============================================================
# LOAD SAM2 DISEASE MASKS
# ============================================================

mask_files = sorted(
    DISEASE_MASK_DIR.glob("*_disease_mask.png")
)

print(
    f"SAM2 disease masks found: {len(mask_files)}"
)

if len(mask_files) != EXPECTED_MASKS:

    raise RuntimeError(
        f"\nExpected {EXPECTED_MASKS} disease masks "
        f"but found {len(mask_files)}.\n"
        "Do not continue until the mask count is correct."
    )


# ============================================================
# LEAF MASK FUNCTION
# ============================================================

def create_leaf_mask(segmented_image):

    """
    Create a binary leaf mask from the PlantVillage
    segmented image.

    The segmented image contains the leaf on a dark
    background.

    We estimate the foreground using grayscale intensity
    and morphology, then keep the largest connected
    component.
    """

    gray = cv2.cvtColor(
        segmented_image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Initial foreground threshold
    # --------------------------------------------------------

    # Pixels with meaningful intensity are considered
    # possible leaf pixels.
    mask = (gray > 5).astype(np.uint8) * 255

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels <= 1:
        return mask

    # Ignore background label 0
    areas = stats[1:, cv2.CC_STAT_AREA]

    largest_label = 1 + np.argmax(areas)

    leaf_mask = (
        labels == largest_label
    ).astype(np.uint8) * 255

    # --------------------------------------------------------
    # Final hole filling
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        leaf_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        filled = np.zeros_like(leaf_mask)

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        cv2.drawContours(
            filled,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

        leaf_mask = filled

    return leaf_mask


# ============================================================
# RESULT STORAGE
# ============================================================

results = []

successful = 0
failed = 0


# ============================================================
# PROCESS EACH MASK
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
        # FIND ORIGINAL IMAGE
        # ====================================================

        image_candidates = list(
            IMAGE_DIR.glob(
                image_stem + ".*"
            )
        )

        if not image_candidates:

            raise FileNotFoundError(
                f"Original image not found for {image_stem}"
            )

        image_path = image_candidates[0]

        original = cv2.imread(
            str(image_path)
        )

        if original is None:

            raise RuntimeError(
                "Could not read original image."
            )

        height, width = original.shape[:2]


        # ====================================================
        # FIND UUID
        # ====================================================

        uid = extract_uuid(
            image_path.name
        )

        if uid is None:

            raise RuntimeError(
                "Could not extract UUID from original image."
            )


        # ====================================================
        # FIND SEGMENTED IMAGE
        # ====================================================

        segmented_path = segmented_index.get(uid)

        if segmented_path is None:

            raise FileNotFoundError(
                f"No segmented image found for UUID {uid}"
            )

        segmented = cv2.imread(
            str(segmented_path)
        )

        if segmented is None:

            raise RuntimeError(
                "Could not read segmented image."
            )


        # ====================================================
        # CREATE LEAF MASK
        # ====================================================

        leaf_mask = create_leaf_mask(
            segmented
        )

        # Resize if necessary
        if (
            leaf_mask.shape[1] != width
            or leaf_mask.shape[0] != height
        ):

            leaf_mask = cv2.resize(
                leaf_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )


        # ====================================================
        # LOAD SAM2 DISEASE MASK
        # ====================================================

        disease_mask = cv2.imread(
            str(disease_mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if disease_mask is None:

            raise RuntimeError(
                "Could not read disease mask."
            )


        # Resize if necessary
        if (
            disease_mask.shape[1] != width
            or disease_mask.shape[0] != height
        ):

            disease_mask = cv2.resize(
                disease_mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )


        # Binary masks
        leaf_binary = (
            leaf_mask > 0
        )

        disease_binary = (
            disease_mask > 127
        )


        # ====================================================
        # CLIP DISEASE TO LEAF
        # ====================================================

        disease_inside_leaf = (
            disease_binary &
            leaf_binary
        )


        # ====================================================
        # PIXEL COUNTS
        # ====================================================

        leaf_pixels = int(
            np.sum(leaf_binary)
        )

        disease_pixels = int(
            np.sum(disease_inside_leaf)
        )


        if leaf_pixels == 0:

            raise RuntimeError(
                "Leaf mask contains zero pixels."
            )


        # ====================================================
        # SEVERITY
        # ====================================================

        severity = (
            disease_pixels /
            leaf_pixels
        ) * 100.0

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
        # CREATE OVERLAY
        # ====================================================

        overlay = original.copy()

        # Disease region shown in green
        disease_color = np.zeros_like(
            original
        )

        disease_color[:, :] = (
            0,
            255,
            0
        )

        alpha = 0.40

        overlay[disease_inside_leaf] = (
            (1 - alpha)
            * overlay[disease_inside_leaf]
            +
            alpha
            * disease_color[disease_inside_leaf]
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


        # Leaf boundary
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
            1
        )


        # ----------------------------------------------------
        # Add severity text
        # ----------------------------------------------------

        text = (
            f"Severity: {severity:.2f}%"
        )

        cv2.rectangle(
            overlay,
            (5, 5),
            (280, 45),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            overlay,
            text,
            (15, 33),
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
        # STORE RESULT
        # ====================================================

        results.append({
            "image": image_path.name,
            "image_stem": image_stem,
            "uuid": uid,
            "segmented_image": segmented_path.name,
            "leaf_pixels": leaf_pixels,
            "disease_pixels": disease_pixels,
            "severity_percent": severity
        })


        successful += 1

        print(
            f"Leaf pixels    : {leaf_pixels:,}"
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
            "severity_percent": np.nan
        })


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "severity_labels_250.csv"
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
print("FINAL 250-IMAGE SEVERITY SUMMARY")
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
        f"Valid labels     : "
        f"{len(valid)}"
    )


# ============================================================
# SEVERITY DISTRIBUTION
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


print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nCSV:")
print(csv_path)

print("\n" + "=" * 75)
print("CALCULATION COMPLETE")
print("=" * 75)