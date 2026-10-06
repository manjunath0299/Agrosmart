from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Original PlantVillage images
ORIGINAL_DIR = (
    PROJECT_ROOT
    / "PlantVillage-Dataset"
    / "raw"
    / "color"
    / "Tomato___Early_blight"
)

# PlantVillage segmented leaf images
SEGMENTED_DIR = (
    PROJECT_ROOT
    / "PlantVillage-Dataset"
    / "raw"
    / "segmented"
    / "Tomato___Early_blight"
)

# All 109 SAM2 disease masks
SAM2_MASK_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "sam2_annotations"
)

# New independent output
OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "severity_109"
)

LEAF_MASK_DIR = OUTPUT_DIR / "leaf_masks"
DISEASE_MASK_DIR = OUTPUT_DIR / "disease_masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

CSV_PATH = OUTPUT_DIR / "severity_labels_109.csv"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LEAF_MASK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DISEASE_MASK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OVERLAY_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# UUID EXTRACTION
# ============================================================

UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}"
)


def extract_uuid(filename):
    match = UUID_PATTERN.search(
        filename
    )

    if match:
        return match.group(0).lower()

    return None


# ============================================================
# INDEX PLANTVILLAGE SEGMENTED IMAGES
# ============================================================

print("=" * 70)
print("SEVERITY CALCULATION — 109 IMAGES")
print("=" * 70)

print()
print("Indexing PlantVillage segmented images...")

segmented_index = {}

segmented_files = list(
    SEGMENTED_DIR.glob("*")
)

for path in segmented_files:

    if not path.is_file():
        continue

    image_uuid = extract_uuid(
        path.name
    )

    if image_uuid:
        segmented_index[
            image_uuid
        ] = path


print(
    f"Segmented images indexed: "
    f"{len(segmented_index)}"
)


# ============================================================
# INDEX ORIGINAL IMAGES
# ============================================================

print()
print("Indexing original images...")

original_index = {}

for path in ORIGINAL_DIR.glob("*"):

    if not path.is_file():
        continue

    image_uuid = extract_uuid(
        path.name
    )

    if image_uuid:
        original_index[
            image_uuid
        ] = path


print(
    f"Original images indexed: "
    f"{len(original_index)}"
)


# ============================================================
# FIND ALL 109 SAM2 MASKS
# ============================================================

sam2_masks = sorted(
    SAM2_MASK_DIR.glob(
        "*_disease_mask.png"
    ),
    key=lambda p: p.name.lower()
)

print()
print(
    f"SAM2 disease masks found: "
    f"{len(sam2_masks)}"
)

if len(sam2_masks) != 109:

    raise RuntimeError(
        f"Expected 109 SAM2 masks, "
        f"but found {len(sam2_masks)}."
    )


# ============================================================
# LEAF MASK CREATION
# ============================================================

def create_leaf_mask(
    segmented_path
):
    """
    Create a binary leaf mask from the PlantVillage
    segmented image.

    This segmented image is used only for
    leaf/background segmentation.

    It is NOT a disease mask.
    """

    image = cv2.imread(
        str(segmented_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read segmented image:\n"
            f"{segmented_path}"
        )

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Non-dark pixels approximately represent
    # the visible leaf.
    mask = gray > 20

    mask = (
        mask.astype(np.uint8)
        * 255
    )

    # Remove small noise
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Keep largest connected component
    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
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

        mask = np.where(
            labels == largest_label,
            255,
            0
        ).astype(np.uint8)

    return mask


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(
    image,
    leaf_mask,
    disease_mask,
    severity,
    output_path
):

    overlay = image.copy()

    leaf_bool = (
        leaf_mask > 0
    )

    disease_bool = (
        disease_mask > 0
    )

    # Disease pixels
    overlay[disease_bool] = (
        0.45 * overlay[disease_bool]
        + 0.55 * np.array(
            [0, 0, 255],
            dtype=np.float32
        )
    ).astype(np.uint8)

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
        f"Early Blight Severity: "
        f"{severity:.2f}%"
    )

    cv2.rectangle(
        overlay,
        (5, 5),
        (470, 42),
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

    cv2.imwrite(
        str(output_path),
        overlay,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )


# ============================================================
# PROCESS ALL MASKS
# ============================================================

records = []

successful = 0
failed = 0

print()
print("=" * 70)
print("PROCESSING")
print("=" * 70)


for counter, disease_path in enumerate(
    sam2_masks,
    start=1
):

    try:

        mask_stem = disease_path.stem

        # ----------------------------------------------------
        # Extract UUID from SAM2 filename
        # ----------------------------------------------------

        image_uuid = extract_uuid(
            mask_stem
        )

        if image_uuid is None:

            raise RuntimeError(
                "Could not extract UUID."
            )

        # ----------------------------------------------------
        # Find original image
        # ----------------------------------------------------

        original_path = (
            original_index.get(
                image_uuid
            )
        )

        if original_path is None:

            raise RuntimeError(
                f"Original image not found "
                f"for UUID {image_uuid}"
            )

        # ----------------------------------------------------
        # Find segmented image
        # ----------------------------------------------------

        segmented_path = (
            segmented_index.get(
                image_uuid
            )
        )

        if segmented_path is None:

            raise RuntimeError(
                f"Segmented image not found "
                f"for UUID {image_uuid}"
            )

        # ----------------------------------------------------
        # Load original image
        # ----------------------------------------------------

        original_image = cv2.imread(
            str(original_path)
        )

        if original_image is None:

            raise RuntimeError(
                "Could not read original image."
            )

        # ----------------------------------------------------
        # Create leaf mask
        # ----------------------------------------------------

        leaf_mask = create_leaf_mask(
            segmented_path
        )

        # ----------------------------------------------------
        # Load SAM2 disease mask
        # ----------------------------------------------------

        disease_mask = cv2.imread(
            str(disease_path),
            cv2.IMREAD_GRAYSCALE
        )

        if disease_mask is None:

            raise RuntimeError(
                "Could not read SAM2 disease mask."
            )

        disease_mask = (
            disease_mask > 0
        ).astype(np.uint8) * 255

        # ----------------------------------------------------
        # Verify dimensions
        # ----------------------------------------------------

        if (
            leaf_mask.shape
            != disease_mask.shape
        ):

            raise RuntimeError(
                f"Mask shape mismatch: "
                f"leaf={leaf_mask.shape}, "
                f"disease={disease_mask.shape}"
            )

        # ----------------------------------------------------
        # Clip disease to leaf
        # ----------------------------------------------------

        disease_inside_leaf = (
            (
                disease_mask > 0
            )
            &
            (
                leaf_mask > 0
            )
        )

        # ----------------------------------------------------
        # Pixel counts
        # ----------------------------------------------------

        leaf_pixels = int(
            np.sum(
                leaf_mask > 0
            )
        )

        disease_pixels = int(
            np.sum(
                disease_inside_leaf
            )
        )

        if leaf_pixels == 0:

            raise RuntimeError(
                "Leaf mask contains zero pixels."
            )

        # ----------------------------------------------------
        # Severity
        # ----------------------------------------------------

        severity = (
            disease_pixels
            / leaf_pixels
            * 100.0
        )

        # ----------------------------------------------------
        # Save masks
        # ----------------------------------------------------

        output_leaf = (
            LEAF_MASK_DIR
            / f"{mask_stem}_leaf_mask.png"
        )

        output_disease = (
            DISEASE_MASK_DIR
            / f"{mask_stem}_disease_mask.png"
        )

        cv2.imwrite(
            str(output_leaf),
            leaf_mask
        )

        cv2.imwrite(
            str(output_disease),
            (
                disease_inside_leaf
                .astype(np.uint8)
                * 255
            )
        )

        # ----------------------------------------------------
        # Save overlay
        # ----------------------------------------------------

        output_overlay = (
            OVERLAY_DIR
            / f"{mask_stem}_severity.jpg"
        )

        create_overlay(
            original_image,
            leaf_mask,
            disease_inside_leaf.astype(
                np.uint8
            ) * 255,
            severity,
            output_overlay
        )

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        records.append({
            "image": original_path.name,
            "uuid": image_uuid,
            "sam2_mask": disease_path.name,
            "leaf_mask": output_leaf.name,
            "disease_mask": output_disease.name,
            "leaf_pixels": leaf_pixels,
            "disease_pixels": disease_pixels,
            "severity_percent": round(
                severity,
                4
            ),
        })

        successful += 1

        print(
            f"[{counter:03d}/109] "
            f"{original_path.name} "
            f"-> {severity:.2f}%"
        )

    except Exception as e:

        failed += 1

        print(
            f"[{counter:03d}/109] FAILED: "
            f"{disease_path.name}"
        )

        print(
            f"    {e}"
        )


# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(
    records
)

df = df.sort_values(
    "severity_percent"
).reset_index(
    drop=True
)

df.to_csv(
    CSV_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SEVERITY CALCULATION COMPLETE")
print("=" * 70)

print(
    f"Expected masks:     109"
)

print(
    f"Successfully done:  {successful}"
)

print(
    f"Failed:             {failed}"
)

if len(df) > 0:

    print()
    print(
        f"Minimum severity: "
        f"{df['severity_percent'].min():.2f}%"
    )

    print(
        f"Maximum severity: "
        f"{df['severity_percent'].max():.2f}%"
    )

    print(
        f"Mean severity: "
        f"{df['severity_percent'].mean():.2f}%"
    )

    print(
        f"Median severity: "
        f"{df['severity_percent'].median():.2f}%"
    )

print()
print("CSV:")
print(CSV_PATH)

print()
print("Leaf masks:")
print(LEAF_MASK_DIR)

print()
print("Disease masks:")
print(DISEASE_MASK_DIR)

print()
print("Overlays:")
print(OVERLAY_DIR)

print()
print("=" * 70)