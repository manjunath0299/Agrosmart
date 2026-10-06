from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ORIGINAL_DIR = (
    PROJECT_ROOT
    / "PlantVillage-Dataset"
    / "raw"
    / "color"
    / "Tomato___Early_blight"
)

SAM2_MASK_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "sam2_annotations"
)

# NEW independent output
OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "severity_109_v3"
)

LEAF_MASK_DIR = OUTPUT_DIR / "leaf_masks"
DISEASE_MASK_DIR = OUTPUT_DIR / "disease_masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

CSV_PATH = OUTPUT_DIR / "severity_labels_109_v3.csv"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for directory in [
    OUTPUT_DIR,
    LEAF_MASK_DIR,
    DISEASE_MASK_DIR,
    OVERLAY_DIR,
]:
    directory.mkdir(
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

    match = UUID_PATTERN.search(filename)

    if match:
        return match.group(0).lower()

    return None


# ============================================================
# INDEX ORIGINAL IMAGES
# ============================================================

print("=" * 70)
print("SEVERITY CALCULATION V3 — 109 IMAGES")
print("=" * 70)

print()
print("Indexing original PlantVillage images...")

original_index = {}

for path in ORIGINAL_DIR.glob("*"):

    if not path.is_file():
        continue

    image_uuid = extract_uuid(path.name)

    if image_uuid:
        original_index[image_uuid] = path


print(
    f"Original images indexed: "
    f"{len(original_index)}"
)


# ============================================================
# FIND SAM2 MASKS
# ============================================================

sam2_masks = sorted(
    SAM2_MASK_DIR.glob("*_disease_mask.png"),
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
# FILL ENCLOSED HOLES
# ============================================================

def fill_holes(binary_mask):
    """
    Fill completely enclosed holes inside the leaf mask.
    """

    mask = binary_mask.copy()

    # Flood-fill the background from the image border.
    flood = mask.copy()

    h, w = flood.shape

    flood_mask = np.zeros(
        (h + 2, w + 2),
        np.uint8
    )

    cv2.floodFill(
        flood,
        flood_mask,
        (0, 0),
        255
    )

    # Pixels that were not reached by the border flood-fill
    # are either foreground or enclosed holes.
    flood_inv = cv2.bitwise_not(
        flood
    )

    filled = (
        mask
        | flood_inv
    )

    return filled


# ============================================================
# GRABCUT WHOLE-LEAF SEGMENTATION
# ============================================================

def create_leaf_mask(image):

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # Initial GrabCut mask
    # --------------------------------------------------------

    mask = np.zeros(
        (h, w),
        np.uint8
    )

    # Definite background
    mask[:] = cv2.GC_BGD

    # Conservative probable foreground
    margin_x = max(
        2,
        int(w * 0.08)
    )

    margin_y = max(
        2,
        int(h * 0.08)
    )

    rect = (
        margin_x,
        margin_y,
        w - 2 * margin_x,
        h - 2 * margin_y
    )

    mask[
        margin_y:h - margin_y,
        margin_x:w - margin_x
    ] = cv2.GC_PR_FGD

    # --------------------------------------------------------
    # GrabCut
    # --------------------------------------------------------

    bgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    fgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    cv2.grabCut(
        image,
        mask,
        rect,
        bgd_model,
        fgd_model,
        8,
        cv2.GC_INIT_WITH_RECT
    )

    leaf_mask = np.where(
        (
            (mask == cv2.GC_FGD)
            |
            (mask == cv2.GC_PR_FGD)
        ),
        255,
        0
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Keep largest connected component
    # --------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            leaf_mask,
            connectivity=8
        )
    )

    if num_labels > 1:

        largest_label = (
            1
            + np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA
                ]
            )
        )

        leaf_mask = np.where(
            labels == largest_label,
            255,
            0
        ).astype(np.uint8)

    # --------------------------------------------------------
    # Fill internal holes
    # --------------------------------------------------------

    leaf_mask = fill_holes(
        leaf_mask
    )

    # --------------------------------------------------------
    # Final small closing
    # --------------------------------------------------------

    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_CLOSE,
        small_kernel,
        iterations=1
    )

    return leaf_mask


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

    # --------------------------------------------------------
    # Disease = red
    # --------------------------------------------------------

    red_layer = np.zeros_like(
        image
    )

    red_layer[:] = (
        0,
        0,
        255
    )

    overlay[disease_bool] = cv2.addWeighted(
        image[disease_bool],
        0.45,
        red_layer[disease_bool],
        0.55,
        0
    )

    # --------------------------------------------------------
    # Leaf boundary = green
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    text = (
        f"Early Blight Severity: "
        f"{severity:.2f}%"
    )

    cv2.rectangle(
        overlay,
        (5, 5),
        (430, 45),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        overlay,
        text,
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(
        str(output_path),
        overlay,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )


# ============================================================
# PROCESS ALL 109
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
        # UUID
        # ----------------------------------------------------

        image_uuid = extract_uuid(
            mask_stem
        )

        if image_uuid is None:

            raise RuntimeError(
                "Could not extract UUID."
            )

        # ----------------------------------------------------
        # Original image
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
        # Load original
        # ----------------------------------------------------

        image = cv2.imread(
            str(original_path)
        )

        if image is None:

            raise RuntimeError(
                "Could not read original image."
            )

        # ----------------------------------------------------
        # Whole leaf
        # ----------------------------------------------------

        leaf_mask = create_leaf_mask(
            image
        )

        # ----------------------------------------------------
        # SAM2 disease mask
        # ----------------------------------------------------

        disease_mask = cv2.imread(
            str(disease_path),
            cv2.IMREAD_GRAYSCALE
        )

        if disease_mask is None:

            raise RuntimeError(
                "Could not read disease mask."
            )

        disease_mask = np.where(
            disease_mask > 0,
            255,
            0
        ).astype(np.uint8)

        # ----------------------------------------------------
        # Dimension check
        # ----------------------------------------------------

        if leaf_mask.shape != disease_mask.shape:

            raise RuntimeError(
                f"Shape mismatch: "
                f"leaf={leaf_mask.shape}, "
                f"disease={disease_mask.shape}"
            )

        # ----------------------------------------------------
        # Clip disease to leaf
        # ----------------------------------------------------

        disease_inside_leaf = (
            (disease_mask > 0)
            &
            (leaf_mask > 0)
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
        # Save leaf mask
        # ----------------------------------------------------

        leaf_output = (
            LEAF_MASK_DIR
            / f"{mask_stem}_leaf_mask.png"
        )

        cv2.imwrite(
            str(leaf_output),
            leaf_mask
        )

        # ----------------------------------------------------
        # Save clipped disease mask
        # ----------------------------------------------------

        disease_output = (
            DISEASE_MASK_DIR
            / f"{mask_stem}_disease_mask.png"
        )

        cv2.imwrite(
            str(disease_output),
            (
                disease_inside_leaf
                .astype(np.uint8)
                * 255
            )
        )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        overlay_output = (
            OVERLAY_DIR
            / f"{mask_stem}_severity.jpg"
        )

        create_overlay(
            image,
            leaf_mask,
            (
                disease_inside_leaf
                .astype(np.uint8)
                * 255
            ),
            severity,
            overlay_output
        )

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        records.append({
            "image": original_path.name,
            "uuid": image_uuid,
            "sam2_mask": disease_path.name,
            "leaf_mask": leaf_output.name,
            "disease_mask": disease_output.name,
            "leaf_pixels": leaf_pixels,
            "disease_pixels": disease_pixels,
            "leaf_area_percent_of_image": round(
                leaf_pixels
                / (
                    image.shape[0]
                    * image.shape[1]
                )
                * 100,
                4
            ),
            "severity_percent": round(
                severity,
                4
            ),
        })

        successful += 1

        print(
            f"[{counter:03d}/109] "
            f"{original_path.name} "
            f"-> severity "
            f"{severity:.2f}% "
            f"| leaf "
            f"{records[-1]['leaf_area_percent_of_image']:.2f}%"
        )

    except Exception as e:

        failed += 1

        print()
        print(
            f"[{counter:03d}/109] FAILED"
        )

        print(
            disease_path.name
        )

        print(
            f"Reason: {e}"
        )


# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(
    records
)

if len(df) > 0:

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
print("SEVERITY V3 COMPLETE")
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
    print(
        f"Minimum leaf area: "
        f"{df['leaf_area_percent_of_image'].min():.2f}%"
    )

    print(
        f"Maximum leaf area: "
        f"{df['leaf_area_percent_of_image'].max():.2f}%"
    )

    print(
        f"Mean leaf area: "
        f"{df['leaf_area_percent_of_image'].mean():.2f}%"
    )

    print()
    print("Samples >= 50% severity:")

    high = df[
        df["severity_percent"] >= 50
    ]

    if len(high) == 0:

        print("None")

    else:

        print(
            high[
                [
                    "image",
                    "severity_percent",
                    "leaf_area_percent_of_image",
                ]
            ].to_string(
                index=False
            )
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