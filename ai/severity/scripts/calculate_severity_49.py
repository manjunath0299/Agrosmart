from pathlib import Path
import re
import csv

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\smart_agriculture")

IMAGE_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "dataset"
    / "images"
    / "annotation_pool"
)

DISEASE_DIR = (
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
    / "severity_49"
)

LEAF_MASK_DIR = OUTPUT_DIR / "leaf_masks"
FINAL_DISEASE_DIR = OUTPUT_DIR / "disease_masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

CSV_PATH = OUTPUT_DIR / "severity_labels_49.csv"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LEAF_MASK_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DISEASE_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# UUID
# ============================================================

UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}"
)


def extract_uuid(filename):

    prefix = filename.split("___", 1)[0]

    # UUID directly at beginning
    match = UUID_PATTERN.match(prefix)

    if match:
        return match.group(0).lower()

    # leaf_2_0_UUID
    parts = prefix.split("_")

    if len(parts) >= 4:

        possible_uuid = parts[-1]

        if UUID_PATTERN.match(possible_uuid):
            return possible_uuid.lower()

    return None


# ============================================================
# BUILD SEGMENTED LOOKUP
# ============================================================

def build_segmented_lookup():

    lookup = {}

    files = SEGMENTED_DIR.glob(
        "*_final_masked.jpg"
    )

    for path in files:

        uuid = extract_uuid(path.name)

        if uuid is not None:
            lookup[uuid] = path

    return lookup


# ============================================================
# LEAF MASK FROM SEGMENTED IMAGE
# ============================================================

def create_leaf_mask(
    original_image,
    segmented_image,
):
    """
    Create a binary leaf mask from the PlantVillage
    segmented/masked image.

    The segmented image has the leaf retained while
    background is masked.

    We use the non-background pixels to construct
    the leaf mask.

    Output:
        uint8 mask
        0 = background
        255 = leaf
    """

    # Resize segmented image if necessary
    if (
        segmented_image.shape[:2]
        != original_image.shape[:2]
    ):

        segmented_image = cv2.resize(
            segmented_image,
            (
                original_image.shape[1],
                original_image.shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    # Convert to grayscale
    gray = cv2.cvtColor(
        segmented_image,
        cv2.COLOR_BGR2GRAY,
    )

    # --------------------------------------------------------
    # Background estimation
    # --------------------------------------------------------
    #
    # PlantVillage masked images normally contain a dark /
    # black background around the retained leaf.
    #
    # Use a low intensity threshold.
    # --------------------------------------------------------

    mask = np.where(
        gray > 10,
        255,
        0,
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    kernel = np.ones(
        (5, 5),
        np.uint8,
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
    # Keep large connected component(s)
    # --------------------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if num_labels > 1:

        areas = stats[1:, cv2.CC_STAT_AREA]

        largest_label = (
            1 + np.argmax(areas)
        )

        mask = np.where(
            labels == largest_label,
            255,
            0,
        ).astype(np.uint8)

    # --------------------------------------------------------
    # Fill small holes
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if contours:

        largest_contour = max(
            contours,
            key=cv2.contourArea,
        )

        clean_mask = np.zeros_like(mask)

        cv2.drawContours(
            clean_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED,
        )

        mask = clean_mask

    return mask


# ============================================================
# DISEASE MASK
# ============================================================

def load_disease_mask(
    disease_mask_path,
    target_shape,
):

    mask = cv2.imread(
        str(disease_mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:

        raise RuntimeError(
            f"Could not read disease mask:\n"
            f"{disease_mask_path}"
        )

    target_h, target_w = target_shape

    if mask.shape != (
        target_h,
        target_w,
    ):

        mask = cv2.resize(
            mask,
            (
                target_w,
                target_h,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    return np.where(
        mask > 127,
        255,
        0,
    ).astype(np.uint8)


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    image,
    leaf_mask,
    disease_mask,
    severity,
):

    overlay = image.copy()

    # --------------------------------------------------------
    # Disease = red
    # --------------------------------------------------------

    disease_bool = (
        disease_mask > 0
    )

    red_layer = np.zeros_like(
        overlay
    )

    red_layer[:, :] = (
        0,
        0,
        255,
    )

    overlay[disease_bool] = cv2.addWeighted(
        overlay[disease_bool],
        0.50,
        red_layer[disease_bool],
        0.50,
        0,
    )

    # --------------------------------------------------------
    # Leaf contour
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
        (0, 255, 0),
        2,
    )

    # --------------------------------------------------------
    # Severity text
    # --------------------------------------------------------

    text = (
        f"Early Blight Severity: "
        f"{severity:.2f}%"
    )

    cv2.rectangle(
        overlay,
        (10, 10),
        (470, 55),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        overlay,
        text,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return overlay


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("CALCULATING SEVERITY FOR 49 SAM2-ANNOTATED IMAGES")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    for directory in [
        IMAGE_DIR,
        DISEASE_DIR,
        SEGMENTED_DIR,
    ]:

        if not directory.exists():

            raise FileNotFoundError(
                f"Directory not found:\n{directory}"
            )

    # --------------------------------------------------------
    # Build original lookup
    # --------------------------------------------------------

    original_lookup = {
        p.stem: p
        for p in IMAGE_DIR.iterdir()
        if p.is_file()
    }

    # --------------------------------------------------------
    # Build segmented lookup
    # --------------------------------------------------------

    segmented_lookup = (
        build_segmented_lookup()
    )

    print(
        f"Segmented images indexed: "
        f"{len(segmented_lookup)}"
    )

    # --------------------------------------------------------
    # Disease masks
    # --------------------------------------------------------

    disease_masks = sorted(
        DISEASE_DIR.glob(
            "*_disease_mask.png"
        )
    )

    print(
        f"Disease masks found: "
        f"{len(disease_masks)}"
    )

    if len(disease_masks) == 0:

        raise RuntimeError(
            "No disease masks found."
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    rows = []

    # --------------------------------------------------------
    # Process each image
    # --------------------------------------------------------

    for index, disease_path in enumerate(
        disease_masks,
        start=1,
    ):

        base_name = disease_path.name.replace(
            "_disease_mask.png",
            "",
        )

        print()
        print(
            f"[{index}/{len(disease_masks)}]"
        )

        print(
            base_name
        )

        # ----------------------------------------------------
        # Original
        # ----------------------------------------------------

        original_path = (
            original_lookup.get(
                base_name
            )
        )

        if original_path is None:

            print(
                "ERROR: Original image missing."
            )

            continue

        image = cv2.imread(
            str(original_path)
        )

        if image is None:

            print(
                "ERROR: Could not read original."
            )

            continue

        # ----------------------------------------------------
        # UUID
        # ----------------------------------------------------

        uuid = extract_uuid(
            base_name
        )

        if uuid is None:

            print(
                "ERROR: UUID not found."
            )

            continue

        # ----------------------------------------------------
        # Segmented image
        # ----------------------------------------------------

        segmented_path = (
            segmented_lookup.get(
                uuid
            )
        )

        if segmented_path is None:

            print(
                "ERROR: Segmented image missing."
            )

            continue

        segmented = cv2.imread(
            str(segmented_path)
        )

        if segmented is None:

            print(
                "ERROR: Could not read segmented image."
            )

            continue

        # ----------------------------------------------------
        # Leaf mask
        # ----------------------------------------------------

        leaf_mask = create_leaf_mask(
            image,
            segmented,
        )

        # ----------------------------------------------------
        # Disease mask
        # ----------------------------------------------------

        disease_mask = load_disease_mask(
            disease_path,
            image.shape[:2],
        )

        # ----------------------------------------------------
        # Clip disease to leaf
        # ----------------------------------------------------

        final_disease_mask = np.where(
            (
                (disease_mask > 0)
                &
                (leaf_mask > 0)
            ),
            255,
            0,
        ).astype(np.uint8)

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
                final_disease_mask > 0
            )
        )

        # ----------------------------------------------------
        # Severity
        # ----------------------------------------------------

        if leaf_pixels == 0:

            print(
                "ERROR: Leaf mask is empty."
            )

            continue

        severity = (
            disease_pixels
            / leaf_pixels
            * 100.0
        )

        # Safety clamp
        severity = max(
            0.0,
            min(
                100.0,
                severity,
            ),
        )

        print(
            f"Leaf pixels    : {leaf_pixels:,}"
        )

        print(
            f"Disease pixels : {disease_pixels:,}"
        )

        print(
            f"Severity       : {severity:.2f}%"
        )

        # ----------------------------------------------------
        # Save leaf mask
        # ----------------------------------------------------

        leaf_path = (
            LEAF_MASK_DIR
            / f"{base_name}_leaf_mask.png"
        )

        cv2.imwrite(
            str(leaf_path),
            leaf_mask,
        )

        # ----------------------------------------------------
        # Save final disease mask
        # ----------------------------------------------------

        final_disease_path = (
            FINAL_DISEASE_DIR
            / f"{base_name}_disease_mask.png"
        )

        cv2.imwrite(
            str(final_disease_path),
            final_disease_mask,
        )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        overlay = create_overlay(
            image,
            leaf_mask,
            final_disease_mask,
            severity,
        )

        overlay_path = (
            OVERLAY_DIR
            / f"{base_name}_severity.jpg"
        )

        cv2.imwrite(
            str(overlay_path),
            overlay,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

        # ----------------------------------------------------
        # CSV row
        # ----------------------------------------------------

        rows.append(
            {
                "image": base_name,
                "uuid": uuid,
                "original_image": str(
                    original_path
                ),
                "segmented_image": str(
                    segmented_path
                ),
                "disease_mask": str(
                    disease_path
                ),
                "leaf_mask": str(
                    leaf_path
                ),
                "final_disease_mask": str(
                    final_disease_path
                ),
                "leaf_pixels": leaf_pixels,
                "disease_pixels": disease_pixels,
                "severity_percent": round(
                    severity,
                    4,
                ),
            }
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    fieldnames = [
        "image",
        "uuid",
        "original_image",
        "segmented_image",
        "disease_mask",
        "leaf_mask",
        "final_disease_mask",
        "leaf_pixels",
        "disease_pixels",
        "severity_percent",
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    severities = np.array(
        [
            row["severity_percent"]
            for row in rows
        ],
        dtype=np.float32,
    )

    print()
    print("=" * 70)
    print("SEVERITY DATASET COMPLETE")
    print("=" * 70)

    print(
        f"Successfully processed: "
        f"{len(rows)}"
    )

    if len(severities) > 0:

        print(
            f"Minimum severity : "
            f"{np.min(severities):.2f}%"
        )

        print(
            f"Maximum severity : "
            f"{np.max(severities):.2f}%"
        )

        print(
            f"Mean severity    : "
            f"{np.mean(severities):.2f}%"
        )

        print(
            f"Median severity  : "
            f"{np.median(severities):.2f}%"
        )

        print(
            f"Std deviation    : "
            f"{np.std(severities):.2f}%"
        )

    print()
    print(
        f"CSV:"
    )

    print(
        CSV_PATH
    )

    print()
    print(
        f"Leaf masks:"
    )

    print(
        LEAF_MASK_DIR
    )

    print()
    print(
        f"Disease masks:"
    )

    print(
        FINAL_DISEASE_DIR
    )

    print()
    print(
        f"Overlays:"
    )

    print(
        OVERLAY_DIR
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()