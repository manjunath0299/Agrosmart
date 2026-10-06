from pathlib import Path
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ORIGINAL_DIR = (
    PROJECT_ROOT
    / "PlantVillage-Dataset"
    / "raw"
    / "color"
    / "Tomato___Early_blight"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "leaf_mask_test_v2_batch"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEST IMAGES
# ============================================================

TEST_IDS = [
    "7637",
    "8328",
    "9419",
    "7617",
]


# ============================================================
# LEAF SEGMENTATION
# ============================================================

def create_leaf_mask(image):

    h, w = image.shape[:2]

    mask = np.zeros(
        (h, w),
        np.uint8
    )

    # Start with everything as background.
    mask[:] = cv2.GC_BGD

    # Conservative probable foreground.
    mx = int(w * 0.08)
    my = int(h * 0.08)

    rect = (
        mx,
        my,
        w - 2 * mx,
        h - 2 * my
    )

    mask[
        my:h-my,
        mx:w-mx
    ] = cv2.GC_PR_FGD

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

    # Morphological cleanup.
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

    # Keep largest connected component.
    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            leaf_mask,
            connectivity=8
        )
    )

    if n > 1:

        largest = (
            1
            + np.argmax(
                stats[1:, cv2.CC_STAT_AREA]
            )
        )

        leaf_mask = np.where(
            labels == largest,
            255,
            0
        ).astype(np.uint8)

    return leaf_mask


# ============================================================
# PROCESS
# ============================================================

print("=" * 70)
print("WHOLE-LEAF SEGMENTATION — 4 IMAGE TEST")
print("=" * 70)

for image_id in TEST_IDS:

    matches = list(
        ORIGINAL_DIR.glob(
            f"*___RS_Erly.B {image_id}.JPG"
        )
    )

    # Some filenames may have slightly different formatting.
    if not matches:
        matches = [
            p for p in ORIGINAL_DIR.glob("*")
            if p.is_file()
            and f"RS_Erly.B {image_id}" in p.name
        ]

    if not matches:
        print()
        print(f"FAILED: Could not find image {image_id}")
        continue

    image_path = matches[0]

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print(
            f"FAILED: Could not read {image_path}"
        )
        continue

    print()
    print(f"Processing: {image_path.name}")

    leaf_mask = create_leaf_mask(
        image
    )

    leaf_pixels = int(
        np.sum(leaf_mask > 0)
    )

    total_pixels = (
        image.shape[0]
        * image.shape[1]
    )

    percentage = (
        leaf_pixels
        / total_pixels
        * 100
    )

    print(
        f"Leaf area: {percentage:.2f}%"
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = image.copy()

    leaf_bool = (
        leaf_mask > 0
    )

    green = np.zeros_like(
        image
    )

    green[:] = (
        0,
        255,
        0
    )

    overlay[leaf_bool] = cv2.addWeighted(
        image[leaf_bool],
        0.55,
        green[leaf_bool],
        0.45,
        0
    )

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

    cv2.rectangle(
        overlay,
        (5, 5),
        (300, 42),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        overlay,
        f"Leaf area: {percentage:.2f}%",
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    output_path = (
        OUTPUT_DIR
        / f"{image_id}_leaf_mask_overlay.jpg"
    )

    cv2.imwrite(
        str(output_path),
        overlay,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    # --------------------------------------------------------
    # Binary mask
    # --------------------------------------------------------

    mask_path = (
        OUTPUT_DIR
        / f"{image_id}_leaf_mask.png"
    )

    cv2.imwrite(
        str(mask_path),
        leaf_mask
    )

    print(
        f"Saved: {output_path.name}"
    )


print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)

print()
print("Results:")
print(OUTPUT_DIR)