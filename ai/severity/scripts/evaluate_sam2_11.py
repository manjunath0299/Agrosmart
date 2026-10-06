from pathlib import Path
import json
import csv

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\smart_agriculture")

ANNOTATION_DIR = (
    ROOT
    / "ai"
    / "severity"
    / "dataset"
    / "images"
    / "annotation_pool"
)

CHECKPOINT = (
    ROOT
    / "ai"
    / "severity"
    / "models"
    / "sam2_repo"
    / "checkpoints"
    / "sam2.1_hiera_tiny.pt"
)

CONFIG = (
    ROOT
    / "ai"
    / "severity"
    / "models"
    / "sam2_repo"
    / "sam2"
    / "configs"
    / "sam2.1"
    / "sam2.1_hiera_t.yaml"
)

OUTPUT_DIR = (
    ROOT
    / "ai"
    / "severity"
    / "results"
    / "sam2_11_images"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

NUM_POSITIVE_POINTS = 5
NUM_NEGATIVE_POINTS = 10

RANDOM_SEED = 42


# ============================================================
# FIND IMAGE FOR JSON
# ============================================================

def find_image(json_path):

    stem = json_path.stem

    extensions = [
        ".JPG",
        ".jpg",
        ".JPEG",
        ".jpeg",
        ".PNG",
        ".png",
    ]

    for extension in extensions:

        path = (
            json_path.parent
            / f"{stem}{extension}"
        )

        if path.exists():
            return path

    matches = list(
        json_path.parent.glob(
            f"{stem}.*"
        )
    )

    for path in matches:

        if path.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
        ]:
            return path

    return None


# ============================================================
# CREATE LABELME MASKS
# ============================================================

def create_masks(
    json_path,
    height,
    width,
):

    leaf_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    disease_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:

        annotation = json.load(f)

    for shape in annotation["shapes"]:

        label = shape["label"]

        points = np.array(
            shape["points"],
            dtype=np.int32,
        )

        if len(points) < 3:
            continue

        if label == "leaf":

            cv2.fillPoly(
                leaf_mask,
                [points],
                1,
            )

        elif label == "disease":

            cv2.fillPoly(
                disease_mask,
                [points],
                1,
            )

    # Disease must be inside leaf
    disease_mask = (
        disease_mask & leaf_mask
    ).astype(np.uint8)

    return leaf_mask, disease_mask


# ============================================================
# CREATE PROMPTS
# ============================================================

def create_prompts(
    leaf_mask,
    disease_mask,
    seed,
):

    rng = np.random.default_rng(seed)

    # Disease pixels
    positive_yx = np.argwhere(
        disease_mask == 1
    )

    # Healthy leaf pixels
    healthy_leaf = (
        (leaf_mask == 1)
        & (disease_mask == 0)
    )

    negative_yx = np.argwhere(
        healthy_leaf
    )

    if len(positive_yx) == 0:
        raise RuntimeError(
            "No disease pixels found."
        )

    if len(negative_yx) == 0:
        raise RuntimeError(
            "No healthy leaf pixels found."
        )

    num_positive = min(
        NUM_POSITIVE_POINTS,
        len(positive_yx),
    )

    num_negative = min(
        NUM_NEGATIVE_POINTS,
        len(negative_yx),
    )

    positive_indices = rng.choice(
        len(positive_yx),
        size=num_positive,
        replace=False,
    )

    negative_indices = rng.choice(
        len(negative_yx),
        size=num_negative,
        replace=False,
    )

    positive_yx = (
        positive_yx[positive_indices]
    )

    negative_yx = (
        negative_yx[negative_indices]
    )

    # Convert (y, x) -> (x, y)
    positive_points = np.array(
        [
            [x, y]
            for y, x in positive_yx
        ],
        dtype=np.float32,
    )

    negative_points = np.array(
        [
            [x, y]
            for y, x in negative_yx
        ],
        dtype=np.float32,
    )

    input_points = np.concatenate(
        [
            positive_points,
            negative_points,
        ],
        axis=0,
    )

    input_labels = np.concatenate(
        [
            np.ones(
                len(positive_points),
                dtype=np.int32,
            ),
            np.zeros(
                len(negative_points),
                dtype=np.int32,
            ),
        ]
    )

    return input_points, input_labels


# ============================================================
# METRICS
# ============================================================

def calculate_iou(
    prediction,
    ground_truth,
):

    intersection = np.logical_and(
        prediction,
        ground_truth,
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth,
    ).sum()

    if union == 0:
        return 0.0

    return (
        intersection / union
    )


def calculate_dice(
    prediction,
    ground_truth,
):

    intersection = np.logical_and(
        prediction,
        ground_truth,
    ).sum()

    total = (
        prediction.sum()
        + ground_truth.sum()
    )

    if total == 0:
        return 0.0

    return (
        2 * intersection / total
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SAM2 BATCH EVALUATION - 11 ANNOTATED IMAGES")
print("=" * 70)


# ============================================================
# DEVICE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:", device)

if device == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# FIND ANNOTATIONS
# ============================================================

json_files = sorted(
    ANNOTATION_DIR.glob("*.json")
)

print(
    f"\nFound {len(json_files)} annotation files."
)

if len(json_files) == 0:

    raise RuntimeError(
        "No LabelMe JSON files found."
    )


# ============================================================
# LOAD SAM2 ONCE
# ============================================================

print("\nLoading SAM2.1 Tiny...")

sam2_model = build_sam2(
    str(CONFIG),
    str(CHECKPOINT),
    device=device,
)

predictor = SAM2ImagePredictor(
    sam2_model
)

print("SAM2 loaded successfully.")


# ============================================================
# RESULTS
# ============================================================

all_results = []


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for image_number, json_path in enumerate(
    json_files,
    start=1,
):

    print("\n")
    print("=" * 70)

    print(
        f"IMAGE {image_number}/{len(json_files)}"
    )

    print(
        json_path.name
    )

    print("=" * 70)


    # --------------------------------------------------------
    # FIND IMAGE
    # --------------------------------------------------------

    image_path = find_image(
        json_path
    )

    if image_path is None:

        print(
            "WARNING: Image not found. Skipping."
        )

        continue


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image_bgr = cv2.imread(
        str(image_path)
    )

    if image_bgr is None:

        print(
            "WARNING: Could not read image. Skipping."
        )

        continue


    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    height, width = image_rgb.shape[:2]


    # --------------------------------------------------------
    # CREATE GROUND-TRUTH MASKS
    # --------------------------------------------------------

    leaf_mask, disease_mask = (
        create_masks(
            json_path,
            height,
            width,
        )
    )


    leaf_pixels = int(
        leaf_mask.sum()
    )

    disease_pixels = int(
        disease_mask.sum()
    )


    if leaf_pixels == 0:

        print(
            "WARNING: Empty leaf mask. Skipping."
        )

        continue


    ground_truth_severity = (
        disease_pixels
        / leaf_pixels
        * 100
    )


    # --------------------------------------------------------
    # CREATE PROMPTS
    # --------------------------------------------------------

    input_points, input_labels = (
        create_prompts(
            leaf_mask,
            disease_mask,
            RANDOM_SEED + image_number,
        )
    )


    # --------------------------------------------------------
    # SAM2 IMAGE
    # --------------------------------------------------------

    predictor.set_image(
        image_rgb
    )


    # --------------------------------------------------------
    # SAM2 PREDICTION
    # --------------------------------------------------------

    masks, scores, _ = (
        predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
    )


    # --------------------------------------------------------
    # EVALUATE ALL MASKS
    # --------------------------------------------------------

    ground_truth = (
        disease_mask.astype(bool)
    )

    image_results = []


    for mask_index, (
        mask,
        score,
    ) in enumerate(
        zip(masks, scores)
    ):

        mask = (
            np.asarray(mask)
            .squeeze()
            .astype(bool)
        )


        iou = calculate_iou(
            mask,
            ground_truth,
        )


        dice = calculate_dice(
            mask,
            ground_truth,
        )


        predicted_disease_pixels = int(
            mask.sum()
        )


        predicted_severity = (
            predicted_disease_pixels
            / leaf_pixels
            * 100
        )


        result = {

            "image":
                image_path.name,

            "mask":
                mask_index,

            "sam2_score":
                float(score),

            "ground_truth_severity":
                float(
                    ground_truth_severity
                ),

            "predicted_severity":
                float(
                    predicted_severity
                ),

            "iou":
                float(iou),

            "dice":
                float(dice),
        }


        image_results.append(
            result
        )


    # --------------------------------------------------------
    # BEST MASK BY IoU
    # --------------------------------------------------------

    best = max(
        image_results,
        key=lambda x: x["iou"],
    )


    all_results.append(
        best
    )


    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print(
        f"\nGround truth severity:"
        f" {ground_truth_severity:.2f}%"
    )

    print(
        "\nCandidate masks:"
    )


    for result in image_results:

        print(
            f"  Mask {result['mask']} | "
            f"IoU={result['iou']:.4f} | "
            f"Dice={result['dice']:.4f} | "
            f"Severity="
            f"{result['predicted_severity']:.2f}%"
        )


    print(
        "\nBest mask:"
    )

    print(
        f"  Mask {best['mask']}"
    )

    print(
        f"  IoU: {best['iou']:.4f}"
    )

    print(
        f"  Dice: {best['dice']:.4f}"
    )

    print(
        f"  Predicted severity:"
        f" {best['predicted_severity']:.2f}%"
    )


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "sam2_11_results.csv"
)


fieldnames = [
    "image",
    "mask",
    "sam2_score",
    "ground_truth_severity",
    "predicted_severity",
    "iou",
    "dice",
]


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        all_results
    )


# ============================================================
# SUMMARY STATISTICS
# ============================================================

if len(all_results) > 0:

    gt_severity = np.array(
        [
            r["ground_truth_severity"]
            for r in all_results
        ]
    )

    pred_severity = np.array(
        [
            r["predicted_severity"]
            for r in all_results
        ]
    )

    ious = np.array(
        [
            r["iou"]
            for r in all_results
        ]
    )

    dices = np.array(
        [
            r["dice"]
            for r in all_results
        ]
    )


    severity_mae = np.mean(
        np.abs(
            gt_severity
            - pred_severity
        )
    )


    severity_rmse = np.sqrt(
        np.mean(
            (
                gt_severity
                - pred_severity
            ) ** 2
        )
    )


    mean_iou = np.mean(
        ious
    )

    mean_dice = np.mean(
        dices
    )


    print("\n")
    print("=" * 70)
    print("FINAL 11-IMAGE SUMMARY")
    print("=" * 70)

    print(
        f"\nImages evaluated: "
        f"{len(all_results)}"
    )

    print(
        f"\nMean IoU: "
        f"{mean_iou:.4f}"
    )

    print(
        f"Mean Dice: "
        f"{mean_dice:.4f}"
    )

    print(
        f"\nSeverity MAE: "
        f"{severity_mae:.2f} percentage points"
    )

    print(
        f"Severity RMSE: "
        f"{severity_rmse:.2f} percentage points"
    )


    print(
        "\nPer-image results:"
    )


    print(
        "\n"
        "Image | GT Severity | "
        "SAM2 Severity | IoU | Dice"
    )

    print(
        "-" * 70
    )


    for result in all_results:

        print(
            f"{result['image'][:25]:25s} | "
            f"{result['ground_truth_severity']:8.2f}% | "
            f"{result['predicted_severity']:10.2f}% | "
            f"{result['iou']:.3f} | "
            f"{result['dice']:.3f}"
        )


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("BATCH EVALUATION COMPLETE")
print("=" * 70)

print(
    "\nCSV saved to:"
)

print(csv_path)

print(
    "\nOpen results with:"
)

print(
    'explorer ".\\ai\\severity\\results\\sam2_11_images"'
)