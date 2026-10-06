import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"E:\smart_agriculture\smart_agriculture")

SAM2_REPO = PROJECT_ROOT / "ai" / "severity" / "models" / "sam2_repo"
CHECKPOINT = SAM2_REPO / "checkpoints" / "sam2.1_hiera_tiny.pt"

CSV_PATH = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "next_annotation_candidates"
    / "remaining_141_for_annotation.csv"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "dataset"
    / "images"
    / "annotation_pool"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "sam2_annotations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SAM2 IMPORT
# ============================================================

sys.path.insert(0, str(SAM2_REPO))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# SETTINGS
# ============================================================

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

WINDOW_NAME = "SAM2 Disease Annotation Tool"

# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("SAM2 DISEASE ANNOTATION TOOL")
print("=" * 70)

print(f"Device       : {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA is not available. Running on CPU.")

print(f"Checkpoint   : {CHECKPOINT}")
print(f"CSV          : {CSV_PATH}")
print(f"Image folder : {IMAGE_DIR}")
print(f"Output       : {OUTPUT_DIR}")
print("=" * 70)


# ============================================================
# CHECK PATHS
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"\nRemaining-image CSV not found:\n{CSV_PATH}"
    )

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"\nSAM2 checkpoint not found:\n{CHECKPOINT}"
    )

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"\nImage directory not found:\n{IMAGE_DIR}"
    )


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

if "image" not in df.columns:
    raise ValueError("CSV must contain an 'image' column.")

# Remove accidental contact sheet if present
df = df[
    df["image"].astype(str).str.lower() != "contact_sheet.jpg"
].reset_index(drop=True)

print(f"\nImages in CSV: {len(df)}")


# ============================================================
# CHECK IMAGE FILES
# ============================================================

image_paths = []

for filename in df["image"]:
    path = IMAGE_DIR / str(filename)

    if path.exists():
        image_paths.append(path)
    else:
        print(f"WARNING: Missing image: {filename}")

print(f"Valid images found: {len(image_paths)}")


# ============================================================
# EXISTING MASKS
# ============================================================

existing_masks = {
    p.name.replace("_disease_mask.png", "")
    for p in OUTPUT_DIR.glob("*_disease_mask.png")
}

print(f"Existing disease masks: {len(existing_masks)}")
print(f"Images remaining in this batch: {len(image_paths)}")

print("=" * 70)


# ============================================================
# LOAD SAM2
# ============================================================

print("\nLoading SAM2 Hiera-Tiny...")

sam2_model = build_sam2(
    MODEL_CFG,
    str(CHECKPOINT),
    device=DEVICE
)

predictor = SAM2ImagePredictor(sam2_model)

print("SAM2 loaded successfully.")

# ============================================================
# GLOBAL STATE
# ============================================================

current_index = 0

image = None
display_image = None

points = []
point_labels = []

candidate_masks = None
candidate_scores = None

selected_mask_index = 0

running_prediction = False


# ============================================================
# COLORS FOR DISPLAY ONLY
# ============================================================

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_current_image():

    global image
    global display_image
    global points
    global point_labels
    global candidate_masks
    global candidate_scores
    global selected_mask_index

    path = image_paths[current_index]

    image = cv2.imread(str(path))

    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image_rgb)

    display_image = image.copy()

    points = []
    point_labels = []

    candidate_masks = None
    candidate_scores = None

    selected_mask_index = 0

    print("\n" + "=" * 70)
    print(
        f"[{current_index + 1:03d}/{len(image_paths):03d}] "
        f"{path.name}"
    )
    print("=" * 70)

    print("Left click       = disease point")
    print("Ctrl + Left      = healthy/negative point")
    print("SPACE            = run SAM2")
    print("1 / 2 / 3        = select candidate mask")
    print("S                = save selected mask")
    print("R                = reset points")
    print("N                = next image")
    print("P                = previous image")
    print("Q                = quit")


# ============================================================
# DRAW DISPLAY
# ============================================================

def create_display():

    global display_image

    canvas = image.copy()

    # --------------------------------------------------------
    # Candidate mask
    # --------------------------------------------------------

    if candidate_masks is not None:

        mask = candidate_masks[selected_mask_index]

        # Transparent mask overlay
        overlay = canvas.copy()

        overlay[mask > 0] = (
            0.6 * overlay[mask > 0]
            + 0.4 * np.array([0, 255, 0])
        ).astype(np.uint8)

        canvas = overlay

        # Mask boundary
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            canvas,
            contours,
            -1,
            GREEN,
            2
        )

    # --------------------------------------------------------
    # Draw points
    # --------------------------------------------------------

    for (x, y), label in zip(points, point_labels):

        if label == 1:
            color = GREEN
        else:
            color = BLUE

        cv2.circle(
            canvas,
            (int(x), int(y)),
            7,
            color,
            -1
        )

        cv2.circle(
            canvas,
            (int(x), int(y)),
            9,
            WHITE,
            2
        )

    # --------------------------------------------------------
    # Information panel
    # --------------------------------------------------------

    panel_height = 125

    panel = np.zeros(
        (panel_height, canvas.shape[1], 3),
        dtype=np.uint8
    )

    path = image_paths[current_index]

    cv2.putText(
        panel,
        f"[{current_index + 1}/{len(image_paths)}] {path.name}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        WHITE,
        2
    )

    cv2.putText(
        panel,
        f"Points: {len(points)}   "
        f"Positive: {sum(point_labels)}   "
        f"Negative: {len(point_labels) - sum(point_labels)}",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        1
    )

    if candidate_masks is not None:

        score = float(candidate_scores[selected_mask_index])

        cv2.putText(
            panel,
            f"Selected mask: {selected_mask_index + 1}/"
            f"{len(candidate_masks)}   "
            f"SAM2 score: {score:.4f}",
            (10, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            WHITE,
            1
        )

    cv2.putText(
        panel,
        "SPACE=Predict | 1/2/3=Mask | S=Save | "
        "R=Reset | N=Next | P=Prev | Q=Quit",
        (10, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        WHITE,
        1
    )

    return np.vstack([panel, canvas])


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global points
    global point_labels

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Panel occupies top 125 pixels
    y_image = y - 125

    if y_image < 0:
        return

    if y_image >= image.shape[0]:
        return

    # Ctrl + left click = negative point
    if flags & cv2.EVENT_FLAG_CTRLKEY:

        points.append([x, y_image])
        point_labels.append(0)

        print(
            f"Negative point: ({x}, {y_image})"
        )

    else:

        points.append([x, y_image])
        point_labels.append(1)

        print(
            f"Positive point: ({x}, {y_image})"
        )


# ============================================================
# RUN SAM2
# ============================================================

def run_prediction():

    global candidate_masks
    global candidate_scores
    global selected_mask_index

    if len(points) == 0:
        print("\nAdd at least one point first.")
        return

    print("\nRunning SAM2...")

    point_array = np.array(points, dtype=np.float32)
    label_array = np.array(point_labels, dtype=np.int32)

    with torch.inference_mode():

        if DEVICE == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                masks, scores, _ = predictor.predict(
                    point_coords=point_array,
                    point_labels=label_array,
                    multimask_output=True
                )

        else:

            masks, scores, _ = predictor.predict(
                point_coords=point_array,
                point_labels=label_array,
                multimask_output=True
            )

    candidate_masks = masks.astype(bool)
    candidate_scores = scores

    selected_mask_index = 0

    print("\nSAM2 candidates:")

    for i, score in enumerate(scores):

        area = int(candidate_masks[i].sum())

        print(
            f"  Mask {i + 1}: "
            f"score={score:.4f}, "
            f"pixels={area}"
        )


# ============================================================
# SAVE MASK
# ============================================================

def save_current_mask():

    if candidate_masks is None:
        print("\nNo mask generated.")
        return False

    path = image_paths[current_index]

    mask = candidate_masks[selected_mask_index]

    # --------------------------------------------------------
    # Save binary disease mask
    # --------------------------------------------------------

    mask_uint8 = (
        mask.astype(np.uint8) * 255
    )

    mask_path = (
        OUTPUT_DIR
        / f"{path.stem}_disease_mask.png"
    )

    cv2.imwrite(
        str(mask_path),
        mask_uint8
    )

    # --------------------------------------------------------
    # Save overlay
    # --------------------------------------------------------

    overlay = image.copy()

    # Green transparent disease region
    colored = np.zeros_like(overlay)
    colored[:, :] = (0, 255, 0)

    alpha = 0.40

    overlay[mask] = (
        (1 - alpha) * overlay[mask]
        + alpha * colored[mask]
    ).astype(np.uint8)

    # Boundary
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        GREEN,
        2
    )

    overlay_path = (
        OUTPUT_DIR
        / f"{path.stem}_overlay.png"
    )

    cv2.imwrite(
        str(overlay_path),
        overlay
    )

    print("\nSAVED:")
    print(f"  Mask    : {mask_path.name}")
    print(f"  Overlay : {overlay_path.name}")

    return True


# ============================================================
# RESET
# ============================================================

def reset_current():

    global points
    global point_labels
    global candidate_masks
    global candidate_scores
    global selected_mask_index

    points = []
    point_labels = []

    candidate_masks = None
    candidate_scores = None

    selected_mask_index = 0

    print("\nReset current image.")


# ============================================================
# MAIN
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1100,
    850
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)

load_current_image()

while True:

    screen = create_display()

    cv2.imshow(
        WINDOW_NAME,
        screen
    )

    key = cv2.waitKey(30) & 0xFF

    # --------------------------------------------------------
    # SPACE = SAM2
    # --------------------------------------------------------

    if key == 32:

        run_prediction()

    # --------------------------------------------------------
    # MASK 1
    # --------------------------------------------------------

    elif key == ord("1"):

        if candidate_masks is not None:
            selected_mask_index = 0
            print("Selected mask 1.")

    # --------------------------------------------------------
    # MASK 2
    # --------------------------------------------------------

    elif key == ord("2"):

        if candidate_masks is not None:

            if len(candidate_masks) >= 2:
                selected_mask_index = 1
                print("Selected mask 2.")

    # --------------------------------------------------------
    # MASK 3
    # --------------------------------------------------------

    elif key == ord("3"):

        if candidate_masks is not None:

            if len(candidate_masks) >= 3:
                selected_mask_index = 2
                print("Selected mask 3.")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    elif key in (ord("s"), ord("S")):

        save_current_mask()

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    elif key in (ord("r"), ord("R")):

        reset_current()

    # --------------------------------------------------------
    # NEXT
    # --------------------------------------------------------

    elif key in (ord("n"), ord("N")):

        # Require saved mask before moving forward
        current_path = image_paths[current_index]

        expected_mask = (
            OUTPUT_DIR
            / f"{current_path.stem}_disease_mask.png"
        )

        if not expected_mask.exists():

            print(
                "\nWARNING: Current image has not been saved."
            )
            print(
                "Press S to save the mask before moving."
            )
            continue

        if current_index < len(image_paths) - 1:

            current_index += 1
            load_current_image()

        else:

            print("\n======================================")
            print("ALL 141 REMAINING IMAGES COMPLETED")
            print("======================================")

            break

    # --------------------------------------------------------
    # PREVIOUS
    # --------------------------------------------------------

    elif key in (ord("p"), ord("P")):

        if current_index > 0:

            current_index -= 1
            load_current_image()

        else:

            print("\nAlready at first image.")

    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    elif key in (ord("q"), ord("Q")):

        print("\nExiting annotation tool.")
        break


cv2.destroyAllWindows()

print("\n" + "=" * 70)
print("ANNOTATION SESSION FINISHED")
print("=" * 70)

final_masks = list(
    OUTPUT_DIR.glob("*_disease_mask.png")
)

final_overlays = list(
    OUTPUT_DIR.glob("*_overlay.png")
)

print(f"Total disease masks currently: {len(final_masks)}")
print(f"Total overlays currently     : {len(final_overlays)}")
print("=" * 70)