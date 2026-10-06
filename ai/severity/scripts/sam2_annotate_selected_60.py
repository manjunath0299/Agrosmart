from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CSV_PATH = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "next_annotation_candidates"
    / "selected_60_for_annotation.csv"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "models"
    / "sam2_repo"
    / "checkpoints"
    / "sam2.1_hiera_tiny.pt"
)

CONFIG = (
    PROJECT_ROOT
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
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "sam2_annotations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

WINDOW_NAME = "SAM2 - Selected 60 Annotation"

POSITIVE_COLOR = (0, 255, 0)
NEGATIVE_COLOR = (255, 0, 0)

MASK_COLORS = [
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
]


# ============================================================
# LOAD CSV
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Selected CSV not found:\n{CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

if len(df) != 60:
    raise ValueError(
        f"Expected exactly 60 images, found {len(df)}"
    )


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("SAM2 SELECTED-60 ANNOTATION TOOL")
print("=" * 70)

print(f"Device: {device}")

if device == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD SAM2
# ============================================================

print("Loading SAM2...")

sam2_model = build_sam2(
    str(CONFIG),
    str(CHECKPOINT),
    device=device
)

predictor = SAM2ImagePredictor(
    sam2_model
)

print("SAM2 loaded.")
print()


# ============================================================
# GLOBAL STATE
# ============================================================

current_index = 0

image = None
display_image = None
image_rgb = None

points = []
labels = []

candidate_masks = None
candidate_scores = None

last_mask = None


# ============================================================
# LOAD IMAGE
# ============================================================

def load_current_image():

    global image
    global display_image
    global image_rgb
    global points
    global labels
    global candidate_masks
    global candidate_scores
    global last_mask

    row = df.iloc[current_index]

    image_path = Path(row["path"])

    print()
    print("-" * 70)
    print(
        f"[{current_index + 1}/60] "
        f"Candidate #{int(row['candidate_number'])}"
    )
    print(image_path.name)

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image:\n{image_path}"
        )

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    display_image = image.copy()

    points = []
    labels = []

    candidate_masks = None
    candidate_scores = None

    last_mask = None

    predictor.set_image(image_rgb)


# ============================================================
# DRAW POINTS
# ============================================================

def redraw():

    global display_image

    display_image = image.copy()

    # Draw positive/negative points
    for point, label in zip(points, labels):

        x, y = point

        if label == 1:
            color = POSITIVE_COLOR
        else:
            color = NEGATIVE_COLOR

        cv2.circle(
            display_image,
            (x, y),
            7,
            color,
            -1
        )

        cv2.circle(
            display_image,
            (x, y),
            9,
            (255, 255, 255),
            2
        )

    # Draw selected mask
    if last_mask is not None:

        mask = last_mask.astype(bool)

        overlay = display_image.copy()

        overlay[mask] = (
            0.55 * overlay[mask]
            + 0.45 * np.array(
                MASK_COLORS[0],
                dtype=np.float32
            )
        ).astype(np.uint8)

        display_image = overlay

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            display_image,
            contours,
            -1,
            (0, 255, 255),
            2
        )

    # Header
    row = df.iloc[current_index]

    header = (
        f"{current_index + 1}/60 | "
        f"#{int(row['candidate_number'])} | "
        f"{row['selection']} | "
        f"score {float(row['triage_score']):.1f}"
    )

    cv2.rectangle(
        display_image,
        (0, 0),
        (display_image.shape[1], 45),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        display_image,
        header,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# RUN SAM2
# ============================================================

def run_sam2():

    global candidate_masks
    global candidate_scores
    global last_mask

    if len(points) == 0:

        print(
            "Add at least one positive point first."
        )

        return

    point_array = np.array(
        points,
        dtype=np.float32
    )

    label_array = np.array(
        labels,
        dtype=np.int32
    )

    with torch.inference_mode():

        if device == "cuda":

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

    candidate_masks = masks
    candidate_scores = scores

    best = int(
        np.argmax(scores)
    )

    last_mask = masks[best]

    print()
    print("SAM2 candidates:")

    for i, score in enumerate(scores):

        print(
            f"  {i + 1}: score={score:.4f}"
        )

    print(
        f"Current mask: {best + 1}"
    )

    redraw()


# ============================================================
# SAVE MASK
# ============================================================

def save_mask():

    global last_mask

    if last_mask is None:

        print(
            "No mask available. Press SPACE first."
        )

        return

    row = df.iloc[current_index]

    image_name = Path(
        row["image"]
    ).stem

    output_path = (
        OUTPUT_DIR
        / f"{image_name}_disease_mask.png"
    )

    # Safety: NEVER overwrite an existing mask.
    if output_path.exists():

        print()
        print("WARNING:")
        print("Mask already exists:")
        print(output_path)
        print("Skipping save.")
        return

    mask_uint8 = (
        last_mask.astype(np.uint8)
        * 255
    )

    cv2.imwrite(
        str(output_path),
        mask_uint8
    )

    # Save a preview overlay
    overlay = image.copy()

    mask_bool = last_mask.astype(bool)

    overlay[mask_bool] = (
        0.55 * overlay[mask_bool]
        + 0.45 * np.array(
            (0, 0, 255),
            dtype=np.float32
        )
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 255),
        2
    )

    overlay_path = (
        OUTPUT_DIR
        / f"{image_name}_disease_overlay.jpg"
    )

    cv2.imwrite(
        str(overlay_path),
        overlay
    )

    print()
    print("SAVED:")
    print(output_path)
    print(overlay_path)


# ============================================================
# CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # CTRL + left click = negative point
    if flags & cv2.EVENT_FLAG_CTRLKEY:

        points.append((x, y))
        labels.append(0)

        print(
            f"Negative point: ({x}, {y})"
        )

    else:

        points.append((x, y))
        labels.append(1)

        print(
            f"Positive point: ({x}, {y})"
        )

    redraw()


# ============================================================
# LOAD FIRST
# ============================================================

load_current_image()
redraw()

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# ============================================================
# MAIN LOOP
# ============================================================

print()
print("=" * 70)
print("CONTROLS")
print("=" * 70)
print("Left click       = disease positive point")
print("CTRL + left      = healthy negative point")
print("SPACE            = run SAM2")
print("1 / 2 / 3        = select candidate mask")
print("S                = save current mask")
print("R                = reset points/mask")
print("N                = save + next")
print("P                = previous image")
print("Q                = quit")
print("=" * 70)


while True:

    cv2.imshow(
        WINDOW_NAME,
        display_image
    )

    key = cv2.waitKey(30) & 0xFF

    # --------------------------------------------------------
    # SPACE
    # --------------------------------------------------------

    if key == 32:

        run_sam2()

    # --------------------------------------------------------
    # MASK 1
    # --------------------------------------------------------

    elif key == ord("1"):

        if candidate_masks is not None:

            last_mask = candidate_masks[0]

            print("Selected mask 1.")

            redraw()

    # --------------------------------------------------------
    # MASK 2
    # --------------------------------------------------------

    elif key == ord("2"):

        if candidate_masks is not None:

            last_mask = candidate_masks[1]

            print("Selected mask 2.")

            redraw()

    # --------------------------------------------------------
    # MASK 3
    # --------------------------------------------------------

    elif key == ord("3"):

        if candidate_masks is not None:

            last_mask = candidate_masks[2]

            print("Selected mask 3.")

            redraw()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    elif key in (ord("s"), ord("S")):

        save_mask()

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    elif key in (ord("r"), ord("R")):

        points = []
        labels = []

        candidate_masks = None
        candidate_scores = None
        last_mask = None

        redraw()

        print("Reset.")

    # --------------------------------------------------------
    # NEXT
    # --------------------------------------------------------

    elif key in (ord("n"), ord("N")):

        save_mask()

        if current_index < len(df) - 1:

            current_index += 1

            load_current_image()
            redraw()

        else:

            print()
            print(
                "Reached the final selected image."
            )

    # --------------------------------------------------------
    # PREVIOUS
    # --------------------------------------------------------

    elif key in (ord("p"), ord("P")):

        if current_index > 0:

            current_index -= 1

            load_current_image()
            redraw()

        else:

            print(
                "Already at first image."
            )

    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    elif key in (ord("q"), ord("Q")):

        print("Exiting.")
        break


cv2.destroyAllWindows()