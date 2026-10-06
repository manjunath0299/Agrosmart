from pathlib import Path
import sys
import cv2
import numpy as np
import torch

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"E:\smart_agriculture\smart_agriculture")

SAM2_REPO = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "models"
    / "sam2_repo"
)

CHECKPOINT = (
    SAM2_REPO
    / "checkpoints"
    / "sam2.1_hiera_tiny.pt"
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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# TARGET IMAGES
# ============================================================

TARGETS = [
    "*9436*",
    "*8220*",
    "*8328*",
]

# ============================================================
# SAM2
# ============================================================

sys.path.insert(
    0,
    str(SAM2_REPO)
)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("SAM2 CORRECTION TOOL")
print("=" * 70)

print("Device:", DEVICE)

if DEVICE == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# FIND TARGET IMAGES
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
            f"Could not find image matching {pattern}"
        )

    image_paths.append(matches[0])


print("\nImages to correct:")

for p in image_paths:
    print(" ", p.name)


# ============================================================
# LOAD SAM2
# ============================================================

print("\nLoading SAM2...")

sam2_model = build_sam2(
    MODEL_CFG,
    str(CHECKPOINT),
    device=DEVICE
)

predictor = SAM2ImagePredictor(
    sam2_model
)

print("SAM2 loaded.")


# ============================================================
# GLOBAL STATE
# ============================================================

current_index = 0

image = None
points = []
labels = []

masks = None
scores = None

selected = 0


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image():

    global image
    global points
    global labels
    global masks
    global scores
    global selected

    path = image_paths[current_index]

    image = cv2.imread(
        str(path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read {path}"
        )

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    predictor.set_image(rgb)

    points = []
    labels = []

    masks = None
    scores = None

    selected = 0

    print("\n" + "=" * 70)
    print(
        f"[{current_index + 1}/3] {path.name}"
    )
    print("=" * 70)

    print("LEFT CLICK       = disease")
    print("CTRL + LEFT      = healthy")
    print("SPACE            = run SAM2")
    print("1 / 2 / 3        = choose mask")
    print("S                = overwrite/save")
    print("R                = reset")
    print("N                = next")
    print("P                = previous")
    print("Q                = quit")


# ============================================================
# DISPLAY
# ============================================================

def make_display():

    canvas = image.copy()

    # --------------------------------------------------------
    # Display selected mask
    # --------------------------------------------------------

    if masks is not None:

        mask = masks[selected]

        overlay = canvas.copy()

        green = np.zeros_like(
            canvas
        )

        green[:, :] = (
            0,
            255,
            0
        )

        alpha = 0.40

        overlay[mask] = (
            (1 - alpha)
            * overlay[mask]
            +
            alpha
            * green[mask]
        ).astype(np.uint8)

        canvas = overlay

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            canvas,
            contours,
            -1,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------------
    # Points
    # --------------------------------------------------------

    for (x, y), label in zip(
        points,
        labels
    ):

        color = (
            (0, 255, 0)
            if label == 1
            else
            (255, 0, 0)
        )

        cv2.circle(
            canvas,
            (x, y),
            7,
            color,
            -1
        )

        cv2.circle(
            canvas,
            (x, y),
            9,
            (255, 255, 255),
            2
        )

    # --------------------------------------------------------
    # Information panel
    # --------------------------------------------------------

    panel_height = 115

    panel = np.zeros(
        (
            panel_height,
            canvas.shape[1],
            3
        ),
        dtype=np.uint8
    )

    path = image_paths[current_index]

    cv2.putText(
        panel,
        f"[{current_index + 1}/3] {path.name}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        panel,
        f"Positive: {sum(labels)}  "
        f"Negative: {len(labels)-sum(labels)}",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    if masks is not None:

        cv2.putText(
            panel,
            f"Mask {selected + 1}/"
            f"{len(masks)}   "
            f"Score: {scores[selected]:.4f}",
            (10, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

    cv2.putText(
        panel,
        "SPACE Predict | 1/2/3 Mask | "
        "S Save | R Reset | N Next | Q Quit",
        (10, 103),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    return np.vstack(
        [panel, canvas]
    )


# ============================================================
# MOUSE
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Account for information panel
    y -= 115

    if y < 0:
        return

    if y >= image.shape[0]:
        return

    if flags & cv2.EVENT_FLAG_CTRLKEY:

        points.append(
            [x, y]
        )

        labels.append(0)

        print(
            f"Negative: ({x}, {y})"
        )

    else:

        points.append(
            [x, y]
        )

        labels.append(1)

        print(
            f"Positive: ({x}, {y})"
        )


# ============================================================
# RUN SAM2
# ============================================================

def predict():

    global masks
    global scores
    global selected

    if len(points) == 0:

        print(
            "\nAdd disease points first."
        )

        return

    print(
        "\nRunning SAM2..."
    )

    point_array = np.array(
        points,
        dtype=np.float32
    )

    label_array = np.array(
        labels,
        dtype=np.int32
    )

    with torch.inference_mode():

        if DEVICE == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                m, s, _ = predictor.predict(
                    point_coords=point_array,
                    point_labels=label_array,
                    multimask_output=True
                )

        else:

            m, s, _ = predictor.predict(
                point_coords=point_array,
                point_labels=label_array,
                multimask_output=True
            )

    masks = m.astype(bool)
    scores = s
    selected = 0

    print("\nCandidates:")

    for i, score in enumerate(scores):

        print(
            f"Mask {i+1}: "
            f"score={score:.4f}, "
            f"pixels={masks[i].sum()}"
        )


# ============================================================
# SAVE / OVERWRITE
# ============================================================

def save():

    if masks is None:

        print(
            "\nNo mask available."
        )

        return

    path = image_paths[current_index]

    mask = masks[selected]

    mask_uint8 = (
        mask.astype(np.uint8)
        * 255
    )

    mask_path = (
        OUTPUT_DIR
        / f"{path.stem}_disease_mask.png"
    )

    overlay_path = (
        OUTPUT_DIR
        / f"{path.stem}_overlay.png"
    )

    # --------------------------------------------------------
    # Overwrite existing mask
    # --------------------------------------------------------

    cv2.imwrite(
        str(mask_path),
        mask_uint8
    )

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = image.copy()

    green = np.zeros_like(
        image
    )

    green[:, :] = (
        0,
        255,
        0
    )

    alpha = 0.40

    overlay[mask] = (
        (1 - alpha)
        * overlay[mask]
        +
        alpha
        * green[mask]
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
        (0, 255, 0),
        2
    )

    cv2.imwrite(
        str(overlay_path),
        overlay
    )

    print("\nCORRECTED AND SAVED:")
    print(
        "Mask:",
        mask_path.name
    )

    print(
        "Overlay:",
        overlay_path.name
    )


# ============================================================
# RESET
# ============================================================

def reset():

    global points
    global labels
    global masks
    global scores
    global selected

    points = []
    labels = []

    masks = None
    scores = None

    selected = 0

    print(
        "\nPoints and candidates reset."
    )


# ============================================================
# MAIN LOOP
# ============================================================

WINDOW = "SAM2 Correction - 3 Images"

cv2.namedWindow(
    WINDOW,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW,
    1100,
    850
)

cv2.setMouseCallback(
    WINDOW,
    mouse_callback
)

load_image()

while True:

    screen = make_display()

    cv2.imshow(
        WINDOW,
        screen
    )

    key = cv2.waitKey(30) & 0xFF

    # SPACE
    if key == 32:

        predict()

    # MASK 1
    elif key == ord("1"):

        if masks is not None:
            selected = 0

    # MASK 2
    elif key == ord("2"):

        if masks is not None and len(masks) >= 2:
            selected = 1

    # MASK 3
    elif key == ord("3"):

        if masks is not None and len(masks) >= 3:
            selected = 2

    # SAVE
    elif key in (
        ord("s"),
        ord("S")
    ):

        save()

    # RESET
    elif key in (
        ord("r"),
        ord("R")
    ):

        reset()

    # NEXT
    elif key in (
        ord("n"),
        ord("N")
    ):

        if current_index < 2:

            current_index += 1
            load_image()

        else:

            print(
                "\nAll 3 images processed."
            )

    # PREVIOUS
    elif key in (
        ord("p"),
        ord("P")
    ):

        if current_index > 0:

            current_index -= 1
            load_image()

    # QUIT
    elif key in (
        ord("q"),
        ord("Q")
    ):

        break


cv2.destroyAllWindows()

print("\nCorrection tool closed.")