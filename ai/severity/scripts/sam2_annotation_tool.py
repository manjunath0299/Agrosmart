import re
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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

CONFIG = (
    SAM2_REPO
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

WINDOW_NAME = "SAM2 Disease Annotation"

# Maximum display size.
# The original image is NOT changed.
MAX_DISPLAY_WIDTH = 1100
MAX_DISPLAY_HEIGHT = 750

# Point colors in BGR format
DISEASE_POINT_COLOR = (0, 255, 0)   # Green
HEALTHY_POINT_COLOR = (255, 0, 0)   # Blue

# Disease mask overlay
MASK_COLOR = (0, 0, 255)            # Red

MASK_ALPHA = 0.45


# ============================================================
# NATURAL SORT
# ============================================================

def natural_sort_key(path):
    """
    Sort:

    leaf_1
    leaf_2
    leaf_9
    leaf_10
    leaf_11
    ...
    leaf_100

    instead of alphabetical sorting:

    leaf_1
    leaf_10
    leaf_100
    leaf_101
    ...
    leaf_2
    """
    name = path.name

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


# ============================================================
# FIND IMAGES
# ============================================================

def get_images():
    """
    Find only original leaf images.

    Ignore:
    - contact sheets
    - overlays
    - masks
    - JSON files
    """

    valid_extensions = {".jpg", ".jpeg", ".png"}

    images = []

    for path in IMAGE_DIR.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() not in valid_extensions:
            continue

        name_lower = path.name.lower()

        # Ignore generated files
        if "contact_sheet" in name_lower:
            continue

        if "overlay" in name_lower:
            continue

        if "mask" in name_lower:
            continue

        images.append(path)

    images.sort(key=natural_sort_key)

    return images


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print()
        print("=" * 60)
        print("CUDA ENABLED")
        print("GPU:", torch.cuda.get_device_name(0))
        print("PyTorch:", torch.__version__)
        print("CUDA:", torch.version.cuda)
        print("=" * 60)
        print()

        return device

    print()
    print("=" * 60)
    print("WARNING: CUDA NOT AVAILABLE")
    print("SAM2 will run on CPU.")
    print("=" * 60)
    print()

    return torch.device("cpu")


# ============================================================
# LOAD SAM2
# ============================================================

def load_sam2(device):

    print("Loading SAM2...")
    print("Config:", CONFIG)
    print("Checkpoint:", CHECKPOINT)

    if not CONFIG.exists():
        raise FileNotFoundError(
            f"SAM2 config not found:\n{CONFIG}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found:\n{CHECKPOINT}"
        )

    model = build_sam2(
        str(CONFIG),
        str(CHECKPOINT),
        device=device,
    )

    predictor = SAM2ImagePredictor(model)

    print("SAM2 loaded successfully.")
    print()

    return predictor


# ============================================================
# DISPLAY RESIZE
# ============================================================

def calculate_display_scale(width, height):

    scale_x = MAX_DISPLAY_WIDTH / width
    scale_y = MAX_DISPLAY_HEIGHT / height

    scale = min(scale_x, scale_y, 1.0)

    return scale


# ============================================================
# ORIGINAL <-> DISPLAY COORDINATES
# ============================================================

def display_to_original(x, y, scale):

    original_x = int(round(x / scale))
    original_y = int(round(y / scale))

    return original_x, original_y


def original_to_display(x, y, scale):

    display_x = int(round(x * scale))
    display_y = int(round(y * scale))

    return display_x, display_y


# ============================================================
# CREATE DISPLAY IMAGE
# ============================================================

def make_display_image(
    image,
    disease_points,
    healthy_points,
    selected_mask,
    scale,
    current_mask_index,
    num_masks,
    image_index,
    total_images,
    image_path,
):

    display = image.copy()

    # --------------------------------------------------------
    # Disease mask
    # --------------------------------------------------------

    if selected_mask is not None:

        mask_bool = selected_mask.astype(bool)

        # Create red overlay
        overlay = np.zeros_like(display)
        overlay[:, :] = MASK_COLOR

        display[mask_bool] = cv2.addWeighted(
            display[mask_bool],
            1.0 - MASK_ALPHA,
            overlay[mask_bool],
            MASK_ALPHA,
            0,
        )

        # Draw mask contour
        mask_uint8 = (
            mask_bool.astype(np.uint8) * 255
        )

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            display,
            contours,
            -1,
            MASK_COLOR,
            2,
        )

    # --------------------------------------------------------
    # Disease points
    # --------------------------------------------------------

    for x, y in disease_points:

        dx, dy = original_to_display(
            x,
            y,
            scale,
        )

        cv2.circle(
            display,
            (dx, dy),
            7,
            DISEASE_POINT_COLOR,
            -1,
        )

        cv2.circle(
            display,
            (dx, dy),
            9,
            (255, 255, 255),
            2,
        )

    # --------------------------------------------------------
    # Healthy points
    # --------------------------------------------------------

    for x, y in healthy_points:

        dx, dy = original_to_display(
            x,
            y,
            scale,
        )

        cv2.circle(
            display,
            (dx, dy),
            7,
            HEALTHY_POINT_COLOR,
            -1,
        )

        cv2.circle(
            display,
            (dx, dy),
            9,
            (255, 255, 255),
            2,
        )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    if scale != 1.0:

        display = cv2.resize(
            display,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    # --------------------------------------------------------
    # Information panel
    # --------------------------------------------------------

    panel_height = 150

    panel = np.zeros(
        (panel_height, display.shape[1], 3),
        dtype=np.uint8,
    )

    # Image name
    name_text = (
        f"{image_index + 1}/{total_images}  "
        f"{image_path.name}"
    )

    cv2.putText(
        panel,
        name_text,
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Points
    cv2.putText(
        panel,
        f"GREEN disease points: {len(disease_points)}    "
        f"BLUE healthy points: {len(healthy_points)}",
        (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Mask
    if selected_mask is None:

        mask_text = "Selected mask: NONE"

    else:

        mask_text = (
            f"Selected mask: "
            f"{current_mask_index + 1}/{num_masks}"
        )

    cv2.putText(
        panel,
        mask_text,
        (15, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Instructions
    cv2.putText(
        panel,
        "Click=disease | CTRL+Click=healthy | SPACE=SAM2",
        (15, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        "1/2/3=mask | S=save | R=reset | N/P=next/previous | Q=quit",
        (15, 137),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return np.vstack(
        [display, panel]
    )


# ============================================================
# SAVE MASK
# ============================================================

def save_results(
    image,
    mask,
    image_path,
    disease_points,
    healthy_points,
):

    if mask is None:

        print("No mask selected. Nothing saved.")
        return

    mask_bool = mask.astype(bool)

    # --------------------------------------------------------
    # Disease mask
    # --------------------------------------------------------

    mask_image = (
        mask_bool.astype(np.uint8) * 255
    )

    mask_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_disease_mask.png"
    )

    cv2.imwrite(
        str(mask_path),
        mask_image,
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = image.copy()

    red_layer = np.zeros_like(image)
    red_layer[:, :] = MASK_COLOR

    overlay[mask_bool] = cv2.addWeighted(
        overlay[mask_bool],
        1.0 - MASK_ALPHA,
        red_layer[mask_bool],
        MASK_ALPHA,
        0,
    )

    # Draw contours

    contours, _ = cv2.findContours(
        mask_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        MASK_COLOR,
        2,
    )

    # Draw points on saved overlay

    for x, y in disease_points:

        cv2.circle(
            overlay,
            (x, y),
            8,
            DISEASE_POINT_COLOR,
            -1,
        )

    for x, y in healthy_points:

        cv2.circle(
            overlay,
            (x, y),
            8,
            HEALTHY_POINT_COLOR,
            -1,
        )

    overlay_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_overlay.jpg"
    )

    cv2.imwrite(
        str(overlay_path),
        overlay,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    print()
    print("=" * 60)
    print("SAVED")
    print("Image :", image_path.name)
    print("Mask  :", mask_path)
    print("Overlay:", overlay_path)
    print("Disease points:", len(disease_points))
    print("Healthy points:", len(healthy_points))
    print("=" * 60)
    print()


# ============================================================
# MOUSE CALLBACK
# ============================================================

class AnnotationState:

    def __init__(self):

        self.image = None
        self.scale = 1.0

        self.disease_points = []
        self.healthy_points = []

        self.masks = None
        self.scores = None

        self.selected_mask_index = 0

        self.image_path = None
        self.image_index = 0
        self.total_images = 0

        self.predictor = None

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param,
    ):

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Ignore clicks in bottom information panel
        display_height = int(
            self.image.shape[0] * self.scale
        )

        if y >= display_height:
            return

        original_x, original_y = display_to_original(
            x,
            y,
            self.scale,
        )

        # Keep coordinates inside image
        original_x = max(
            0,
            min(
                original_x,
                self.image.shape[1] - 1,
            ),
        )

        original_y = max(
            0,
            min(
                original_y,
                self.image.shape[0] - 1,
            ),
        )

        # CTRL = healthy point
        if flags & cv2.EVENT_FLAG_CTRLKEY:

            self.healthy_points.append(
                (original_x, original_y)
            )

            print(
                f"Healthy point: "
                f"({original_x}, {original_y})"
            )

        # Normal click = disease point
        else:

            self.disease_points.append(
                (original_x, original_y)
            )

            print(
                f"Disease point: "
                f"({original_x}, {original_y})"
            )


# ============================================================
# RUN SAM2
# ============================================================

def run_prediction(state):

    total_points = (
        len(state.disease_points)
        + len(state.healthy_points)
    )

    if total_points == 0:

        print(
            "Add disease/healthy points first."
        )

        return

    points = []
    labels = []

    # Disease = label 1
    for point in state.disease_points:

        points.append(point)
        labels.append(1)

    # Healthy = label 0
    for point in state.healthy_points:

        points.append(point)
        labels.append(0)

    point_coords = np.array(
        points,
        dtype=np.float32,
    )

    point_labels = np.array(
        labels,
        dtype=np.int32,
    )

    print()
    print("Running SAM2...")
    print(
        f"Disease points: {len(state.disease_points)}"
    )
    print(
        f"Healthy points: {len(state.healthy_points)}"
    )

    state.predictor.set_image(
        cv2.cvtColor(
            state.image,
            cv2.COLOR_BGR2RGB,
        )
    )

    masks, scores, _ = state.predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    # Ensure correct shape
    masks = np.asarray(masks)

    if masks.ndim == 2:
        masks = masks[None, ...]

    state.masks = masks.astype(bool)
    state.scores = np.asarray(scores)

    # Start with highest SAM2 score
    state.selected_mask_index = int(
        np.argmax(state.scores)
    )

    print()
    print("SAM2 candidates:")

    for i, score in enumerate(state.scores):

        area = (
            np.sum(state.masks[i])
            / state.masks[i].size
            * 100
        )

        print(
            f"Mask {i + 1}: "
            f"score={score:.4f}, "
            f"area={area:.2f}%"
        )

    print(
        f"Selected mask: "
        f"{state.selected_mask_index + 1}"
    )

    print()


# ============================================================
# RESET
# ============================================================

def reset_state(state):

    state.disease_points = []
    state.healthy_points = []

    state.masks = None
    state.scores = None

    state.selected_mask_index = 0

    print("Annotation reset.")


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(state, image_path, image_index, total_images):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"Could not read image:\n{image_path}"
        )

    state.image = image
    state.image_path = image_path
    state.image_index = image_index
    state.total_images = total_images

    state.disease_points = []
    state.healthy_points = []

    state.masks = None
    state.scores = None
    state.selected_mask_index = 0

    state.scale = calculate_display_scale(
        image.shape[1],
        image.shape[0],
    )

    print()
    print("=" * 60)
    print(
        f"IMAGE {image_index + 1}/{total_images}"
    )
    print(image_path.name)
    print(
        f"Original size: "
        f"{image.shape[1]} x {image.shape[0]}"
    )
    print(
        f"Display scale: {state.scale:.3f}"
    )
    print("=" * 60)
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("SAM2 DISEASE ANNOTATION TOOL")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_DIR}"
        )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"SAM2 checkpoint not found:\n{CHECKPOINT}"
        )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    images = get_images()

    if len(images) == 0:

        raise RuntimeError(
            "No images found in annotation_pool."
        )

    print(
        f"Found {len(images)} images."
    )

    print()
    print("First 15 images in correct order:")

    for i, image_path in enumerate(
        images[:15]
    ):

        print(
            f"{i + 1:3d}. {image_path.name}"
        )

    print()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Load SAM2
    # --------------------------------------------------------

    predictor = load_sam2(
        device
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    state = AnnotationState()

    state.predictor = predictor

    # --------------------------------------------------------
    # Open window
    # --------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    # --------------------------------------------------------
    # START AT IMAGE 1
    # --------------------------------------------------------

    current_index = 0

    load_image(
        state,
        images[current_index],
        current_index,
        len(images),
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        state.mouse_callback,
    )

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    while True:

        # Selected mask
        selected_mask = None

        if (
            state.masks is not None
            and len(state.masks) > 0
        ):

            selected_mask = state.masks[
                state.selected_mask_index
            ]

        display = make_display_image(
            image=state.image,
            disease_points=state.disease_points,
            healthy_points=state.healthy_points,
            selected_mask=selected_mask,
            scale=state.scale,
            current_mask_index=state.selected_mask_index,
            num_masks=(
                len(state.masks)
                if state.masks is not None
                else 0
            ),
            image_index=state.image_index,
            total_images=state.total_images,
            image_path=state.image_path,
        )

        cv2.imshow(
            WINDOW_NAME,
            display,
        )

        key = cv2.waitKey(30) & 0xFF

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if key in (
            ord("q"),
            ord("Q"),
        ):

            print("Exiting...")
            break

        # ----------------------------------------------------
        # RUN SAM2
        # ----------------------------------------------------

        elif key == ord(" "):

            run_prediction(state)

        # ----------------------------------------------------
        # MASK 1
        # ----------------------------------------------------

        elif key == ord("1"):

            if (
                state.masks is not None
                and len(state.masks) >= 1
            ):

                state.selected_mask_index = 0

                print(
                    "Selected mask 1"
                )

        # ----------------------------------------------------
        # MASK 2
        # ----------------------------------------------------

        elif key == ord("2"):

            if (
                state.masks is not None
                and len(state.masks) >= 2
            ):

                state.selected_mask_index = 1

                print(
                    "Selected mask 2"
                )

        # ----------------------------------------------------
        # MASK 3
        # ----------------------------------------------------

        elif key == ord("3"):

            if (
                state.masks is not None
                and len(state.masks) >= 3
            ):

                state.selected_mask_index = 2

                print(
                    "Selected mask 3"
                )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        elif key in (
            ord("s"),
            ord("S"),
        ):

            if state.masks is None:

                print(
                    "No SAM2 mask available."
                )

            else:

                save_results(
                    image=state.image,
                    mask=state.masks[
                        state.selected_mask_index
                    ],
                    image_path=state.image_path,
                    disease_points=state.disease_points,
                    healthy_points=state.healthy_points,
                )

        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------

        elif key in (
            ord("r"),
            ord("R"),
        ):

            reset_state(state)

        # ----------------------------------------------------
        # NEXT IMAGE
        # ----------------------------------------------------

        elif key in (
            ord("n"),
            ord("N"),
        ):

            if current_index < len(images) - 1:

                current_index += 1

                load_image(
                    state,
                    images[current_index],
                    current_index,
                    len(images),
                )

            else:

                print(
                    "Already at the last image."
                )

        # ----------------------------------------------------
        # PREVIOUS IMAGE
        # ----------------------------------------------------

        elif key in (
            ord("p"),
            ord("P"),
        ):

            if current_index > 0:

                current_index -= 1

                load_image(
                    state,
                    images[current_index],
                    current_index,
                    len(images),
                )

            else:

                print(
                    "Already at the first image."
                )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cv2.destroyAllWindows()

    print()
    print("SAM2 annotation tool closed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()