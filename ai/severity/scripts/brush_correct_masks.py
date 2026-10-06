from pathlib import Path
import cv2
import numpy as np


# ============================================================
# PATHS
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

SAM2_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "sam2_annotations"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "corrected_masks"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMAGES TO CORRECT
# ============================================================

TARGETS = [
    "*9436*",
    "*8220*",
    "*7597*",
    "*8328*",
    "*6401*",
    "*9411*",
]


# ============================================================
# FIND IMAGES
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
            f"Image not found: {pattern}"
        )

    image_paths.append(matches[0])


# ============================================================
# GLOBAL VARIABLES
# ============================================================

current_index = 0

image = None
original_mask = None
mask = None

brush_size = 12

mouse_button = None

WINDOW_NAME = "Disease Mask Correction"

# Height of information panel
PANEL_HEIGHT = 110


# ============================================================
# LOAD CURRENT IMAGE
# ============================================================

def load_current():

    global image
    global original_mask
    global mask
    global mouse_button
    global brush_size

    image_path = image_paths[current_index]

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image:\n{image_path}"
        )

    mask_path = (
        SAM2_DIR
        / f"{image_path.stem}_disease_mask.png"
    )

    original_mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if original_mask is None:
        raise RuntimeError(
            f"SAM2 mask not found:\n{mask_path}"
        )

    # Ensure mask and image have identical dimensions
    if original_mask.shape != image.shape[:2]:

        original_mask = cv2.resize(
            original_mask,
            (
                image.shape[1],
                image.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    mask = original_mask.copy()

    mouse_button = None
    brush_size = 12

    print("\n" + "=" * 75)

    print(
        f"[{current_index + 1}/{len(image_paths)}] "
        f"{image_path.name}"
    )

    print("=" * 75)

    print("LEFT mouse drag  = ADD disease")
    print("RIGHT mouse drag = ERASE disease")
    print("Mouse wheel      = brush size")
    print("R                = reset to SAM2")
    print("S                = save corrected mask")
    print("N                = next")
    print("P                = previous")
    print("Q                = quit")

    print(
        f"Brush size: {brush_size}px"
    )


# ============================================================
# PAINT
# ============================================================

def paint(x, y, erase):

    global mask

    if erase:

        # Remove disease
        cv2.circle(
            mask,
            (x, y),
            brush_size,
            0,
            -1
        )

    else:

        # Add disease
        cv2.circle(
            mask,
            (x, y),
            brush_size,
            255,
            -1
        )


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global mouse_button
    global brush_size

    # --------------------------------------------------------
    # Mouse wheel
    # --------------------------------------------------------

    if event == cv2.EVENT_MOUSEWHEEL:

        if flags > 0:

            brush_size = min(
                brush_size + 3,
                100
            )

        else:

            brush_size = max(
                brush_size - 3,
                2
            )

        print(
            f"Brush size: {brush_size}px"
        )

        return


    # --------------------------------------------------------
    # IMPORTANT:
    # The top panel is not part of the image.
    # --------------------------------------------------------

    if y < PANEL_HEIGHT:

        return

    # Convert window coordinate → image coordinate

    image_y = y - PANEL_HEIGHT

    # Safety check

    if image_y < 0:
        return

    if image_y >= image.shape[0]:
        return

    if x < 0 or x >= image.shape[1]:
        return


    # --------------------------------------------------------
    # LEFT BUTTON
    # --------------------------------------------------------

    if event == cv2.EVENT_LBUTTONDOWN:

        mouse_button = "left"

        paint(
            x,
            image_y,
            erase=False
        )

        return


    # --------------------------------------------------------
    # LEFT DRAG
    # --------------------------------------------------------

    if event == cv2.EVENT_MOUSEMOVE:

        if mouse_button == "left":

            paint(
                x,
                image_y,
                erase=False
            )

        elif mouse_button == "right":

            paint(
                x,
                image_y,
                erase=True
            )

        return


    # --------------------------------------------------------
    # LEFT RELEASE
    # --------------------------------------------------------

    if event == cv2.EVENT_LBUTTONUP:

        if mouse_button == "left":

            mouse_button = None

        return


    # --------------------------------------------------------
    # RIGHT BUTTON
    # --------------------------------------------------------

    if event == cv2.EVENT_RBUTTONDOWN:

        mouse_button = "right"

        paint(
            x,
            image_y,
            erase=True
        )

        return


    # --------------------------------------------------------
    # RIGHT RELEASE
    # --------------------------------------------------------

    if event == cv2.EVENT_RBUTTONUP:

        if mouse_button == "right":

            mouse_button = None

        return


# ============================================================
# CREATE DISPLAY
# ============================================================

def make_display():

    canvas = image.copy()

    disease = mask > 127

    # --------------------------------------------------------
    # Green disease overlay
    # --------------------------------------------------------

    green = np.zeros_like(
        canvas
    )

    green[:, :] = (
        0,
        255,
        0
    )

    alpha = 0.45

    if np.any(disease):

        canvas[disease] = (
            (1 - alpha)
            * canvas[disease]
            +
            alpha
            * green[disease]
        ).astype(np.uint8)


    # --------------------------------------------------------
    # Disease boundary
    # --------------------------------------------------------

    mask_uint8 = (
        disease.astype(np.uint8)
        * 255
    )

    contours, _ = cv2.findContours(
        mask_uint8,
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
    # Information panel
    # --------------------------------------------------------

    panel = np.zeros(
        (
            PANEL_HEIGHT,
            canvas.shape[1],
            3
        ),
        dtype=np.uint8
    )

    image_name = (
        image_paths[current_index].name
    )

    disease_pixels = int(
        disease.sum()
    )

    cv2.putText(
        panel,
        f"[{current_index + 1}/{len(image_paths)}] "
        f"{image_name}",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1
    )

    cv2.putText(
        panel,
        f"Disease pixels: {disease_pixels:,}   "
        f"Brush: {brush_size}px",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        panel,
        "LEFT=ADD   RIGHT=ERASE   "
        "WHEEL=SIZE   R=RESET   S=SAVE",
        (10, 77),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )

    cv2.putText(
        panel,
        "N=NEXT   P=PREVIOUS   Q=QUIT",
        (10, 99),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 255),
        1
    )

    return np.vstack(
        [
            panel,
            canvas
        ]
    )


# ============================================================
# SAVE
# ============================================================

def save_mask():

    image_path = (
        image_paths[current_index]
    )

    mask_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_corrected_mask.png"
    )

    overlay_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_corrected_overlay.png"
    )

    # --------------------------------------------------------
    # Save mask
    # --------------------------------------------------------

    cv2.imwrite(
        str(mask_path),
        mask
    )

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = image.copy()

    disease = mask > 127

    green = np.zeros_like(
        image
    )

    green[:, :] = (
        0,
        255,
        0
    )

    if np.any(disease):

        overlay[disease] = (
            0.55
            * overlay[disease]
            +
            0.45
            * green[disease]
        ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
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

    print("\nSAVED:")
    print(
        f"  {mask_path.name}"
    )
    print(
        f"  {overlay_path.name}"
    )


# ============================================================
# RESET
# ============================================================

def reset_mask():

    global mask

    mask = original_mask.copy()

    print(
        "\nReset to original SAM2 mask."
    )


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

load_current()


while True:

    display = make_display()

    cv2.imshow(
        WINDOW_NAME,
        display
    )

    key = cv2.waitKey(30) & 0xFF


    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    if key in (
        ord("r"),
        ord("R")
    ):

        reset_mask()


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    elif key in (
        ord("s"),
        ord("S")
    ):

        save_mask()


    # --------------------------------------------------------
    # NEXT
    # --------------------------------------------------------

    elif key in (
        ord("n"),
        ord("N")
    ):

        current_path = (
            image_paths[current_index]
        )

        expected = (
            OUTPUT_DIR
            / f"{current_path.stem}_corrected_mask.png"
        )

        if not expected.exists():

            print(
                "\nPlease press S to save "
                "this corrected mask first."
            )

            continue

        if current_index < len(image_paths) - 1:

            current_index += 1

            load_current()

        else:

            print(
                "\nAll 6 images completed."
            )

            break


    # --------------------------------------------------------
    # PREVIOUS
    # --------------------------------------------------------

    elif key in (
        ord("p"),
        ord("P")
    ):

        if current_index > 0:

            current_index -= 1

            load_current()


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    elif key in (
        ord("q"),
        ord("Q")
    ):

        print(
            "\nCorrection tool closed."
        )

        break


cv2.destroyAllWindows()

print(
    "\nCorrected masks are saved in:"
)

print(
    OUTPUT_DIR
)