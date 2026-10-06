from pathlib import Path

from PIL import Image, UnidentifiedImageError

import torch
from torchvision import transforms

from .disease_config import (
    IMAGE_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
    is_allowed_image
)


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGE_MEAN,
        std=IMAGE_STD
    )
])


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):
    """
    Load an image from disk and convert it to RGB.

    Parameters
    ----------
    image_path : str or Path
        Path to the input image.

    Returns
    -------
    PIL.Image.Image
        RGB image.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    if not is_allowed_image(image_path):
        raise ValueError(
            f"Unsupported image format: "
            f"{image_path.suffix}"
        )

    try:

        image = Image.open(image_path)

        # Convert all images to RGB.
        # This handles grayscale/RGBA images safely.
        image = image.convert("RGB")

        return image

    except UnidentifiedImageError:

        raise ValueError(
            f"The file is not a valid image:\n{image_path}"
        )


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image_path):
    """
    Load and preprocess an image for the disease model.

    Returns
    -------
    tensor : torch.Tensor
        Shape: [1, 3, 224, 224]

    original_image : PIL.Image.Image
        Original RGB image.
    """

    original_image = load_image(image_path)

    tensor = image_transform(original_image)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor, original_image


# ============================================================
# PREPROCESS PIL IMAGE
# ============================================================

def preprocess_pil_image(image):
    """
    Preprocess an already-loaded PIL image.

    Useful later for:
    - Telegram
    - Flask/FastAPI uploads
    - camera images
    - dashboard uploads
    """

    if not isinstance(image, Image.Image):
        raise TypeError(
            "Expected a PIL.Image.Image object."
        )

    image = image.convert("RGB")

    tensor = image_transform(image)

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# IMAGE INFORMATION
# ============================================================

def get_image_info(image_path):
    """
    Return basic information about an input image.
    """

    image = load_image(image_path)

    width, height = image.size

    return {
        "width": width,
        "height": height,
        "mode": image.mode,
        "format": image.format
    }


# ============================================================
# TEST PREPROCESSING
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Disease Image Preprocessing Test")
    print("=" * 60)

    print(f"Target image size: {IMAGE_SIZE} x {IMAGE_SIZE}")

    print("\nPreprocessing module loaded successfully.")

    print("\nExpected tensor shape:")
    print(
        f"(1, 3, {IMAGE_SIZE}, {IMAGE_SIZE})"
    )

    print("=" * 60)