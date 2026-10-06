from pathlib import Path
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

# C:\smart_agriculture
BASE_DIR = Path(__file__).resolve().parents[2]

# Disease AI directory
DISEASE_DIR = BASE_DIR / "ai" / "disease"

# Trained domain-adapted model
MODEL_PATH = (
    DISEASE_DIR
    / "models"
    / "domain_adaptation"
    / "best_domain_adapted_model.pth"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "tf_efficientnetv2_b0"

NUM_CLASSES = 38

IMAGE_SIZE = 224

# Engineering confidence gate
# >= 70%  -> accepted prediction
# < 70%   -> uncertain prediction
CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

# ImageNet normalization used during model training
IMAGE_MEAN = [0.485, 0.456, 0.406]

IMAGE_STD = [0.229, 0.224, 0.225]


# Supported image formats
ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# PLANT DISEASE CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",

    "Blueberry___healthy",

    "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",

    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",

    "Orange___Haunglongbing_(Citrus_greening)",

    "Peach___Bacterial_spot",
    "Peach___healthy",

    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",

    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",

    "Raspberry___healthy",

    "Soybean___healthy",

    "Squash___Powdery_mildew",

    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",

    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]


# ============================================================
# SAFETY CHECKS
# ============================================================

if len(CLASS_NAMES) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(CLASS_NAMES)}."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_allowed_image(file_path):
    """
    Check whether the supplied file has a supported
    image extension.
    """

    path = Path(file_path)

    return path.suffix.lower() in ALLOWED_EXTENSIONS


def get_class_name(label):
    """
    Convert numeric model output into the corresponding
    PlantVillage class name.
    """

    if label < 0 or label >= NUM_CLASSES:
        raise ValueError(f"Invalid class label: {label}")

    return CLASS_NAMES[label]


def get_clean_class_name(class_name):
    """
    Convert PlantVillage naming format into a cleaner
    human-readable disease name.
    """

    return class_name.replace("___", " - ").replace("_", " ")