# ============================================================
# predict.py
#
# Final Plant Disease Inference
# Domain-Adapted EfficientNetV2-B0
#
# Features:
#   - 38 PlantVillage disease classes
#   - Confidence score
#   - Low-confidence rejection
#   - Image quality check
#   - Farmer-friendly output
# ============================================================

from pathlib import Path
import sys

import torch
import torch.nn as nn
from PIL import Image, ImageStat

import timm
from torchvision import transforms


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "ai"
    / "disease"
    / "models"
    / "domain_adaptation"
    / "best_domain_adapted_model.pth"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

NUM_CLASSES = 38

CONFIDENCE_THRESHOLD = 0.70

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 3. CLASS NAMES
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
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]


# ============================================================
# 4. MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("LOADING FINAL DISEASE MODEL")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Model: EfficientNetV2-B0"
    )

    print(
        f"Classes: {NUM_CLASSES}"
    )

    print()


    model = timm.create_model(
        "tf_efficientnetv2_b0",

        pretrained=False,

        num_classes=NUM_CLASSES
    )


    checkpoint = torch.load(
        MODEL_PATH,

        map_location=DEVICE,

        weights_only=False
    )


    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint


    model.load_state_dict(
        state_dict
    )


    model.to(
        DEVICE
    )

    model.eval()


    print(
        "Model loaded successfully."
    )

    print()

    return model


# ============================================================
# 5. IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose(
    [

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        ),
    ]
)


# ============================================================
# 6. IMAGE QUALITY CHECK
# ============================================================

def check_image_quality(
    image
):

    width, height = image.size


    if width < 100 or height < 100:

        return False, (
            "Image resolution is too low."
        )


    stat = ImageStat.Stat(
        image.convert("RGB")
    )


    brightness = sum(
        stat.mean
    ) / 3


    if brightness < 20:

        return False, (
            "Image is too dark."
        )


    if brightness > 245:

        return False, (
            "Image is overexposed."
        )


    return True, "OK"


# ============================================================
# 7. FORMAT CLASS NAME
# ============================================================

def format_class_name(
    class_name
):

    return (
        class_name
        .replace("___", " - ")
        .replace("_", " ")
    )


# ============================================================
# 8. PREDICTION
# ============================================================

def predict_image(
    model,
    image_path
):

    image_path = Path(
        image_path
    )


    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_path}"
        )


    image = Image.open(
        image_path
    ).convert("RGB")


    # --------------------------------------------------------
    # Quality check
    # --------------------------------------------------------

    quality_ok, quality_message = (
        check_image_quality(
            image
        )
    )


    if not quality_ok:

        return {
            "status":
                "IMAGE_REJECTED",

            "message":
                quality_message,
        }


    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    tensor = transform(
        image
    ).unsqueeze(0).to(
        DEVICE
    )


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


    confidence, predicted_index = (
        torch.max(
            probabilities,
            dim=1
        )
    )


    confidence = (
        confidence.item()
    )


    predicted_index = (
        predicted_index.item()
    )


    predicted_class = (
        CLASS_NAMES[
            predicted_index
        ]
    )


    # --------------------------------------------------------
    # Top 5 predictions
    # --------------------------------------------------------

    top_probabilities, top_indices = (
        torch.topk(
            probabilities[0],
            k=5
        )
    )


    top_predictions = []


    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        top_predictions.append(
            {
                "class":
                    CLASS_NAMES[
                        index.item()
                    ],

                "confidence":
                    probability.item()
                    * 100,
            }
        )


    # --------------------------------------------------------
    # Confidence gate
    # --------------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        status = (
            "LOW_CONFIDENCE"
        )

        message = (
            "The model is not sufficiently "
            "confident. Please provide a "
            "clearer leaf image or consult "
            "an agricultural expert."
        )

    else:

        status = (
            "PREDICTION_ACCEPTED"
        )

        message = (
            "Prediction confidence is "
            "above the configured threshold."
        )


    return {

        "status":
            status,

        "message":
            message,

        "predicted_class":
            predicted_class,

        "predicted_disease":
            format_class_name(
                predicted_class
            ),

        "confidence":
            confidence * 100,

        "top_5":
            top_predictions,
    }


# ============================================================
# 9. MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print()
        print(
            "Usage:"
        )

        print(
            "python "
            "ai/disease/predict.py "
            "<image_path>"
        )

        print()

        print(
            "Example:"
        )

        print(
            "python "
            "ai/disease/predict.py "
            "ai/disease/test_images/"
            "tomato_field_01.jpg"
        )

        sys.exit(1)


    image_path = sys.argv[1]


    model = load_model()


    print(
        "=" * 70
    )

    print(
        "DISEASE PREDICTION"
    )

    print(
        "=" * 70
    )

    print(
        f"Image: {image_path}"
    )

    print()


    result = predict_image(
        model,
        image_path
    )


    if result[
        "status"
    ] == "IMAGE_REJECTED":

        print(
            "STATUS: IMAGE REJECTED"
        )

        print(
            result[
                "message"
            ]
        )

        return


    print(
        f"Status: "
        f"{result['status']}"
    )

    print()

    print(
        f"Prediction: "
        f"{result['predicted_disease']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']:.2f}%"
    )

    print()

    print(
        "Message:"
    )

    print(
        result[
            "message"
        ]
    )

    print()

    print(
        "-" * 70
    )

    print(
        "TOP 5 PREDICTIONS"
    )

    print(
        "-" * 70
    )


    for rank, prediction in enumerate(
        result["top_5"],
        start=1
    ):

        print(
            f"{rank}. "
            f"{format_class_name(prediction['class'])}"
            f" : "
            f"{prediction['confidence']:.2f}%"
        )


    print()

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()