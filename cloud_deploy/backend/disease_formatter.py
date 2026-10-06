"""
Farmer-friendly disease result formatter.

Converts the technical AI prediction result into
a readable message for the farmer.
"""


def format_disease_result(result):
    """
    Convert disease AI result into a farmer-friendly message.
    """

    status = result.get("status")

    prediction = result.get(
        "prediction",
        {}
    )

    # --------------------------------------------------------
    # LOW CONFIDENCE
    # --------------------------------------------------------

    if status == "LOW_CONFIDENCE":

        disease = prediction.get(
            "disease",
            "Unknown"
        )

        confidence = prediction.get(
            "confidence",
            0
        )

        top_predictions = result.get(
            "top_predictions",
            []
        )

        message = (
            "🌱 DISEASE DETECTION\n\n"
            f"⚠️ Prediction is uncertain\n\n"
            f"Possible disease: {disease}\n"
            f"Model confidence: {confidence}%\n\n"
            "📸 Please take another photo with:\n"
            "• Good lighting\n"
            "• The leaf clearly visible\n"
            "• Minimal background\n"
            "• The affected area in focus\n\n"
            "The system will not provide a disease "
            "management recommendation from an "
            "uncertain prediction."
        )

        return message

    # --------------------------------------------------------
    # ACCEPTED PREDICTION
    # --------------------------------------------------------

    crop = prediction.get(
        "crop",
        "Unknown"
    )

    disease = prediction.get(
        "disease",
        "Unknown"
    )

    confidence = prediction.get(
        "confidence",
        0
    )

    disease_type = prediction.get(
        "type",
        "Unknown"
    )

    severity = prediction.get(
        "severity",
        "Unknown"
    )

    symptoms = prediction.get(
        "symptoms",
        []
    )

    management = prediction.get(
        "management",
        []
    )

    # --------------------------------------------------------
    # Build message
    # --------------------------------------------------------

    message = (
        "🌱 SMART AGRICULTURE\n"
        "🔬 AI DISEASE DETECTION\n\n"

        f"🌾 Crop: {crop}\n"
        f"🔍 Disease: {disease}\n"
        f"📊 Confidence: {confidence}%\n"
        f"🦠 Type: {disease_type}\n"
        f"⚠️ Severity: {severity}\n\n"

        "🔎 Symptoms:\n"
    )

    for symptom in symptoms:

        message += (
            f"• {symptom}\n"
        )

    message += "\n🛠️ Management Guidance:\n"

    for recommendation in management:

        message += (
            f"• {recommendation}\n"
        )

    message += (
        "\n⚠️ Note:\n"
        "This is an AI-based screening result. "
        "For important crop-management decisions, "
        "verify the diagnosis with a qualified "
        "agricultural professional."
    )

    return message


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_result = {

        "status": "PREDICTION_ACCEPTED",

        "prediction": {

            "crop": "Tomato",

            "disease": "Tomato Early Blight",

            "confidence": 82.4,

            "type": "Fungal disease",

            "severity": "Moderate to High",

            "symptoms": [
                "Brown circular lesions",
                "Concentric rings may form",
                "Lower leaves are commonly affected first"
            ],

            "management": [
                "Remove severely affected leaves",
                "Maintain good plant spacing and airflow",
                "Avoid prolonged leaf wetness"
            ]
        },

        "top_predictions": []
    }

    print(
        format_disease_result(
            test_result
        )
    )