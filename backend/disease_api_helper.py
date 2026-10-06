"""
Helper functions for the disease detection API.
"""

from pathlib import Path

from ai.disease.disease_service import (
    DiseaseDetectionService
)


# ============================================================
# LOAD SERVICE ONCE
# ============================================================

_disease_service = None


def get_disease_service():
    """
    Load the disease service once and reuse it.

    This prevents the EfficientNet model from being
    loaded for every API request.
    """

    global _disease_service

    if _disease_service is None:

        _disease_service = (
            DiseaseDetectionService()
        )

    return _disease_service


# ============================================================
# PREDICT UPLOADED IMAGE
# ============================================================

def predict_uploaded_image(
    image_path
):
    """
    Run disease prediction on an uploaded image.
    """

    image_path = Path(
        image_path
    )

    service = get_disease_service()

    return service.predict(
        image_path
    )