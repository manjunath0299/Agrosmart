"""
Main AI disease detection service.

Pipeline:

Image
    ↓
Preprocessing
    ↓
EfficientNetV2-B0
    ↓
38-class prediction
    ↓
Confidence gate
    ↓
Disease knowledge
    ↓
Structured result
"""

from pathlib import Path

import torch

from .disease_config import (
    CLASS_NAMES,
    NUM_CLASSES,
    CONFIDENCE_THRESHOLD,
    DEVICE,
    get_clean_class_name
)

from .disease_model import DiseaseModel

from .disease_preprocessing import (
    preprocess_image,
    preprocess_pil_image
)

from .disease_knowledge import (
    enrich_prediction
)


class DiseaseDetectionService:
    """
    Main disease detection service.

    This class provides a single interface for disease
    prediction from images.
    """

    def __init__(self):

        print("\nInitializing Disease Detection Service...")

        # Load model once
        self.model = DiseaseModel()

        print("Disease Detection Service ready.\n")

    # ========================================================
    # INTERNAL INFERENCE
    # ========================================================

    def _run_inference(self, image_tensor):
        """
        Run model inference.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Shape:
            [1, 3, 224, 224]

        Returns
        -------
        probabilities : torch.Tensor
            Shape:
            [1, 38]
        """

        image_tensor = image_tensor.to(
            DEVICE
        )

        with torch.inference_mode():

            logits = self.model.model(
                image_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

        return probabilities

    # ========================================================
    # TOP-5 PREDICTIONS
    # ========================================================

    def _get_top_predictions(
        self,
        probabilities
    ):
        """
        Extract top-5 model predictions.
        """

        top_k = min(
            5,
            NUM_CLASSES
        )

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=top_k,
            dim=1
        )

        predictions = []

        for probability, index in zip(
            top_probabilities[0],
            top_indices[0]
        ):

            class_index = index.item()

            class_name = CLASS_NAMES[
                class_index
            ]

            predictions.append({

                "class_index": class_index,

                "class_name": class_name,

                "display_name": get_clean_class_name(
                    class_name
                ),

                "confidence": round(
                    probability.item() * 100,
                    2
                )
            })

        return predictions

    # ========================================================
    # CONFIDENCE GATE
    # ========================================================

    def _apply_confidence_gate(
        self,
        confidence
    ):
        """
        Apply the engineering confidence threshold.

        >= 70%:
            PREDICTION_ACCEPTED

        < 70%:
            LOW_CONFIDENCE
        """

        if confidence >= CONFIDENCE_THRESHOLD:

            return (
                "PREDICTION_ACCEPTED",
                "The model has sufficient confidence "
                "in this prediction."
            )

        return (
            "LOW_CONFIDENCE",
            "The model is not sufficiently confident. "
            "Please provide a clearer image of the leaf."
        )

    # ========================================================
    # BUILD FINAL RESULT
    # ========================================================

    def _build_result(
        self,
        probabilities,
        image_info=None
    ):
        """
        Convert raw probabilities into the final
        structured prediction.
        """

        # ----------------------------------------------------
        # Top prediction
        # ----------------------------------------------------

        confidence_tensor, class_tensor = torch.max(
            probabilities,
            dim=1
        )

        confidence = confidence_tensor.item()

        class_index = class_tensor.item()

        class_name = CLASS_NAMES[
            class_index
        ]

        # ----------------------------------------------------
        # Confidence gate
        # ----------------------------------------------------

        status, message = (
            self._apply_confidence_gate(
                confidence
            )
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = {

            "class_index": class_index,

            "class_name": class_name,

            "display_name": get_clean_class_name(
                class_name
            ),

            "confidence": round(
                confidence * 100,
                2
            )
        }

        # ----------------------------------------------------
        # Add disease information
        # ----------------------------------------------------

        prediction = enrich_prediction(
            prediction
        )

        # ----------------------------------------------------
        # Top-5
        # ----------------------------------------------------

        top_predictions = (
            self._get_top_predictions(
                probabilities
            )
        )

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        result = {

            "status": status,

            "prediction": prediction,

            "top_predictions": top_predictions,

            "message": message
        }

        # Add image information when available
        if image_info is not None:

            result["image"] = image_info

        return result

    # ========================================================
    # PREDICT FROM FILE
    # ========================================================

    def predict(
        self,
        image_path
    ):
        """
        Predict disease from an image file.

        Parameters
        ----------
        image_path : str or Path

        Returns
        -------
        dict
        """

        image_path = Path(
            image_path
        )

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        image_tensor, original_image = (
            preprocess_image(
                image_path
            )
        )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        probabilities = self._run_inference(
            image_tensor
        )

        # ----------------------------------------------------
        # Image information
        # ----------------------------------------------------

        image_info = {

            "filename": image_path.name,

            "width": original_image.width,

            "height": original_image.height
        }

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        return self._build_result(
            probabilities,
            image_info=image_info
        )

    # ========================================================
    # PREDICT FROM PIL IMAGE
    # ========================================================

    def predict_pil(
        self,
        image
    ):
        """
        Predict disease from a PIL image.

        Useful for:
            - Telegram
            - Flask
            - FastAPI
            - Web dashboard
            - Camera input
        """

        image_tensor = (
            preprocess_pil_image(
                image
            )
        )

        probabilities = self._run_inference(
            image_tensor
        )

        image_info = {

            "width": image.width,

            "height": image.height
        }

        return self._build_result(
            probabilities,
            image_info=image_info
        )


# ============================================================
# TEST SERVICE
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DISEASE DETECTION SERVICE TEST")
    print("=" * 60)

    # Existing PlantDoc test image
    test_image = (
        Path(r"C:\smart_agriculture")
        / "ai"
        / "disease"
        / "external_test"
        / "images"
        / "Tomato leaf late blight"
        / "70.jpg"
    )

    print("\nTest image:")
    print(test_image)

    if not test_image.exists():

        print(
            "\nTest image does not exist."
        )

        print(
            "Change 'test_image' to "
            "an existing image."
        )

    else:

        # ----------------------------------------------------
        # Initialize service
        # ----------------------------------------------------

        service = (
            DiseaseDetectionService()
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        result = service.predict(
            test_image
        )

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("FINAL AI RESULT")
        print("=" * 60)

        print(
            f"\nStatus: "
            f"{result['status']}"
        )

        prediction = (
            result["prediction"]
        )

        print(
            f"\nCrop: "
            f"{prediction['crop']}"
        )

        print(
            f"Disease: "
            f"{prediction['disease']}"
        )

        print(
            f"Type: "
            f"{prediction['type']}"
        )

        print(
            f"Confidence: "
            f"{prediction['confidence']}%"
        )

        print(
            f"Severity: "
            f"{prediction['severity']}"
        )

        print(
            f"\nMessage:\n"
            f"{result['message']}"
        )

        print("\nSymptoms:")

        for symptom in prediction[
            "symptoms"
        ]:

            print(
                f"- {symptom}"
            )

        print("\nManagement:")

        for recommendation in prediction[
            "management"
        ]:

            print(
                f"- {recommendation}"
            )

        print("\nTop-5 Predictions:")

        for i, item in enumerate(
            result["top_predictions"],
            start=1
        ):

            print(
                f"{i}. "
                f"{item['display_name']} "
                f"— "
                f"{item['confidence']}%"
            )

        print("\n" + "=" * 60)