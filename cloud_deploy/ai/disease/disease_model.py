import torch
import timm
import torch.nn as nn

from .disease_config import (
    MODEL_NAME,
    MODEL_PATH,
    NUM_CLASSES,
    DEVICE
)


class DiseaseModel:
    """
    Loads and manages the trained plant disease classification model.
    """

    def __init__(self):
        self.device = DEVICE

        print("=" * 60)
        print("Loading Disease Detection Model")
        print("=" * 60)

        print(f"Model architecture : {MODEL_NAME}")
        print(f"Number of classes  : {NUM_CLASSES}")
        print(f"Device             : {self.device}")
        print(f"Checkpoint         : {MODEL_PATH}")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found:\n{MODEL_PATH}"
            )

        # ----------------------------------------------------
        # Create model architecture
        # ----------------------------------------------------

        self.model = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=NUM_CLASSES
        )

        # ----------------------------------------------------
        # Load trained checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
            weights_only=False
        )

        # ----------------------------------------------------
        # Handle different checkpoint formats
        # ----------------------------------------------------

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            elif "model" in checkpoint:
                state_dict = checkpoint["model"]

            else:
                # Sometimes the checkpoint itself is the state_dict
                state_dict = checkpoint

        else:
            state_dict = checkpoint

        # ----------------------------------------------------
        # Remove possible DataParallel prefix
        # ----------------------------------------------------

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("module."):
                key = key[len("module."):]

            cleaned_state_dict[key] = value

        # ----------------------------------------------------
        # Load weights
        # ----------------------------------------------------

        missing_keys, unexpected_keys = self.model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )

        if missing_keys:
            print("\nWarning: Missing keys detected:")
            for key in missing_keys[:10]:
                print(f"  {key}")

            if len(missing_keys) > 10:
                print(
                    f"  ... and {len(missing_keys) - 10} more"
                )

        if unexpected_keys:
            print("\nWarning: Unexpected keys detected:")
            for key in unexpected_keys[:10]:
                print(f"  {key}")

            if len(unexpected_keys) > 10:
                print(
                    f"  ... and {len(unexpected_keys) - 10} more"
                )

        # ----------------------------------------------------
        # Move model to device
        # ----------------------------------------------------

        self.model = self.model.to(self.device)

        # ----------------------------------------------------
        # Evaluation mode
        # ----------------------------------------------------

        self.model.eval()

        print("\nModel loaded successfully.")
        print("=" * 60)

    def predict_logits(self, image_tensor):
        """
        Run inference and return raw model logits.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Preprocessed image tensor with shape:
            [1, 3, 224, 224]

        Returns
        -------
        torch.Tensor
            Raw logits for all 38 classes.
        """

        image_tensor = image_tensor.to(self.device)

        with torch.inference_mode():

            logits = self.model(image_tensor)

        return logits

    def predict_probabilities(self, image_tensor):
        """
        Run inference and return class probabilities.
        """

        logits = self.predict_logits(image_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        return probabilities


# ============================================================
# TEST MODEL LOADING
# ============================================================

if __name__ == "__main__":

    model = DiseaseModel()

    print("\nModel test completed successfully.")