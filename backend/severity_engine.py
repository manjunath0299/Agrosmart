"""
AgroSmart - Severity Inference Engine

Purpose:
    Predict disease severity from a leaf image using the
    human-GT trained EfficientNet-B0 U-Net model.

Severity:
    disease_pixels / leaf_pixels * 100

Severity bands:
    0-10%    -> LOW
    >10-40%  -> MEDIUM
    >40-60%  -> HIGH
    >60%     -> SEVERE
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "human_gt_unet"
    / "best_human_gt_severity_unet.pth"
)

IMG_SIZE = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL
# Exact architecture used during training
# ============================================================

class ConvBlock(nn.Module):
    """
    Two convolution + BatchNorm + ReLU blocks.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    U-Net decoder block.

    Upsamples the decoder feature map to the spatial
    dimensions of the skip connection and then applies
    the ConvBlock.
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.conv = ConvBlock(
            in_ch + skip_ch,
            out_ch,
        )

    def forward(self, x, skip):

        x = torch.nn.functional.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        return self.conv(x)


class EfficientNetUNet(nn.Module):
    """
    EfficientNet-B0 encoder + U-Net decoder.

    This must remain identical to the training architecture
    because the checkpoint contains weights for this exact
    structure.
    """

    def __init__(self):
        super().__init__()

        # Same EfficientNet-B0 used during training.
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        self.features = backbone.features

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.d1 = DecoderBlock(
            320,
            192,
            192,
        )

        self.d2 = DecoderBlock(
            192,
            112,
            112,
        )

        self.d3 = DecoderBlock(
            112,
            80,
            80,
        )

        self.d4 = DecoderBlock(
            80,
            40,
            48,
        )

        self.d5 = DecoderBlock(
            48,
            24,
            32,
        )

        # ----------------------------------------------------
        # Final segmentation head
        # ----------------------------------------------------

        self.final = nn.Sequential(
            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                16,
                1,
                kernel_size=1,
            ),
        )

    def forward(self, x):

        skips = []

        # EfficientNet-B0 stages.
        for i, layer in enumerate(self.features):

            x = layer(x)

            # Exact feature stages used during training.
            if i in [2, 3, 4, 5, 6, 7]:
                skips.append(x)

        (
            s24,
            s40,
            s80,
            s112,
            s192,
            s320,
        ) = skips

        # Decoder.
        x = self.d1(s320, s192)
        x = self.d2(x, s112)
        x = self.d3(x, s80)
        x = self.d4(x, s40)
        x = self.d5(x, s24)

        # Final upsampling used during training.
        x = torch.nn.functional.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        return self.final(x)


# ============================================================
# SEVERITY ENGINE
# ============================================================

class SeverityEngine:
    """
    Handles:

        image
          ↓
        leaf mask estimation
          ↓
        disease segmentation
          ↓
        disease ∩ leaf
          ↓
        severity percentage
          ↓
        severity level
    """

    def __init__(self):

        print("=" * 60)
        print("Loading AgroSmart Severity Engine")
        print("=" * 60)

        print("Device:", DEVICE)
        print("Checkpoint:", CHECKPOINT_PATH)

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"\nSeverity checkpoint not found:\n"
                f"{CHECKPOINT_PATH}"
            )

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        self.model = EfficientNetUNet().to(DEVICE)

        # ----------------------------------------------------
        # Load trained checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False,
        )

        if "model_state_dict" not in checkpoint:
            raise RuntimeError(
                "Checkpoint does not contain "
                "'model_state_dict'."
            )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        # ----------------------------------------------------
        # Image preprocessing
        # Must match training preprocessing.
        # ----------------------------------------------------

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (IMG_SIZE, IMG_SIZE)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ]
        )

        print("Severity model loaded successfully.")
        print()

    # ========================================================
    # LEAF MASK ESTIMATION
    # ========================================================

    def estimate_leaf_mask(self, image_rgb):
        """
        Estimate the visible leaf region.

        NOTE:
            The trained severity model predicts disease
            segmentation. It does NOT contain a separate
            trained leaf-segmentation network.

            Therefore, at inference time we estimate the
            leaf region using image processing.
        """

        hsv = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2HSV,
        )

        # ----------------------------------------------------
        # Green vegetation mask
        # ----------------------------------------------------

        lower_green = np.array(
            [20, 25, 20],
            dtype=np.uint8,
        )

        upper_green = np.array(
            [100, 255, 255],
            dtype=np.uint8,
        )

        green_mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green,
        )

        # ----------------------------------------------------
        # Broad vegetation/leaf mask
        #
        # This helps retain yellow/brown diseased regions
        # that may no longer appear strongly green.
        # ----------------------------------------------------

        _, saturation, value = cv2.split(hsv)

        broad_leaf = (
            (saturation > 15)
            & (value > 25)
            & (value < 245)
        ).astype(np.uint8) * 255

        mask = cv2.bitwise_or(
            green_mask,
            broad_leaf,
        )

        # ----------------------------------------------------
        # Morphological cleanup
        # ----------------------------------------------------

        kernel = np.ones(
            (7, 7),
            dtype=np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        # ----------------------------------------------------
        # Keep largest connected component
        # ----------------------------------------------------

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        if num_labels > 1:

            largest_label = (
                1
                + np.argmax(
                    stats[
                        1:,
                        cv2.CC_STAT_AREA,
                    ]
                )
            )

            mask = np.where(
                labels == largest_label,
                255,
                0,
            ).astype(np.uint8)

        return mask

    # ========================================================
    # SEVERITY LEVEL
    # ========================================================

    @staticmethod
    def get_severity_level(severity):
        """
        Convert severity percentage into project severity band.
        """

        if severity <= 10:
            return "LOW"

        elif severity <= 40:
            return "MEDIUM"

        elif severity <= 60:
            return "HIGH"

        else:
            return "SEVERE"

    # ========================================================
    # IMAGE CONVERSION
    # ========================================================

    @staticmethod
    def prepare_image(image):
        """
        Convert supported input formats into RGB NumPy array.
        """

        # PIL image
        if isinstance(image, Image.Image):

            return np.array(
                image.convert("RGB")
            )

        # File path
        if isinstance(
            image,
            (str, Path),
        ):

            image = cv2.imread(
                str(image)
            )

            if image is None:
                raise ValueError(
                    f"Could not read image:\n{image}"
                )

            return cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

        # NumPy image
        if isinstance(
            image,
            np.ndarray,
        ):

            if image.ndim != 3:
                raise ValueError(
                    "Image must have shape "
                    "(H, W, 3)."
                )

            if image.shape[2] != 3:
                raise ValueError(
                    "Image must have 3 color channels."
                )

            return image

        raise TypeError(
            "Supported image types are "
            "PIL.Image, NumPy array, or file path."
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(self, image):
        """
        Predict disease severity.

        Returns:
            dict
        """

        # ----------------------------------------------------
        # Prepare image
        # ----------------------------------------------------

        image_rgb = self.prepare_image(
            image
        )

        original_height, original_width = (
            image_rgb.shape[:2]
        )

        # ----------------------------------------------------
        # Estimate leaf mask
        # ----------------------------------------------------

        leaf_mask = self.estimate_leaf_mask(
            image_rgb
        )

        MODEL_OUTPUT_SIZE = 128

        leaf_mask_small = cv2.resize(
            leaf_mask,
            (MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

        leaf_binary = (
            leaf_mask_small > 127
        ).astype(np.float32)

        # ----------------------------------------------------
        # Prepare model input
        # ----------------------------------------------------

        pil_image = Image.fromarray(
            image_rgb
        )

        tensor = self.transform(
            pil_image
        )

        tensor = tensor.unsqueeze(0).to(
            DEVICE
        )

        # ----------------------------------------------------
        # Disease segmentation
        # ----------------------------------------------------

        with torch.no_grad():

            logits = self.model(
                tensor
            )

            probabilities = torch.sigmoid(
                logits
            )

            disease_binary = (
                probabilities >= 0.5
            ).float()

        disease_binary = (
            disease_binary[0, 0]
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Disease must be inside the leaf.
        # Same principle used during training/evaluation.
        # ----------------------------------------------------

        disease_binary = (
            disease_binary
            * leaf_binary
        )

        # ----------------------------------------------------
        # Calculate pixels
        # ----------------------------------------------------

        disease_pixels = float(
            disease_binary.sum()
        )

        leaf_pixels = float(
            leaf_binary.sum()
        )

        # ----------------------------------------------------
        # No leaf detected
        # ----------------------------------------------------

        if leaf_pixels <= 0:

            return {
                "severity_available": False,
                "severity": None,
                "severity_level": None,
                "leaf_pixels": 0,
                "disease_pixels": 0,
                "reason": (
                    "Leaf could not be detected."
                ),
            }

        # ----------------------------------------------------
        # Severity percentage
        # ----------------------------------------------------

        severity = (
            100.0
            * disease_pixels
            / leaf_pixels
        )

        severity = float(
            np.clip(
                severity,
                0.0,
                100.0,
            )
        )

        severity_level = (
            self.get_severity_level(
                severity
            )
        )

        # ----------------------------------------------------
        # Resize masks back to original image size
        # ----------------------------------------------------

        disease_mask = cv2.resize(
            (
                disease_binary * 255
            ).astype(np.uint8),
            (
                original_width,
                original_height,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        return {
            "severity_available": True,
            "severity": round(
                severity,
                2,
            ),
            "severity_level": severity_level,
            "leaf_pixels": int(
                leaf_pixels
            ),
            "disease_pixels": int(
                disease_pixels
            ),
            "leaf_mask": leaf_mask,
            "disease_mask": disease_mask,
        }


# ============================================================
# SINGLETON
# ============================================================

severity_engine = SeverityEngine()