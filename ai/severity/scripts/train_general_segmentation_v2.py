# ============================================================
# GENERAL DISEASE SEVERITY SEGMENTATION - V2
# ============================================================
#
# Dataset:
#   2382 QC-clean samples
#   Existing SAM2 pseudo-masks
#
# V2 improvements:
#   - Native 256x256 resolution
#   - EfficientNet-B0 encoder
#   - Dice + Tversky loss
#   - Disease + severity balanced sampling
#   - Stronger image/mask augmentation
#   - Mixed precision on CUDA
#   - Gradient clipping
#   - Early stopping
#   - Predicted mask saving
#   - Disease-wise evaluation
#   - Severity estimation from predicted masks
#
# IMPORTANT:
#   SAM2 masks are pseudo-labels, NOT independently verified
#   ground-truth annotations.
#
# Severity levels:
#   0-10%       LOW
#   >10-40%     MEDIUM
#   >40-60%     HIGH
#   >60%        SEVERE
#
# ============================================================

import os
import json
import random
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# WINDOWS CUDA DLL SETUP
# ============================================================

TORCH_LIB = (
    r"E:\smart_agriculture\smart_agriculture"
    r"\.venv\Lib\site-packages\torch\lib"
)

CUDA_BIN = r"E:\bin\x64"
CUDA_ROOT_BIN = r"E:\bin"

if os.path.exists(TORCH_LIB):
    os.add_dll_directory(TORCH_LIB)

if os.path.exists(CUDA_BIN):
    os.add_dll_directory(CUDA_BIN)

if os.path.exists(CUDA_ROOT_BIN):
    os.add_dll_directory(CUDA_ROOT_BIN)

# ============================================================
# IMPORTS
# ============================================================

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMG_SIZE = 256

BATCH_SIZE = 16

NUM_EPOCHS = 60

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 12

NUM_WORKERS = 0

PIN_MEMORY = True

# ------------------------------------------------------------
# Loss weights
# ------------------------------------------------------------

DICE_WEIGHT = 0.45

TVERSKY_WEIGHT = 0.55

TVERSKY_ALPHA = 0.30

TVERSKY_BETA = 0.70

TVERSKY_GAMMA = 1.33

# ------------------------------------------------------------
# Segmentation threshold
# ------------------------------------------------------------

MASK_THRESHOLD = 0.50

# ------------------------------------------------------------
# Save predicted masks
# ------------------------------------------------------------

SAVE_PREDICTED_MASKS = True

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = (
    r"E:\smart_agriculture\smart_agriculture"
)

# ------------------------------------------------------------
# Existing clean CSV
# ------------------------------------------------------------

CLEAN_CSV = os.path.join(
    ROOT,
    r"ai\severity\results"
    r"\general_severity_sam2_3000_qc",
    "severity_labels_sam2_3000_clean.csv"
)

# ------------------------------------------------------------
# Existing verified mapping CSV
# ------------------------------------------------------------

MAPPING_CSV = os.path.join(
    ROOT,
    r"ai\severity\results"
    r"\general_severity_sam2_3000_qc",
    "clean_image_mask_mapping.csv"
)

# ------------------------------------------------------------
# Existing SAM2 masks
# ------------------------------------------------------------

MASK_DIR = os.path.join(
    ROOT,
    r"ai\severity\results"
    r"\general_severity_sam2_3000",
    "masks"
)

# ------------------------------------------------------------
# V2 output
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    ROOT,
    r"ai\severity\results"
    r"\general_segmentation_model_v2"
)

PRED_MASK_DIR = os.path.join(
    OUTPUT_DIR,
    "predicted_masks"
)

OVERLAY_DIR = os.path.join(
    OUTPUT_DIR,
    "prediction_overlays"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PRED_MASK_DIR,
    exist_ok=True
)

os.makedirs(
    OVERLAY_DIR,
    exist_ok=True
)

# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


seed_everything(SEED)

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("GENERAL SEVERITY SEGMENTATION V2")
print("=" * 70)

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

# ============================================================
# SEVERITY LEVEL MAPPING
# ============================================================

LEVEL_ORDER = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "SEVERE"
]


def severity_level(value):

    value = float(value)

    if value <= 10.0:

        return "LOW"

    elif value <= 40.0:

        return "MEDIUM"

    elif value <= 60.0:

        return "HIGH"

    else:

        return "SEVERE"


# ============================================================
# LOAD CLEAN DATASET
# ============================================================

print("\nLoading dataset...")

if not os.path.exists(CLEAN_CSV):

    raise FileNotFoundError(
        f"\nClean CSV not found:\n{CLEAN_CSV}"
    )

if not os.path.exists(MASK_DIR):

    raise FileNotFoundError(
        f"\nMask directory not found:\n{MASK_DIR}"
    )

df = pd.read_csv(
    CLEAN_CSV
)

print(
    "Clean rows:",
    len(df)
)

# ============================================================
# LOAD MAPPING CSV
# ============================================================

if os.path.exists(MAPPING_CSV):

    mapping = pd.read_csv(
        MAPPING_CSV
    )

    print(
        "Mapping rows:",
        len(mapping)
    )

else:

    mapping = None

    print(
        "Mapping CSV not found."
    )

# ============================================================
# BUILD MASK LOOKUP
# ============================================================

print(
    "\nBuilding mask lookup..."
)

mask_files = [
    f
    for f in os.listdir(MASK_DIR)
    if f.lower().endswith(".png")
]

print(
    "Mask files available:",
    len(mask_files)
)

# ------------------------------------------------------------
# Exact filename lookup
# ------------------------------------------------------------

mask_lookup = {}

for filename in mask_files:

    stem = os.path.splitext(
        filename
    )[0].lower()

    mask_lookup[stem] = os.path.join(
        MASK_DIR,
        filename
    )

# ============================================================
# RESOLVE IMAGE -> MASK
# ============================================================

print(
    "\nResolving image-mask pairs..."
)


def normalize_stem(path):

    filename = os.path.basename(
        str(path)
    )

    return os.path.splitext(
        filename
    )[0].lower()


resolved_masks = []

missing_masks = []

ambiguous_masks = []

for _, row in df.iterrows():

    image_path = row["image_path"]

    # --------------------------------------------------------
    # First try image_path filename
    # --------------------------------------------------------

    image_stem = normalize_stem(
        image_path
    )

    mask_path = mask_lookup.get(
        image_stem
    )

    # --------------------------------------------------------
    # Try filename column
    # --------------------------------------------------------

    if (
        mask_path is None
        and "filename" in df.columns
    ):

        filename_stem = normalize_stem(
            row["filename"]
        )

        mask_path = mask_lookup.get(
            filename_stem
        )

    # --------------------------------------------------------
    # Try mapping CSV
    # --------------------------------------------------------

    if (
        mask_path is None
        and mapping is not None
        and "image_path" in mapping.columns
    ):

        matched = mapping[
            mapping["image_path"].astype(str)
            ==
            str(image_path)
        ]

        if len(matched) == 1:

            mapping_row = matched.iloc[0]

            # Check possible columns
            for col in [
                "mask_path",
                "mask",
                "mask_file",
                "mask_filename"
            ]:

                if col in mapping.columns:

                    value = mapping_row[col]

                    if pd.notna(value):

                        value = str(value)

                        # Absolute path
                        if os.path.exists(value):

                            mask_path = value

                        else:

                            # Filename
                            value_stem = normalize_stem(
                                value
                            )

                            mask_path = (
                                mask_lookup.get(
                                    value_stem
                                )
                            )

                        if mask_path is not None:

                            break

    # --------------------------------------------------------
    # Last-resort suffix matching
    # --------------------------------------------------------

    if mask_path is None:

        candidates = []

        for stem, path in mask_lookup.items():

            if (
                stem.endswith(image_stem)
                or image_stem.endswith(stem)
            ):

                candidates.append(path)

        if len(candidates) == 1:

            mask_path = candidates[0]

        elif len(candidates) > 1:

            ambiguous_masks.append(
                image_path
            )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    if mask_path is None:

        missing_masks.append(
            image_path
        )

    resolved_masks.append(
        mask_path
    )

df["mask_path"] = resolved_masks

print(
    "Resolved masks:",
    df["mask_path"].notna().sum()
)

print(
    "Missing masks:",
    len(missing_masks)
)

print(
    "Ambiguous masks:",
    len(ambiguous_masks)
)

# ------------------------------------------------------------
# Show problems if any
# ------------------------------------------------------------

if missing_masks:

    print(
        "\nFirst missing masks:"
    )

    for path in missing_masks[:10]:

        print(
            " ",
            path
        )

if ambiguous_masks:

    print(
        "\nFirst ambiguous masks:"
    )

    for path in ambiguous_masks[:10]:

        print(
            " ",
            path
        )

# ------------------------------------------------------------
# Stop if mapping is not complete
# ------------------------------------------------------------

if (
    len(missing_masks) > 0
    or len(ambiguous_masks) > 0
):

    raise RuntimeError(
        "\nImage-mask mapping is incomplete. "
        "Do not start training until all "
        "2382 clean samples have exactly "
        "one mask."
    )

# ============================================================
# VALIDATE IMAGE + MASK FILES
# ============================================================

print(
    "\nValidating image-mask pairs..."
)

valid_rows = []

invalid_images = []

invalid_masks = []

for _, row in df.iterrows():

    image_path = row["image_path"]

    mask_path = row["mask_path"]

    image_ok = os.path.exists(
        image_path
    )

    mask_ok = (
        isinstance(mask_path, str)
        and os.path.exists(mask_path)
    )

    if not image_ok:

        invalid_images.append(
            image_path
        )

    if not mask_ok:

        invalid_masks.append(
            mask_path
        )

    valid_rows.append(
        image_ok and mask_ok
    )

df["valid"] = valid_rows

print(
    "Valid image-mask pairs:",
    int(df["valid"].sum())
)

print(
    "Invalid images:",
    len(invalid_images)
)

print(
    "Invalid masks:",
    len(invalid_masks)
)

if not df["valid"].all():

    raise RuntimeError(
        "Some image-mask pairs are invalid."
    )

# ============================================================
# VALIDATE MASK FORMAT
# ============================================================

print(
    "\nValidating masks..."
)

mask_shapes = set()

mask_values = set()

invalid_mask_count = 0

for mask_path in df["mask_path"]:

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:

        invalid_mask_count += 1

        continue

    mask_shapes.add(
        tuple(mask.shape)
    )

    values = np.unique(
        mask
    )

    mask_values.update(
        values.tolist()
    )

    if not set(values.tolist()).issubset(
        {0, 255}
    ):

        invalid_mask_count += 1

print(
    "Mask shapes:",
    mask_shapes
)

print(
    "Mask pixel values:",
    sorted(mask_values)
)

print(
    "Invalid masks:",
    invalid_mask_count
)

if invalid_mask_count > 0:

    raise RuntimeError(
        "Invalid masks detected."
    )

# ============================================================
# APPLY USER SEVERITY LEVELS
# ============================================================

df["severity_level_user"] = (
    df["severity"]
    .apply(
        severity_level
    )
)

print(
    "\nSeverity distribution:"
)

severity_counts = (
    df["severity_level_user"]
    .value_counts()
    .reindex(
        LEVEL_ORDER,
        fill_value=0
    )
)

print(
    severity_counts
)

# ============================================================
# DATASET STATISTICS
# ============================================================

print(
    "\nSeverity statistics:"
)

print(
    df["severity"].describe()
)

# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print(
    "\nCreating disease-stratified split..."
)

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["disease"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["disease"]
)

train_df = train_df.reset_index(
    drop=True
)

val_df = val_df.reset_index(
    drop=True
)

test_df = test_df.reset_index(
    drop=True
)

print(
    "Train:",
    len(train_df)
)

print(
    "Validation:",
    len(val_df)
)

print(
    "Test:",
    len(test_df)
)

# ============================================================
# SAVE SPLITS
# ============================================================

train_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "train_split_v2.csv"
    ),
    index=False
)

val_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "validation_split_v2.csv"
    ),
    index=False
)

test_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_split_v2.csv"
    ),
    index=False
)

# ============================================================
# JOINT AUGMENTATION
# ============================================================

class JointTransform:

    def __init__(
        self,
        train=True
    ):

        self.train = train

    def __call__(
        self,
        image,
        mask
    ):

        if not self.train:

            return image, mask

        # ----------------------------------------------------
        # Horizontal flip
        # ----------------------------------------------------

        if random.random() < 0.50:

            image = cv2.flip(
                image,
                1
            )

            mask = cv2.flip(
                mask,
                1
            )

        # ----------------------------------------------------
        # Vertical flip
        # ----------------------------------------------------

        if random.random() < 0.25:

            image = cv2.flip(
                image,
                0
            )

            mask = cv2.flip(
                mask,
                0
            )

        # ----------------------------------------------------
        # Rotation
        # ----------------------------------------------------

        if random.random() < 0.50:

            angle = random.uniform(
                -15.0,
                15.0
            )

            h, w = image.shape[:2]

            center = (
                w / 2.0,
                h / 2.0
            )

            matrix = cv2.getRotationMatrix2D(
                center,
                angle,
                1.0
            )

            image = cv2.warpAffine(
                image,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )

            mask = cv2.warpAffine(
                mask,
                matrix,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        # ----------------------------------------------------
        # Brightness / contrast
        # ----------------------------------------------------

        if random.random() < 0.50:

            alpha = random.uniform(
                0.85,
                1.15
            )

            beta = random.uniform(
                -15,
                15
            )

            image = cv2.convertScaleAbs(
                image,
                alpha=alpha,
                beta=beta
            )

        # ----------------------------------------------------
        # Slight blur
        # ----------------------------------------------------

        if random.random() < 0.15:

            image = cv2.GaussianBlur(
                image,
                (3, 3),
                0
            )

        return image, mask


# ============================================================
# DATASET CLASS
# ============================================================

class SeveritySegmentationDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        train=False
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.train = train

        self.transform = JointTransform(
            train=train
        )

        self.mean = np.array(
            [
                0.485,
                0.456,
                0.406
            ],
            dtype=np.float32
        )

        self.std = np.array(
            [
                0.229,
                0.224,
                0.225
            ],
            dtype=np.float32
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(
        self,
        idx
    ):

        row = self.df.iloc[idx]

        image_path = row["image_path"]

        mask_path = row["mask_path"]

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise RuntimeError(
                f"Cannot read image:\n{image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Read mask
        # ----------------------------------------------------

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:

            raise RuntimeError(
                f"Cannot read mask:\n{mask_path}"
            )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA
        )

        mask = cv2.resize(
            mask,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        # ----------------------------------------------------
        # Joint augmentation
        # ----------------------------------------------------

        image, mask = self.transform(
            image,
            mask
        )

        # ----------------------------------------------------
        # Normalize image
        # ----------------------------------------------------

        image = (
            image.astype(
                np.float32
            )
            / 255.0
        )

        image = (
            image - self.mean
        ) / self.std

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # ----------------------------------------------------
        # Binary mask
        # ----------------------------------------------------

        mask = (
            mask > 127
        ).astype(
            np.float32
        )

        mask = np.expand_dims(
            mask,
            axis=0
        )

        # ----------------------------------------------------
        # CSV severity target
        # ----------------------------------------------------

        severity = float(
            row["severity"]
        )

        return (
            torch.tensor(
                image,
                dtype=torch.float32
            ),
            torch.tensor(
                mask,
                dtype=torch.float32
            ),
            torch.tensor(
                severity,
                dtype=torch.float32
            ),
            str(row["disease"]),
            str(row["filename"])
        )


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = SeveritySegmentationDataset(
    train_df,
    train=True
)

val_dataset = SeveritySegmentationDataset(
    val_df,
    train=False
)

test_dataset = SeveritySegmentationDataset(
    test_df,
    train=False
)

# ============================================================
# BALANCED SAMPLER
# ============================================================

print(
    "\nCreating disease + severity balanced sampler..."
)

train_sampling = train_df.copy()

train_sampling["severity_bin"] = pd.cut(
    train_sampling["severity"],
    bins=[
        -0.001,
        10,
        20,
        30,
        40,
        50,
        100
    ],
    labels=False
)

train_sampling["sampling_group"] = (
    train_sampling["disease"].astype(str)
    + "_"
    + train_sampling["severity_bin"].astype(str)
)

group_counts = (
    train_sampling[
        "sampling_group"
    ]
    .value_counts()
)

sample_weights = (
    train_sampling[
        "sampling_group"
    ]
    .map(
        lambda x:
        1.0 / group_counts[x]
    )
    .values
)

sample_weights = torch.DoubleTensor(
    sample_weights
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

# ============================================================
# MODEL
# ============================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):

        super().__init__()

        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels
        )

    def forward(
        self,
        x,
        skip
    ):

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat(
            [x, skip],
            dim=1
        )

        return self.conv(x)


class EfficientNetSegV2(
    nn.Module
):

    def __init__(self):

        super().__init__()

        backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        features = backbone.features

        # ----------------------------------------------------
        # Encoder feature channels
        #
        # 256 input
        #
        # e0 = 32
        # e1 = 24
        # e2 = 24
        # e3 = 40
        # e4 = 80
        # e5 = 112
        # e6 = 192
        # e7 = 320
        # ----------------------------------------------------

        self.enc0 = features[0]

        self.enc1 = features[1]

        self.enc2 = features[2]

        self.enc3 = features[3]

        self.enc4 = features[4]

        self.enc5 = features[5]

        self.enc6 = features[6]

        self.enc7 = features[7]

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.dec1 = DecoderBlock(
            320,
            192,
            160
        )

        self.dec2 = DecoderBlock(
            160,
            112,
            96
        )

        self.dec3 = DecoderBlock(
            96,
            80,
            64
        )

        self.dec4 = DecoderBlock(
            64,
            40,
            32
        )

        self.dec5 = DecoderBlock(
            32,
            24,
            24
        )

        self.dec6 = DecoderBlock(
            24,
            32,
            16
        )

        # ----------------------------------------------------
        # Output head
        # ----------------------------------------------------

        self.head = nn.Sequential(

            nn.Conv2d(
                16,
                8,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                8,
                1,
                kernel_size=1
            )
        )

    def forward(
        self,
        x
    ):

        e0 = self.enc0(x)

        e1 = self.enc1(e0)

        e2 = self.enc2(e1)

        e3 = self.enc3(e2)

        e4 = self.enc4(e3)

        e5 = self.enc5(e4)

        e6 = self.enc6(e5)

        e7 = self.enc7(e6)

        x = self.dec1(
            e7,
            e6
        )

        x = self.dec2(
            x,
            e5
        )

        x = self.dec3(
            x,
            e4
        )

        x = self.dec4(
            x,
            e3
        )

        x = self.dec5(
            x,
            e2
        )

        x = self.dec6(
            x,
            e0
        )

        x = F.interpolate(
            x,
            size=(
                IMG_SIZE,
                IMG_SIZE
            ),
            mode="bilinear",
            align_corners=False
        )

        return self.head(x)


# ============================================================
# INITIALIZE MODEL
# ============================================================

model = EfficientNetSegV2()

model = model.to(
    DEVICE
)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "\nModel parameters:",
    f"{parameter_count:,}"
)

# ============================================================
# ARCHITECTURE TEST
# ============================================================

print(
    "\nTesting model architecture..."
)

with torch.no_grad():

    dummy = torch.randn(
        2,
        3,
        IMG_SIZE,
        IMG_SIZE,
        device=DEVICE
    )

    dummy_output = model(
        dummy
    )

print(
    "Input:",
    tuple(dummy.shape)
)

print(
    "Output:",
    tuple(dummy_output.shape)
)

expected_shape = (
    2,
    1,
    IMG_SIZE,
    IMG_SIZE
)

if tuple(dummy_output.shape) != expected_shape:

    raise RuntimeError(
        "Model output shape is incorrect."
    )

print(
    "Architecture test: PASS"
)

del dummy
del dummy_output

if torch.cuda.is_available():

    torch.cuda.empty_cache()

# ============================================================
# LOSS FUNCTIONS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1.0
    ):

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        targets
    ):

        probs = torch.sigmoid(
            logits
        )

        probs = probs.reshape(
            probs.size(0),
            -1
        )

        targets = targets.reshape(
            targets.size(0),
            -1
        )

        intersection = (
            probs * targets
        ).sum(
            dim=1
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            probs.sum(dim=1)
            +
            targets.sum(dim=1)
            +
            self.smooth
        )

        return (
            1.0 - dice
        ).mean()


class TverskyLoss(nn.Module):

    def __init__(
        self,
        alpha=0.3,
        beta=0.7,
        gamma=1.33
    ):

        super().__init__()

        self.alpha = alpha

        self.beta = beta

        self.gamma = gamma

    def forward(
        self,
        logits,
        targets
    ):

        probs = torch.sigmoid(
            logits
        )

        probs = probs.reshape(
            probs.size(0),
            -1
        )

        targets = targets.reshape(
            targets.size(0),
            -1
        )

        true_positive = (
            probs * targets
        ).sum(
            dim=1
        )

        false_positive = (
            probs * (1.0 - targets)
        ).sum(
            dim=1
        )

        false_negative = (
            (1.0 - probs) * targets
        ).sum(
            dim=1
        )

        tversky = (
            true_positive
            + 1e-6
        ) / (
            true_positive
            +
            self.alpha * false_positive
            +
            self.beta * false_negative
            +
            1e-6
        )

        return (
            1.0 - tversky
        ).pow(
            self.gamma
        ).mean()


dice_loss_fn = DiceLoss()

tversky_loss_fn = TverskyLoss(
    alpha=TVERSKY_ALPHA,
    beta=TVERSKY_BETA,
    gamma=TVERSKY_GAMMA
)


def combined_loss(
    logits,
    targets
):

    dice_loss = dice_loss_fn(
        logits,
        targets
    )

    tversky_loss = tversky_loss_fn(
        logits,
        targets
    )

    return (
        DICE_WEIGHT * dice_loss
        +
        TVERSKY_WEIGHT * tversky_loss
    )

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=4,
    min_lr=1e-6
)

# ============================================================
# MIXED PRECISION
# ============================================================

USE_AMP = (
    DEVICE.type == "cuda"
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP
)

# ============================================================
# SEGMENTATION METRICS
# ============================================================

def segmentation_metrics(
    logits,
    targets,
    threshold=0.5
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = predictions.reshape(
        predictions.size(0),
        -1
    )

    targets = targets.reshape(
        targets.size(0),
        -1
    )

    intersection = (
        predictions * targets
    ).sum(
        dim=1
    )

    prediction_area = (
        predictions.sum(
            dim=1
        )
    )

    target_area = (
        targets.sum(
            dim=1
        )
    )

    union = (
        prediction_area
        +
        target_area
        -
        intersection
    )

    dice = (
        2.0 * intersection
        + 1e-7
    ) / (
        prediction_area
        +
        target_area
        +
        1e-7
    )

    iou = (
        intersection
        + 1e-7
    ) / (
        union
        + 1e-7
    )

    return (
        dice.mean().item(),
        iou.mean().item()
    )

# ============================================================
# TRAINING
# ============================================================

history = []

best_val_dice = -1.0

best_epoch = 0

patience_counter = 0

checkpoint_path = os.path.join(
    OUTPUT_DIR,
    "best_general_segmentation_model_v2.pth"
)

print(
    "\nStarting training..."
)

print(
    "=" * 70
)

for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_loss = 0.0

    running_dice = 0.0

    running_iou = 0.0

    sample_count = 0

    for batch in train_loader:

        (
            images,
            masks,
            severities,
            diseases,
            filenames
        ) = batch

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP
        ):

            outputs = model(
                images
            )

            loss = combined_loss(
                outputs,
                masks
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size = images.size(0)

        batch_dice, batch_iou = (
            segmentation_metrics(
                outputs.detach(),
                masks
            )
        )

        running_loss += (
            loss.item()
            *
            batch_size
        )

        running_dice += (
            batch_dice
            *
            batch_size
        )

        running_iou += (
            batch_iou
            *
            batch_size
        )

        sample_count += batch_size

    train_loss = (
        running_loss
        /
        sample_count
    )

    train_dice = (
        running_dice
        /
        sample_count
    )

    train_iou = (
        running_iou
        /
        sample_count
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_running_loss = 0.0

    val_running_dice = 0.0

    val_running_iou = 0.0

    val_count = 0

    with torch.no_grad():

        for batch in val_loader:

            (
                images,
                masks,
                severities,
                diseases,
                filenames
            ) = batch

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_AMP
            ):

                outputs = model(
                    images
                )

                loss = combined_loss(
                    outputs,
                    masks
                )

            batch_size = images.size(0)

            batch_dice, batch_iou = (
                segmentation_metrics(
                    outputs,
                    masks
                )
            )

            val_running_loss += (
                loss.item()
                *
                batch_size
            )

            val_running_dice += (
                batch_dice
                *
                batch_size
            )

            val_running_iou += (
                batch_iou
                *
                batch_size
            )

            val_count += batch_size

    val_loss = (
        val_running_loss
        /
        val_count
    )

    val_dice = (
        val_running_dice
        /
        val_count
    )

    val_iou = (
        val_running_iou
        /
        val_count
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(
        val_dice
    )

    current_lr = (
        optimizer
        .param_groups[0]
        ["lr"]
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Dice {train_dice:.4f} | "
        f"Train IoU {train_iou:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Dice {val_dice:.4f} | "
        f"Val IoU {val_iou:.4f} | "
        f"LR {current_lr:.2e}"
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "learning_rate": current_lr
        }
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    if val_dice > best_val_dice:

        best_val_dice = val_dice

        best_epoch = epoch

        patience_counter = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "val_dice":
                    val_dice,
                "val_iou":
                    val_iou,
                "img_size":
                    IMG_SIZE,
                "severity_mapping":
                    {
                        "LOW": "0-10%",
                        "MEDIUM": ">10-40%",
                        "HIGH": ">40-60%",
                        "SEVERE": ">60%"
                    }
            },
            checkpoint_path
        )

        print(
            f"  --> NEW BEST MODEL "
            f"(Val Dice={val_dice:.4f})"
        )

    else:

        patience_counter += 1

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping."
        )

        break

# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    OUTPUT_DIR,
    "training_history_v2.json"
)

with open(
    history_path,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=2
    )

print(
    "\nBest epoch:",
    best_epoch
)

print(
    "Best validation Dice:",
    f"{best_val_dice:.4f}"
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best V2 checkpoint..."
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

# ============================================================
# TEST EVALUATION
# ============================================================

test_dice_values = []

test_iou_values = []

true_severities = []

pred_severities = []

test_diseases = []

test_filenames = []

# ============================================================
# PREDICTION SAVE FUNCTION
# ============================================================

def save_prediction(
    filename,
    original_image,
    predicted_mask
):

    base_name = os.path.splitext(
        filename
    )[0]

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    mask_path = os.path.join(
        PRED_MASK_DIR,
        base_name + "_pred.png"
    )

    cv2.imwrite(
        mask_path,
        (
            predicted_mask
            *
            255
        ).astype(
            np.uint8
        )
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = (
        original_image.copy()
    )

    # Red overlay for predicted disease
    overlay[
        predicted_mask > 0
    ] = [
        255,
        0,
        0
    ]

    blended = cv2.addWeighted(
        original_image,
        0.65,
        overlay,
        0.35,
        0
    )

    overlay_path = os.path.join(
        OVERLAY_DIR,
        base_name + "_overlay.jpg"
    )

    cv2.imwrite(
        overlay_path,
        cv2.cvtColor(
            blended,
            cv2.COLOR_RGB2BGR
        )
    )

# ============================================================
# TEST LOOP
# ============================================================

print(
    "\nRunning test evaluation..."
)

with torch.no_grad():

    for batch in test_loader:

        (
            images,
            masks,
            severities,
            diseases,
            filenames
        ) = batch

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP
        ):

            outputs = model(
                images
            )

        probabilities = torch.sigmoid(
            outputs
        )

        predictions = (
            probabilities
            >= MASK_THRESHOLD
        ).float()

        batch_dice, batch_iou = (
            segmentation_metrics(
                outputs,
                masks,
                MASK_THRESHOLD
            )
        )

        test_dice_values.append(
            batch_dice
        )

        test_iou_values.append(
            batch_iou
        )

        # ----------------------------------------------------
        # Per-image evaluation
        # ----------------------------------------------------

        for i in range(
            images.size(0)
        ):

            predicted_mask = (
                predictions[
                    i,
                    0
                ]
                .cpu()
                .numpy()
                .astype(
                    np.uint8
                )
            )

            true_mask = (
                masks[
                    i,
                    0
                ]
                .cpu()
                .numpy()
                .astype(
                    np.uint8
                )
            )

            # ------------------------------------------------
            # Predicted severity
            #
            # IMPORTANT:
            # This uses disease pixels / image pixels.
            #
            # For final production we will later add
            # explicit leaf segmentation so severity is:
            #
            # disease pixels / leaf pixels
            # ------------------------------------------------

            predicted_disease_pixels = (
                predicted_mask.sum()
            )

            total_pixels = (
                IMG_SIZE
                *
                IMG_SIZE
            )

            predicted_severity = (
                predicted_disease_pixels
                /
                total_pixels
            ) * 100.0

            true_severity = float(
                severities[
                    i
                ].item()
            )

            true_severities.append(
                true_severity
            )

            pred_severities.append(
                predicted_severity
            )

            test_diseases.append(
                diseases[i]
            )

            test_filenames.append(
                filenames[i]
            )

            # ------------------------------------------------
            # Save prediction visualization
            # ------------------------------------------------

            if SAVE_PREDICTED_MASKS:

                image = (
                    images[
                        i
                    ]
                    .cpu()
                    .numpy()
                    .transpose(
                        1,
                        2,
                        0
                    )
                )

                # Undo ImageNet normalization
                image = (
                    image
                    *
                    np.array(
                        [
                            0.229,
                            0.224,
                            0.225
                        ],
                        dtype=np.float32
                    )
                    +
                    np.array(
                        [
                            0.485,
                            0.456,
                            0.406
                        ],
                        dtype=np.float32
                    )
                )

                image = np.clip(
                    image,
                    0.0,
                    1.0
                )

                image = (
                    image * 255.0
                ).astype(
                    np.uint8
                )

                save_prediction(
                    filenames[i],
                    image,
                    predicted_mask
                )

# ============================================================
# GLOBAL TEST METRICS
# ============================================================

test_dice = float(
    np.mean(
        test_dice_values
    )
)

test_iou = float(
    np.mean(
        test_iou_values
    )
)

true_severities = np.array(
    true_severities,
    dtype=np.float32
)

pred_severities = np.array(
    pred_severities,
    dtype=np.float32
)

severity_mae = mean_absolute_error(
    true_severities,
    pred_severities
)

severity_rmse = np.sqrt(
    mean_squared_error(
        true_severities,
        pred_severities
    )
)

severity_r2 = r2_score(
    true_severities,
    pred_severities
)

# ============================================================
# SEVERITY LEVEL ACCURACY
# ============================================================

true_levels = [
    severity_level(x)
    for x in true_severities
]

predicted_levels = [
    severity_level(x)
    for x in pred_severities
]

level_accuracy = (
    np.mean(
        np.array(true_levels)
        ==
        np.array(predicted_levels)
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_levels,
    predicted_levels,
    labels=LEVEL_ORDER
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"True_{x}"
        for x in LEVEL_ORDER
    ],
    columns=[
        f"Pred_{x}"
        for x in LEVEL_ORDER
    ]
)

cm_path = os.path.join(
    OUTPUT_DIR,
    "severity_confusion_matrix_v2.csv"
)

cm_df.to_csv(
    cm_path
)

# ============================================================
# TEST PREDICTIONS CSV
# ============================================================

prediction_df = pd.DataFrame(
    {
        "filename":
            test_filenames,

        "disease":
            test_diseases,

        "true_severity":
            true_severities,

        "predicted_severity":
            pred_severities,

        "true_level":
            true_levels,

        "predicted_level":
            predicted_levels
    }
)

prediction_path = os.path.join(
    OUTPUT_DIR,
    "test_predictions_v2.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)

# ============================================================
# DISEASE-WISE RESULTS
# ============================================================

disease_results = []

unique_diseases = sorted(
    set(test_diseases)
)

for disease in unique_diseases:

    indices = [
        i
        for i, current_disease
        in enumerate(test_diseases)
        if current_disease == disease
    ]

    disease_true = (
        true_severities[
            indices
        ]
    )

    disease_pred = (
        pred_severities[
            indices
        ]
    )

    disease_mae = (
        mean_absolute_error(
            disease_true,
            disease_pred
        )
    )

    disease_rmse = np.sqrt(
        mean_squared_error(
            disease_true,
            disease_pred
        )
    )

    disease_results.append(
        {
            "disease":
                disease,

            "samples":
                len(indices),

            "MAE":
                float(disease_mae),

            "RMSE":
                float(disease_rmse),

            "true_mean":
                float(disease_true.mean()),

            "predicted_mean":
                float(disease_pred.mean())
        }
    )

disease_results_df = pd.DataFrame(
    disease_results
)

disease_results_df = (
    disease_results_df
    .sort_values(
        "MAE"
    )
)

disease_results_path = os.path.join(
    OUTPUT_DIR,
    "disease_wise_results_v2.csv"
)

disease_results_df.to_csv(
    disease_results_path,
    index=False
)

# ============================================================
# SUMMARY JSON
# ============================================================

summary = {

    "model":
        "EfficientNet-B0 Segmentation V2",

    "dataset_size":
        int(len(df)),

    "train_size":
        int(len(train_df)),

    "validation_size":
        int(len(val_df)),

    "test_size":
        int(len(test_df)),

    "image_size":
        IMG_SIZE,

    "batch_size":
        BATCH_SIZE,

    "best_epoch":
        int(best_epoch),

    "best_validation_dice":
        float(best_val_dice),

    "test_dice":
        float(test_dice),

    "test_iou":
        float(test_iou),

    "severity_MAE_percent":
        float(severity_mae),

    "severity_RMSE_percent":
        float(severity_rmse),

    "severity_R2":
        float(severity_r2),

    "severity_level_accuracy":
        float(level_accuracy),

    "severity_mapping":
        {
            "LOW":
                "0-10%",

            "MEDIUM":
                ">10-40%",

            "HIGH":
                ">40-60%",

            "SEVERE":
                ">60%"
        },

    "loss":
        {
            "dice_weight":
                DICE_WEIGHT,

            "tversky_weight":
                TVERSKY_WEIGHT,

            "tversky_alpha":
                TVERSKY_ALPHA,

            "tversky_beta":
                TVERSKY_BETA,

            "tversky_gamma":
                TVERSKY_GAMMA
        },

    "warning":
        (
            "SAM2 masks are automatically "
            "generated pseudo-labels and "
            "are not independently verified "
            "ground truth."
        ),

    "production_warning":
        (
            "Current predicted severity uses "
            "disease pixels divided by image "
            "pixels. Final production severity "
            "must use disease pixels divided "
            "by total leaf pixels."
        )
}

summary_path = os.path.join(
    OUTPUT_DIR,
    "summary_v2.json"
)

with open(
    summary_path,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

# ============================================================
# FINAL RESULTS
# ============================================================

print(
    "\n"
)

print(
    "=" * 70
)

print(
    "V2 TEST RESULTS"
)

print(
    "=" * 70
)

print(
    f"Test samples       : {len(test_df)}"
)

print(
    f"Dice               : {test_dice:.4f}"
)

print(
    f"IoU                : {test_iou:.4f}"
)

print(
    f"Severity MAE       : {severity_mae:.4f}%"
)

print(
    f"Severity RMSE      : {severity_rmse:.4f}%"
)

print(
    f"Severity R2        : {severity_r2:.4f}"
)

print(
    f"Level accuracy     : "
    f"{level_accuracy * 100:.2f}%"
)

print(
    "\nSeverity confusion matrix:"
)

print(
    cm_df
)

print(
    "\nDisease-wise results:"
)

print(
    disease_results_df.to_string(
        index=False
    )
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nBest checkpoint:"
)

print(
    checkpoint_path
)

print(
    "\nTraining history:"
)

print(
    history_path
)

print(
    "\nTest predictions:"
)

print(
    prediction_path
)

print(
    "\nDone."
)

print(
    "=" * 70
)