# ============================================================
# GENERALIZED DISEASE SEVERITY SEGMENTATION
#
# Model:
#   U-Net with EfficientNet-B0 encoder
#
# Dataset:
#   2382 QC-clean SAM2 pseudo-labeled images
#
# Input:
#   RGB leaf image
#
# Output:
#   Binary disease/lesion segmentation mask
#
# Severity:
#   disease pixels / leaf pixels * 100
#
# IMPORTANT:
#   SAM2 masks are pseudo-labels, not independently verified
#   ground-truth masks.
# ============================================================

import os
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from sklearn.model_selection import train_test_split

from tqdm import tqdm


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 10

NUM_WORKERS = 0


# ============================================================
# PATHS
# ============================================================

CLEAN_CSV = (
    "ai/severity/results/"
    "general_severity_sam2_3000_qc/"
    "severity_labels_sam2_3000_clean.csv"
)

MAPPING_CSV = (
    "ai/severity/results/"
    "general_severity_sam2_3000_qc/"
    "clean_image_mask_mapping.csv"
)

MASK_DIR = (
    "ai/severity/results/"
    "general_severity_sam2_3000/"
    "masks"
)

OUTPUT_DIR = Path(
    "ai/severity/results/"
    "general_segmentation_model"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.benchmark = True


set_seed(SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("GENERALIZED DISEASE SEVERITY SEGMENTATION")
print("U-Net + EfficientNet-B0")
print("=" * 70)

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: {torch.version.cuda}"
    )


# ============================================================
# SEVERITY LEVEL
# ============================================================

def get_severity_level(severity):

    if severity <= 10:

        return "LOW"

    elif severity <= 40:

        return "MEDIUM"

    elif severity <= 60:

        return "HIGH"

    else:

        return "SEVERE"


# ============================================================
# LOAD CLEAN CSV
# ============================================================

print(
    "\nLoading clean dataset..."
)

df = pd.read_csv(
    CLEAN_CSV
)

print(
    f"Clean CSV rows: {len(df)}"
)


# ============================================================
# LOAD VERIFIED IMAGE-MASK MAPPING
# ============================================================

print(
    "\nLoading verified image-mask mapping..."
)

mapping = pd.read_csv(
    MAPPING_CSV
)

print(
    f"Mapping rows: {len(mapping)}"
)

print(
    "Mapping columns:",
    list(mapping.columns)
)


# ============================================================
# IDENTIFY MAPPING COLUMNS
# ============================================================

image_column_candidates = [

    "image_filename",

    "filename",

    "image",

    "image_path"

]

mask_column_candidates = [

    "mask_filename",

    "mask_file",

    "mask",

    "mask_path"

]


image_mapping_column = None

mask_mapping_column = None


for column in image_column_candidates:

    if column in mapping.columns:

        image_mapping_column = column

        break


for column in mask_column_candidates:

    if column in mapping.columns:

        mask_mapping_column = column

        break


if image_mapping_column is None:

    raise RuntimeError(
        "Could not find image filename column "
        "in clean_image_mask_mapping.csv"
    )


if mask_mapping_column is None:

    raise RuntimeError(
        "Could not find mask filename column "
        "in clean_image_mask_mapping.csv"
    )


print(
    f"Image mapping column: "
    f"{image_mapping_column}"
)

print(
    f"Mask mapping column: "
    f"{mask_mapping_column}"
)


# ============================================================
# BUILD IMAGE → MASK LOOKUP
# ============================================================

mask_lookup = {}

for _, row in mapping.iterrows():

    image_name = str(
        row[image_mapping_column]
    )

    mask_name = str(
        row[mask_mapping_column]
    )

    mask_lookup[image_name] = mask_name


# ============================================================
# VERIFY ALL CLEAN SAMPLES
# ============================================================

print(
    "\nVerifying image-mask pairs..."
)

valid_rows = []

missing_images = []

missing_masks = []

duplicate_mapping = []


for _, row in df.iterrows():

    filename = str(
        row["filename"]
    )

    image_path = str(
        row["image_path"]
    )


    # --------------------------------------------------------
    # Check mapping
    # --------------------------------------------------------

    if filename not in mask_lookup:

        missing_images.append(
            filename
        )

        continue


    mask_filename = mask_lookup[
        filename
    ]


    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not os.path.exists(
        image_path
    ):

        missing_images.append(
            filename
        )

        continue


    # --------------------------------------------------------
    # Check mask
    # --------------------------------------------------------

    mask_path = os.path.join(
        MASK_DIR,
        mask_filename
    )


    if not os.path.exists(
        mask_path
    ):

        missing_masks.append(
            mask_filename
        )

        continue


    row_dict = row.to_dict()

    row_dict["mask_path"] = mask_path

    valid_rows.append(
        row_dict
    )


df = pd.DataFrame(
    valid_rows
).reset_index(
    drop=True
)


# ============================================================
# VERIFICATION SUMMARY
# ============================================================

print()
print("=" * 70)
print("DATASET VERIFICATION")
print("=" * 70)

print(
    f"Original clean rows : {len(pd.read_csv(CLEAN_CSV))}"
)

print(
    f"Mapping rows        : {len(mapping)}"
)

print(
    f"Valid image-mask pairs: {len(df)}"
)

print(
    f"Missing images      : {len(missing_images)}"
)

print(
    f"Missing masks       : {len(missing_masks)}"
)


if len(df) != len(pd.read_csv(CLEAN_CSV)):

    raise RuntimeError(
        "\nDataset verification failed.\n"
        f"Expected {len(pd.read_csv(CLEAN_CSV))} "
        f"valid samples but found {len(df)}.\n"
        "Do not continue until the mapping is fixed."
    )


print(
    "✓ All clean samples have valid masks."
)


# ============================================================
# MASK VALIDATION
# ============================================================

print(
    "\nValidating mask files..."
)

invalid_masks = 0

mask_shapes = set()

mask_values = set()


for mask_path in tqdm(
    df["mask_path"],
    desc="Checking masks"
):

    try:

        mask = Image.open(
            mask_path
        ).convert("L")

        array = np.array(
            mask
        )

        mask_shapes.add(
            array.shape
        )

        mask_values.update(
            np.unique(array).tolist()
        )

        if not set(
            np.unique(array)
        ).issubset(
            {0, 255}
        ):

            invalid_masks += 1

    except Exception:

        invalid_masks += 1


print(
    f"Mask shapes: {mask_shapes}"
)

print(
    f"Mask pixel values: "
    f"{sorted(mask_values)}"
)

print(
    f"Invalid masks: "
    f"{invalid_masks}"
)


if invalid_masks > 0:

    raise RuntimeError(
        "Invalid masks detected."
    )


# ============================================================
# NEW SEVERITY LEVELS
# ============================================================

df["severity_level_new"] = (
    df["severity"]
    .apply(get_severity_level)
)


print(
    "\nSeverity distribution:"
)

print(

    df["severity_level_new"]
    .value_counts()
    .reindex(
        [
            "LOW",
            "MEDIUM",
            "HIGH",
            "SEVERE"
        ],
        fill_value=0
    )
)


# ============================================================
# SEVERITY STATISTICS
# ============================================================

print(
    "\nSeverity statistics:"
)

print(
    df["severity"].describe()
)


# ============================================================
# DISEASE-STRATIFIED SPLIT
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
    f"Train      : {len(train_df)}"
)

print(
    f"Validation : {len(val_df)}"
)

print(
    f"Test       : {len(test_df)}"
)


# ============================================================
# SAVE SPLITS
# ============================================================

train_df.to_csv(

    OUTPUT_DIR
    / "train_split.csv",

    index=False
)

val_df.to_csv(

    OUTPUT_DIR
    / "validation_split.csv",

    index=False
)

test_df.to_csv(

    OUTPUT_DIR
    / "test_split.csv",

    index=False
)


# ============================================================
# DATASET
# ============================================================

class LeafSegmentationDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        image_size=224,
        augment=False
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.image_size = image_size

        self.augment = augment

        self.mean = torch.tensor(
            [
                0.485,
                0.456,
                0.406
            ]
        ).view(
            3,
            1,
            1
        )

        self.std = torch.tensor(
            [
                0.229,
                0.224,
                0.225
            ]
        ).view(
            3,
            1,
            1
        )


    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):

        row = self.df.iloc[
            index
        ]


        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(
            row["image_path"]
        ).convert("RGB")


        # ----------------------------------------------------
        # LOAD DISEASE MASK
        # ----------------------------------------------------

        mask = Image.open(
            row["mask_path"]
        ).convert("L")


        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        image = image.resize(

            (
                self.image_size,
                self.image_size
            ),

            Image.Resampling.BILINEAR
        )


        mask = mask.resize(

            (
                self.image_size,
                self.image_size
            ),

            Image.Resampling.NEAREST
        )


        # ----------------------------------------------------
        # NUMPY
        # ----------------------------------------------------

        image = np.array(
            image
        )

        mask = np.array(
            mask
        )


        # ----------------------------------------------------
        # AUGMENTATION
        # ----------------------------------------------------

        if self.augment:

            if random.random() < 0.5:

                image = np.fliplr(
                    image
                ).copy()

                mask = np.fliplr(
                    mask
                ).copy()


            if random.random() < 0.5:

                image = np.flipud(
                    image
                ).copy()

                mask = np.flipud(
                    mask
                ).copy()


        # ----------------------------------------------------
        # IMAGE TENSOR
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        ).permute(
            2,
            0,
            1
        ).float()

        image = image / 255.0


        # ----------------------------------------------------
        # IMAGENET NORMALIZATION
        # ----------------------------------------------------

        image = (

            image
            - self.mean

        ) / self.std


        # ----------------------------------------------------
        # MASK TENSOR
        # ----------------------------------------------------

        mask = torch.from_numpy(
            mask
        ).float()

        mask = mask / 255.0

        mask = mask.unsqueeze(
            0
        )


        return image, mask


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = LeafSegmentationDataset(

    train_df,

    IMAGE_SIZE,

    augment=True
)


val_dataset = LeafSegmentationDataset(

    val_df,

    IMAGE_SIZE,

    augment=False
)


test_dataset = LeafSegmentationDataset(

    test_df,

    IMAGE_SIZE,

    augment=False
)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


# ============================================================
# DOUBLE CONV
# ============================================================

class DoubleConv(
    nn.Module
):

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


# ============================================================
# EFFICIENTNET-B0 U-NET
#
# Actual torchvision EfficientNet-B0:
#
# features[0] = 32
# features[1] = 16
# features[2] = 24
# features[3] = 40
# features[4] = 80
# features[5] = 112
# features[6] = 192
# features[7] = 320
#
# Spatial:
#
# e1 = 112x112, 32
# e2 =  56x56,  24
# e3 =  28x28,  40
# e4 =  14x14,  80
# bottleneck = 7x7, 320
#
# Decoder:
#
# 320 -> 80  -> 14x14
# 80  -> 40  -> 28x28
# 40  -> 24  -> 56x56
# 24  -> 32  -> 112x112
# 32  -> 16  -> 224x224
# ============================================================

class EfficientNetUNet(
    nn.Module
):

    def __init__(self):

        super().__init__()


        backbone = efficientnet_b0(

            weights=
            EfficientNet_B0_Weights.DEFAULT
        )


        features = backbone.features


        # ====================================================
        # ENCODER
        # ====================================================

        self.enc1 = features[0]

        # 32 channels
        # 224 -> 112


        self.enc2 = nn.Sequential(

            features[1],

            features[2]
        )

        # 24 channels
        # 112 -> 56


        self.enc3 = features[3]

        # 40 channels
        # 56 -> 28


        self.enc4 = features[4]

        # 80 channels
        # 28 -> 14


        self.bottleneck = nn.Sequential(

            features[5],

            features[6],

            features[7]
        )

        # 320 channels
        # 14 -> 7


        # ====================================================
        # DECODER 1
        #
        # bottleneck:
        # 320 @ 7x7
        #
        # skip:
        # 80 @ 14x14
        # ====================================================

        self.up1 = nn.ConvTranspose2d(

            320,

            80,

            kernel_size=2,

            stride=2
        )


        self.dec1 = DoubleConv(

            80 + 80,

            80
        )


        # ====================================================
        # DECODER 2
        #
        # 80 @ 14x14
        # skip:
        # 40 @ 28x28
        # ====================================================

        self.up2 = nn.ConvTranspose2d(

            80,

            40,

            kernel_size=2,

            stride=2
        )


        self.dec2 = DoubleConv(

            40 + 40,

            40
        )


        # ====================================================
        # DECODER 3
        #
        # 40 @ 28x28
        # skip:
        # 24 @ 56x56
        # ====================================================

        self.up3 = nn.ConvTranspose2d(

            40,

            24,

            kernel_size=2,

            stride=2
        )


        self.dec3 = DoubleConv(

            24 + 24,

            24
        )


        # ====================================================
        # DECODER 4
        #
        # 24 @ 56x56
        # skip:
        # 32 @ 112x112
        # ====================================================

        self.up4 = nn.ConvTranspose2d(

            24,

            32,

            kernel_size=2,

            stride=2
        )


        self.dec4 = DoubleConv(

            32 + 32,

            32
        )


        # ====================================================
        # FINAL UPSAMPLE
        #
        # 32 @ 112x112
        # -> 16 @ 224x224
        # ====================================================

        self.final_up = nn.ConvTranspose2d(

            32,

            16,

            kernel_size=2,

            stride=2
        )


        self.final_conv = nn.Conv2d(

            16,

            1,

            kernel_size=1
        )


    def forward(self, x):


        # ====================================================
        # ENCODER
        # ====================================================

        e1 = self.enc1(x)

        # e1:
        # 32 x 112 x 112


        e2 = self.enc2(e1)

        # e2:
        # 24 x 56 x 56


        e3 = self.enc3(e2)

        # e3:
        # 40 x 28 x 28


        e4 = self.enc4(e3)

        # e4:
        # 80 x 14 x 14


        b = self.bottleneck(e4)

        # b:
        # 320 x 7 x 7


        # ====================================================
        # DECODER 1
        # ====================================================

        x = self.up1(b)

        # 80 x 14 x 14


        x = torch.cat(
            [x, e4],
            dim=1
        )

        # 160 x 14 x 14


        x = self.dec1(x)

        # 80 x 14 x 14


        # ====================================================
        # DECODER 2
        # ====================================================

        x = self.up2(x)

        # 40 x 28 x 28


        x = torch.cat(
            [x, e3],
            dim=1
        )

        # 80 x 28 x 28


        x = self.dec2(x)

        # 40 x 28 x 28


        # ====================================================
        # DECODER 3
        # ====================================================

        x = self.up3(x)

        # 24 x 56 x 56


        x = torch.cat(
            [x, e2],
            dim=1
        )

        # 48 x 56 x 56


        x = self.dec3(x)

        # 24 x 56 x 56


        # ====================================================
        # DECODER 4
        # ====================================================

        x = self.up4(x)

        # 32 x 112 x 112


        x = torch.cat(
            [x, e1],
            dim=1
        )

        # 64 x 112 x 112


        x = self.dec4(x)

        # 32 x 112 x 112


        # ====================================================
        # FINAL
        # ====================================================

        x = self.final_up(x)

        # 16 x 224 x 224


        x = self.final_conv(x)

        # 1 x 224 x 224


        return x


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(
    nn.Module
):

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

        probabilities = torch.sigmoid(
            logits
        )


        probabilities = probabilities.reshape(

            probabilities.size(0),

            -1
        )


        targets = targets.reshape(

            targets.size(0),

            -1
        )


        intersection = (

            probabilities
            * targets

        ).sum(
            dim=1
        )


        dice = (

            2.0 * intersection
            + self.smooth

        ) / (

            probabilities.sum(
                dim=1
            )

            +

            targets.sum(
                dim=1
            )

            +

            self.smooth
        )


        return (
            1.0 - dice.mean()
        )


# ============================================================
# BCE + DICE LOSS
# ============================================================

class BCEDiceLoss(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.bce = (
            nn.BCEWithLogitsLoss()
        )

        self.dice = DiceLoss()


    def forward(
        self,
        logits,
        targets
    ):

        return (

            self.bce(
                logits,
                targets
            )

            +

            self.dice(
                logits,
                targets
            )
        )


# ============================================================
# DICE METRIC
# ============================================================

def calculate_dice(
    logits,
    targets,
    threshold=0.5
):

    probabilities = torch.sigmoid(
        logits
    )


    predictions = (

        probabilities
        >= threshold

    ).float()


    intersection = (

        predictions
        * targets

    ).sum(
        dim=(1, 2, 3)
    )


    denominator = (

        predictions.sum(
            dim=(1, 2, 3)
        )

        +

        targets.sum(
            dim=(1, 2, 3)
        )
    )


    dice = (

        2.0 * intersection
        + 1e-7

    ) / (

        denominator
        + 1e-7
    )


    return dice.mean().item()


# ============================================================
# IOU METRIC
# ============================================================

def calculate_iou(
    logits,
    targets,
    threshold=0.5
):

    probabilities = torch.sigmoid(
        logits
    )


    predictions = (

        probabilities
        >= threshold

    ).float()


    intersection = (

        predictions
        * targets

    ).sum(
        dim=(1, 2, 3)
    )


    union = (

        predictions
        + targets
        - predictions * targets

    ).sum(
        dim=(1, 2, 3)
    )


    iou = (

        intersection
        + 1e-7

    ) / (

        union
        + 1e-7
    )


    return iou.mean().item()


# ============================================================
# CREATE MODEL
# ============================================================

print(
    "\nCreating model..."
)

model = EfficientNetUNet()

model = model.to(
    DEVICE
)


parameter_count = sum(

    p.numel()

    for p in model.parameters()
)


print(
    f"Parameters: "
    f"{parameter_count:,}"
)


# ============================================================
# ARCHITECTURE TEST
# ============================================================

print(
    "\nTesting model architecture..."
)


with torch.no_grad():

    test_input = torch.randn(

        2,

        3,

        IMAGE_SIZE,

        IMAGE_SIZE

    ).to(
        DEVICE
    )


    test_output = model(
        test_input
    )


print(
    "Input shape :",
    tuple(
        test_input.shape
    )
)


print(
    "Output shape:",
    tuple(
        test_output.shape
    )
)


expected_shape = (

    2,

    1,

    IMAGE_SIZE,

    IMAGE_SIZE
)


if tuple(
    test_output.shape
) != expected_shape:

    raise RuntimeError(

        "Architecture test failed.\n"

        f"Expected: {expected_shape}\n"

        f"Got: "
        f"{tuple(test_output.shape)}"
    )


print(
    "✓ Model architecture test PASSED"
)


del test_input

del test_output


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# LOSS
# ============================================================

criterion = BCEDiceLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = (

    torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=3
    )
)


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {

    "train_loss": [],

    "val_loss": [],

    "val_dice": [],

    "val_iou": [],

    "learning_rate": []

}


# ============================================================
# BEST MODEL
# ============================================================

best_val_dice = -1.0

epochs_without_improvement = 0


best_model_path = (

    OUTPUT_DIR
    / "best_general_segmentation_model.pth"
)


# ============================================================
# TRAINING
# ============================================================

print(
    "\nStarting training...\n"
)


for epoch in range(

    1,

    EPOCHS + 1
):


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()


    train_loss_total = 0.0


    train_bar = tqdm(

        train_loader,

        desc=(
            f"Epoch "
            f"{epoch}/{EPOCHS} [TRAIN]"
        )
    )


    for images, masks in train_bar:


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


        logits = model(
            images
        )


        loss = criterion(

            logits,

            masks
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0
        )


        optimizer.step()


        train_loss_total += (
            loss.item()
        )


        train_bar.set_postfix(

            loss=(
                f"{loss.item():.4f}"
            )
        )


    train_loss = (

        train_loss_total
        / len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    val_loss_total = 0.0

    val_dice_total = 0.0

    val_iou_total = 0.0


    with torch.no_grad():


        val_bar = tqdm(

            val_loader,

            desc=(
                f"Epoch "
                f"{epoch}/{EPOCHS} [VAL]"
            )
        )


        for images, masks in val_bar:


            images = images.to(

                DEVICE,

                non_blocking=True
            )


            masks = masks.to(

                DEVICE,

                non_blocking=True
            )


            logits = model(
                images
            )


            loss = criterion(

                logits,

                masks
            )


            val_loss_total += (
                loss.item()
            )


            val_dice_total += (
                calculate_dice(
                    logits,
                    masks
                )
            )


            val_iou_total += (
                calculate_iou(
                    logits,
                    masks
                )
            )


    val_loss = (

        val_loss_total
        / len(val_loader)
    )


    val_dice = (

        val_dice_total
        / len(val_loader)
    )


    val_iou = (

        val_iou_total
        / len(val_loader)
    )


    current_lr = (

        optimizer
        .param_groups[0]["lr"]
    )


    # ========================================================
    # HISTORY
    # ========================================================

    history[
        "train_loss"
    ].append(
        train_loss
    )


    history[
        "val_loss"
    ].append(
        val_loss
    )


    history[
        "val_dice"
    ].append(
        val_dice
    )


    history[
        "val_iou"
    ].append(
        val_iou
    )


    history[
        "learning_rate"
    ].append(
        current_lr
    )


    # ========================================================
    # LR UPDATE
    # ========================================================

    scheduler.step(
        val_dice
    )


    # ========================================================
    # PRINT
    # ========================================================

    print()

    print(

        f"Epoch {epoch:03d} | "

        f"Train Loss: "
        f"{train_loss:.4f} | "

        f"Val Loss: "
        f"{val_loss:.4f} | "

        f"Dice: "
        f"{val_dice:.4f} | "

        f"IoU: "
        f"{val_iou:.4f} | "

        f"LR: "
        f"{current_lr:.2e}"
    )


    # ========================================================
    # SAVE BEST
    # ========================================================

    if val_dice > best_val_dice:


        best_val_dice = val_dice

        epochs_without_improvement = 0


        torch.save(

            {

                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch,

                "val_dice":
                    val_dice,

                "val_iou":
                    val_iou,

                "image_size":
                    IMAGE_SIZE,

                "model":
                    "EfficientNet-B0 U-Net"

            },

            best_model_path
        )


        print(

            f"✓ Best model saved "
            f"(Dice={val_dice:.4f})"
        )


    else:

        epochs_without_improvement += 1


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (

        epochs_without_improvement
        >= PATIENCE

    ):

        print()

        print(

            f"Early stopping at "
            f"epoch {epoch}"
        )

        break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(

    OUTPUT_DIR
    / "training_history.json",

    "w"

) as file:

    json.dump(

        history,

        file,

        indent=2
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best model..."
)


checkpoint = torch.load(

    best_model_path,

    map_location=DEVICE,

    weights_only=False
)


model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ]
)


model.eval()


print(

    f"Best epoch: "
    f"{checkpoint['epoch']}"
)


print(

    f"Best validation Dice: "
    f"{checkpoint['val_dice']:.4f}"
)


print(

    f"Best validation IoU: "
    f"{checkpoint['val_iou']:.4f}"
)


# ============================================================
# TEST
# ============================================================

print(
    "\nRunning final test evaluation..."
)


test_rows = test_df.reset_index(
    drop=True
)


test_predictions = []

test_dice_values = []

test_iou_values = []


sample_index = 0


with torch.no_grad():


    test_bar = tqdm(

        test_loader,

        desc="TEST"
    )


    for images, masks in test_bar:


        batch_size = images.size(
            0
        )


        images = images.to(

            DEVICE,

            non_blocking=True
        )


        masks = masks.to(

            DEVICE,

            non_blocking=True
        )


        logits = model(
            images
        )


        probabilities = torch.sigmoid(
            logits
        )


        predicted_masks = (

            probabilities >= 0.5

        ).float()


        # ----------------------------------------------------
        # SEGMENTATION METRICS
        # ----------------------------------------------------

        test_dice_values.append(

            calculate_dice(

                logits,

                masks
            )
        )


        test_iou_values.append(

            calculate_iou(

                logits,

                masks
            )
        )


        # ----------------------------------------------------
        # SAMPLE-LEVEL SEVERITY
        # ----------------------------------------------------

        for i in range(
            batch_size
        ):


            row = test_rows.iloc[
                sample_index
            ]


            predicted_mask = (
                predicted_masks[
                    i,
                    0
                ]
            )


            predicted_pixels_224 = int(

                predicted_mask.sum()
                .item()
            )


            # ------------------------------------------------
            # Original masks are 256x256.
            #
            # Model output is 224x224.
            #
            # Convert predicted pixel area approximately
            # back to the 256x256 coordinate system.
            # ------------------------------------------------

            scale_factor = (

                256.0 * 256.0

            ) / (

                IMAGE_SIZE
                * IMAGE_SIZE
            )


            predicted_disease_pixels = (

                predicted_pixels_224
                * scale_factor
            )


            # ------------------------------------------------
            # Leaf pixels from QC CSV
            # ------------------------------------------------

            leaf_pixels = float(

                row[
                    "leaf_pixels"
                ]
            )


            # ------------------------------------------------
            # Severity
            # ------------------------------------------------

            predicted_severity = (

                predicted_disease_pixels
                /
                max(
                    leaf_pixels,
                    1.0
                )

            ) * 100.0


            predicted_severity = float(

                np.clip(

                    predicted_severity,

                    0.0,

                    100.0
                )
            )


            true_severity = float(

                row[
                    "severity"
                ]
            )


            true_level = (
                get_severity_level(
                    true_severity
                )
            )


            predicted_level = (
                get_severity_level(
                    predicted_severity
                )
            )


            absolute_error = abs(

                predicted_severity
                - true_severity
            )


            test_predictions.append(

                {

                    "filename":
                        row["filename"],

                    "plant":
                        row["plant"],

                    "disease":
                        row["disease"],

                    "true_severity":
                        true_severity,

                    "predicted_severity":
                        predicted_severity,

                    "absolute_error":
                        absolute_error,

                    "true_level":
                        true_level,

                    "predicted_level":
                        predicted_level,

                    "leaf_pixels":
                        leaf_pixels,

                    "predicted_disease_pixels":
                        predicted_disease_pixels

                }
            )


            sample_index += 1


# ============================================================
# TEST DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    test_predictions
)


# ============================================================
# SEVERITY METRICS
# ============================================================

true_severity = (

    results_df[
        "true_severity"
    ].values
)


predicted_severity = (

    results_df[
        "predicted_severity"
    ].values
)


errors = (

    predicted_severity
    - true_severity
)


mae = np.mean(
    np.abs(errors)
)


rmse = np.sqrt(

    np.mean(
        errors ** 2
    )
)


ss_res = np.sum(
    errors ** 2
)


ss_tot = np.sum(

    (
        true_severity
        - np.mean(
            true_severity
        )
    ) ** 2
)


if ss_tot > 0:

    r2 = (

        1.0
        - ss_res / ss_tot
    )

else:

    r2 = 0.0


# ============================================================
# SEVERITY LEVEL ACCURACY
# ============================================================

level_accuracy = np.mean(

    results_df[
        "true_level"
    ].values

    ==

    results_df[
        "predicted_level"
    ].values
)


# ============================================================
# SEGMENTATION METRICS
# ============================================================

test_dice = np.mean(
    test_dice_values
)


test_iou = np.mean(
    test_iou_values
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()

print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)


print(
    f"Test samples     : "
    f"{len(results_df)}"
)


print(
    f"Dice coefficient : "
    f"{test_dice:.4f}"
)


print(
    f"IoU              : "
    f"{test_iou:.4f}"
)


print(
    f"Severity MAE     : "
    f"{mae:.4f}%"
)


print(
    f"Severity RMSE    : "
    f"{rmse:.4f}%"
)


print(
    f"Severity R²      : "
    f"{r2:.4f}"
)


print(
    f"Level accuracy   : "
    f"{level_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

results_df.to_csv(

    OUTPUT_DIR
    / "test_predictions.csv",

    index=False
)


# ============================================================
# DISEASE-WISE RESULTS
# ============================================================

print(
    "\nDisease-wise results:"
)


disease_results = []


for disease, group in (

    results_df.groupby(
        "disease"
    )

):


    disease_mae = np.mean(

        np.abs(

            group[
                "predicted_severity"
            ].values

            -

            group[
                "true_severity"
            ].values

        )
    )


    disease_results.append(

        {

            "disease":
                disease,

            "samples":
                len(group),

            "MAE":
                disease_mae,

            "true_mean_severity":
                group[
                    "true_severity"
                ].mean(),

            "predicted_mean_severity":
                group[
                    "predicted_severity"
                ].mean()

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


print(

    disease_results_df.to_string(
        index=False
    )
)


disease_results_df.to_csv(

    OUTPUT_DIR
    / "disease_wise_results.csv",

    index=False
)


# ============================================================
# SEVERITY CONFUSION MATRIX
# ============================================================

level_order = [

    "LOW",

    "MEDIUM",

    "HIGH",

    "SEVERE"
]


confusion_matrix = pd.crosstab(

    pd.Categorical(

        results_df[
            "true_level"
        ],

        categories=level_order,

        ordered=True
    ),

    pd.Categorical(

        results_df[
            "predicted_level"
        ],

        categories=level_order,

        ordered=True
    ),

    rownames=[
        "True"
    ],

    colnames=[
        "Predicted"
    ],

    dropna=False
)


print(
    "\nSeverity-level confusion matrix:"
)

print(
    confusion_matrix
)


confusion_matrix.to_csv(

    OUTPUT_DIR
    / "severity_level_confusion_matrix.csv"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "model":
        "U-Net with EfficientNet-B0 encoder",

    "total_clean_samples":
        int(len(df)),

    "train_samples":
        int(len(train_df)),

    "validation_samples":
        int(len(val_df)),

    "test_samples":
        int(len(test_df)),

    "best_epoch":
        int(
            checkpoint["epoch"]
        ),

    "best_validation_dice":
        float(
            checkpoint["val_dice"]
        ),

    "best_validation_iou":
        float(
            checkpoint["val_iou"]
        ),

    "test_dice":
        float(
            test_dice
        ),

    "test_iou":
        float(
            test_iou
        ),

    "severity_mae_percent":
        float(
            mae
        ),

    "severity_rmse_percent":
        float(
            rmse
        ),

    "severity_r2":
        float(
            r2
        ),

    "severity_level_accuracy":
        float(
            level_accuracy
        ),

    "severity_definition":
        "disease pixels / leaf pixels * 100",

    "severity_levels":
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

    "image_size":
        IMAGE_SIZE,

    "source_mask_size":
        "256x256",

    "pseudo_label_source":
        "SAM2",

    "warning":
        "SAM2-generated pseudo-labels are not "
        "independently verified ground truth."

}


with open(

    OUTPUT_DIR
    / "test_summary.json",

    "w"

) as file:

    json.dump(

        summary,

        file,

        indent=2
    )


# ============================================================
# FINAL FILE LIST
# ============================================================

print()

print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)


print(
    "\nGenerated files:"
)


for file in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        f"  {file.name}"
    )


print(
    "\nDone."
)