
import os
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"E:\smart_agriculture\smart_agriculture")
DATASET = ROOT / "ai" / "severity" / "dataset" / "human_gt_120"
RESULTS = ROOT / "ai" / "severity" / "results" / "human_gt_unet"
RESULTS.mkdir(parents=True, exist_ok=True)

IMAGE_DIR = DATASET / "images"
MASK_DIR = DATASET / "masks"
CSV_PATH = DATASET / "human_gt_results.csv"

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 80
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ============================================================
# DATASET
# ============================================================

def severity_level(x):
    if x <= 10:
        return "LOW"
    if x <= 40:
        return "MEDIUM"
    if x <= 60:
        return "HIGH"
    return "SEVERE"


class HumanSeverityDataset(Dataset):
    def __init__(self, rows, train=False):
        self.rows = rows.reset_index(drop=True)
        self.train = train

        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]

        image_path = IMAGE_DIR / str(row["image_file"])
        stem = Path(str(row["image_file"])).stem

        leaf_path = MASK_DIR / f"{stem}_leaf.png"
        disease_path = MASK_DIR / f"{stem}_disease.png"

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        leaf = cv2.imread(str(leaf_path), cv2.IMREAD_GRAYSCALE)
        disease = cv2.imread(str(disease_path), cv2.IMREAD_GRAYSCALE)

        if leaf is None or disease is None:
            raise RuntimeError(
                f"Missing masks for {row['image_file']}\n"
                f"Leaf: {leaf_path}\nDisease: {disease_path}"
            )

        leaf = (leaf > 127).astype(np.float32)
        disease = (disease > 127).astype(np.float32)

        # Keep disease strictly inside leaf.
        disease = disease * leaf

        # Joint geometric augmentation.
        if self.train:
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                leaf = np.fliplr(leaf).copy()
                disease = np.fliplr(disease).copy()

            if random.random() < 0.5:
                image = np.flipud(image).copy()
                leaf = np.flipud(leaf).copy()
                disease = np.flipud(disease).copy()

            if random.random() < 0.5:
                angle = random.uniform(-12, 12)
                h, w = leaf.shape
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

                image = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                leaf = cv2.warpAffine(
                    leaf, M, (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                disease = cv2.warpAffine(
                    disease, M, (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )

            # Mild photometric augmentation only on image.
            if random.random() < 0.35:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-12, 12)
                image = np.clip(
                    image.astype(np.float32) * alpha + beta,
                    0, 255
                ).astype(np.uint8)

        image = self.image_transform(image)

        leaf = cv2.resize(
            leaf, (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_NEAREST
        )
        disease = cv2.resize(
            disease, (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        leaf = torch.from_numpy(leaf).unsqueeze(0).float()
        disease = torch.from_numpy(disease).unsqueeze(0).float()

        return image, leaf, disease, str(row["image_file"])


# ============================================================
# MODEL: EfficientNet-B0 U-NET
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = torch.nn.functional.interpolate(
            x, size=skip.shape[-2:],
            mode="bilinear", align_corners=False
        )
        return self.conv(torch.cat([x, skip], dim=1))


class EfficientNetUNet(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        self.features = backbone.features

        # EfficientNet-B0 feature channels:
        # 32, 24, 40, 80, 112, 192, 320
        self.d1 = DecoderBlock(320, 192, 192)
        self.d2 = DecoderBlock(192, 112, 112)
        self.d3 = DecoderBlock(112, 80, 80)
        self.d4 = DecoderBlock(80, 40, 48)
        self.d5 = DecoderBlock(48, 24, 32)

        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        skips = []

        # EfficientNet stages.
        for i, layer in enumerate(self.features):
            x = layer(x)
            # EfficientNet-B0:
            # index 2=24, 3=40, 4=80, 5=112, 6=192, 7=320
            if i in [2, 3, 4, 5, 6, 7]:
                skips.append(x)

        s24, s40, s80, s112, s192, s320 = skips

        x = self.d1(s320, s192)
        x = self.d2(x, s112)
        x = self.d3(x, s80)
        x = self.d4(x, s40)
        x = self.d5(x, s24)

        x = torch.nn.functional.interpolate(
            x, 
            scale_factor=4,
            mode="bilinear", 
            align_corners=False
        )

        return self.final(x)


# ============================================================
# LOSS / METRICS
# ============================================================

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        bce = self.bce(logits, target)

        probs = torch.sigmoid(logits)
        smooth = 1.0

        intersection = (probs * target).sum(dim=(1, 2, 3))
        dice = (
            2 * intersection + smooth
        ) / (
            probs.sum(dim=(1, 2, 3))
            + target.sum(dim=(1, 2, 3))
            + smooth
        )

        dice_loss = 1 - dice.mean()

        return 0.5 * bce + 0.5 * dice_loss


def dice_score(logits, target, threshold=0.5):
    pred = (torch.sigmoid(logits) >= threshold).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    return ((2 * intersection + 1e-6) / (denom + 1e-6)).mean().item()


def iou_score(logits, target, threshold=0.5):
    pred = (torch.sigmoid(logits) >= threshold).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (
        pred.sum(dim=(1, 2, 3))
        + target.sum(dim=(1, 2, 3))
        - intersection
    )

    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()


# ============================================================
# HUMAN SEVERITY FROM PREDICTED MASK
# ============================================================

def predicted_severity(logits, leaf):
    disease = (torch.sigmoid(logits) >= 0.5).float()

    disease = disease * leaf

    disease_pixels = disease.sum(dim=(1, 2, 3))
    leaf_pixels = leaf.sum(dim=(1, 2, 3)).clamp_min(1)

    return 100.0 * disease_pixels / leaf_pixels


# ============================================================
# TRAIN / VALIDATE
# ============================================================

def run_epoch(model, loader, optimizer, criterion, scaler, train):
    model.train(train)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_severity_mae = 0.0
    n = 0

    iterator = tqdm(loader, leave=False)

    for images, leaf, disease, _ in iterator:
        images = images.to(DEVICE, non_blocking=True)
        leaf = leaf.to(DEVICE, non_blocking=True)
        disease = disease.to(DEVICE, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, disease)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        bs = images.size(0)

        total_loss += loss.item() * bs
        total_dice += dice_score(logits.detach(), disease) * bs
        total_iou += iou_score(logits.detach(), disease) * bs

        pred_sev = predicted_severity(logits.detach(), leaf)
        true_sev = (
            100.0
            * disease.sum(dim=(1, 2, 3))
            / leaf.sum(dim=(1, 2, 3)).clamp_min(1)
        )

        mae = torch.abs(pred_sev - true_sev).mean().item()

        total_severity_mae += mae * bs
        n += bs

    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
        "severity_mae": total_severity_mae / n,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("HUMAN-GT SEVERITY MODEL")
    print("EfficientNet-B0 U-Net")
    print("=" * 70)
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    df = pd.read_csv(CSV_PATH)

    # Only use successfully processed annotations.
    if "image_file" not in df.columns:
        raise RuntimeError(
            "human_gt_results.csv must contain image_file."
        )

    # Remove rows whose image/masks are unavailable.
    valid_rows = []

    for _, row in df.iterrows():
        image_file = str(row["image_file"])
        stem = Path(image_file).stem

        if (
            (IMAGE_DIR / image_file).exists()
            and (MASK_DIR / f"{stem}_leaf.png").exists()
            and (MASK_DIR / f"{stem}_disease.png").exists()
        ):
            valid_rows.append(row)

    df = pd.DataFrame(valid_rows).reset_index(drop=True)

    print("Valid annotated images:", len(df))

    # Disease-stratified 80/20 split.
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=SEED,
        stratify=df["disease"],
    )

    # Validation is taken from training only.
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.15,
        random_state=SEED,
        stratify=train_df["disease"],
    )

    print("Train:", len(train_df))
    print("Val  :", len(val_df))
    print("Test :", len(test_df))

    # Save split for reproducibility.
    train_df.assign(split="train").to_csv(
        RESULTS / "train_split.csv", index=False
    )
    val_df.assign(split="val").to_csv(
        RESULTS / "val_split.csv", index=False
    )
    test_df.assign(split="test").to_csv(
        RESULTS / "test_split.csv", index=False
    )

    train_ds = HumanSeverityDataset(train_df, train=True)
    val_ds = HumanSeverityDataset(val_df, train=False)
    test_ds = HumanSeverityDataset(test_df, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )

    model = EfficientNetUNet().to(DEVICE)

    # Verify output size before training.
    with torch.no_grad():
        dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        out = model(dummy)

    print("Model output:", tuple(out.shape))

    if tuple(out.shape[-2:]) != (IMG_SIZE, IMG_SIZE):
        raise RuntimeError(
            f"Unexpected output shape: {out.shape}"
        )

    criterion = DiceBCELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=DEVICE.type == "cuda"
    )

    best_dice = -1
    best_epoch = 0
    bad_epochs = 0

    history = []

    checkpoint_path = RESULTS / "best_human_gt_severity_unet.pth"

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_metrics = run_epoch(
            model, train_loader, optimizer,
            criterion, scaler, train=True
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, optimizer,
                criterion, scaler, train=False
            )

        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss {train_metrics['loss']:.4f} | "
            f"Dice {train_metrics['dice']:.4f} | "
            f"IoU {train_metrics['iou']:.4f}"
        )

        print(
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"Dice {val_metrics['dice']:.4f} | "
            f"IoU {val_metrics['iou']:.4f} | "
            f"Severity MAE {val_metrics['severity_mae']:.2f}% | "
            f"LR {lr_now:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": lr_now,
        })

        scheduler.step(val_metrics["dice"])

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            bad_epochs = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "img_size": IMG_SIZE,
                "best_val_dice": best_dice,
                "architecture": "EfficientNet-B0-U-Net",
                "severity_definition": "disease_pixels / leaf_pixels * 100",
                "severity_bands": {
                    "LOW": [0, 10],
                    "MEDIUM": [10, 40],
                    "HIGH": [40, 60],
                    "SEVERE": [60, 100],
                },
            }, checkpoint_path)

            print("  >>> BEST MODEL SAVED")
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print("Early stopping.")
            break

    with open(RESULTS / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ========================================================
    # FINAL TEST
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL HUMAN-GT TEST")
    print("=" * 70)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_rows = []
    all_dice = []
    all_iou = []

    with torch.no_grad():
        for images, leaf, disease, filenames in tqdm(test_loader):
            images = images.to(DEVICE)
            leaf = leaf.to(DEVICE)
            disease = disease.to(DEVICE)

            logits = model(images)

            pred = (torch.sigmoid(logits) >= 0.5).float()
            pred = pred * leaf

            true = disease * leaf

            inter = (pred * true).sum(dim=(1,2,3))
            dice = (
                (2 * inter + 1e-6) /
                (pred.sum(dim=(1,2,3)) +
                 true.sum(dim=(1,2,3)) + 1e-6)
            )

            union = (
                pred.sum(dim=(1,2,3))
                + true.sum(dim=(1,2,3))
                - inter
            )
            iou = (inter + 1e-6) / (union + 1e-6)

            pred_sev = (
                100 * pred.sum(dim=(1,2,3))
                / leaf.sum(dim=(1,2,3)).clamp_min(1)
            )

            true_sev = (
                100 * true.sum(dim=(1,2,3))
                / leaf.sum(dim=(1,2,3)).clamp_min(1)
            )

            for i, name in enumerate(filenames):
                p = float(pred_sev[i].cpu())
                t = float(true_sev[i].cpu())

                all_rows.append({
                    "image_file": name,
                    "true_severity": t,
                    "predicted_severity": p,
                    "absolute_error": abs(p - t),
                    "true_level": severity_level(t),
                    "predicted_level": severity_level(p),
                    "dice": float(dice[i].cpu()),
                    "iou": float(iou[i].cpu()),
                })

            all_dice.extend(dice.cpu().numpy().tolist())
            all_iou.extend(iou.cpu().numpy().tolist())

    results = pd.DataFrame(all_rows)
    results.to_csv(
        RESULTS / "human_gt_test_predictions.csv",
        index=False
    )

    severity_mae = results["absolute_error"].mean()
    level_acc = (
        results["true_level"] == results["predicted_level"]
    ).mean() * 100

    print("\nBEST EPOCH:", best_epoch)
    print("BEST VAL DICE:", round(best_dice, 4))
    print("TEST DICE:", round(float(np.mean(all_dice)), 4))
    print("TEST IoU:", round(float(np.mean(all_iou)), 4))
    print("TEST SEVERITY MAE:", round(float(severity_mae), 2), "%")
    print("TEST LEVEL ACCURACY:", round(float(level_acc), 2), "%")

    print("\nCheckpoint:")
    print(checkpoint_path)

    print("\nResults:")
    print(RESULTS / "human_gt_test_predictions.csv")

    print("\nDONE.")


if __name__ == "__main__":
    main()
