"""
SAM2 V2 - Automatic General Plant Disease Severity Pseudo-Label Generator

Purpose
-------
Generate fully automatic disease-area pseudo-labels for the 60-image pilot
without manual LabelMe annotation.

Pipeline
--------
1. Read general_60 metadata.
2. Resize image for SAM2.
3. Estimate the visible leaf region.
4. Generate SAM2 automatic mask proposals.
5. Keep proposals that are plausible leaf sub-regions.
6. Score each proposal using:
      - disease-specific color evidence
      - local contrast
      - texture/edge evidence
      - lesion compactness
      - overlap with the leaf
      - penalty for suspiciously large regions
7. Select a set of non-overlapping high-confidence lesion proposals.
8. Add a conservative pixel-level disease evidence mask.
9. Apply outlier protection.
10. Save severity %, mask, overlay, CSV and summary.

IMPORTANT
---------
These are pseudo-labels, NOT ground-truth disease severity.
Do not train the final model until the 60-image pilot looks reasonable.
"""

from pathlib import Path
import json
import math
import time

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = ROOT / "ai" / "severity" / "dataset" / "general_60"
RESULTS_DIR = ROOT / "ai" / "severity" / "results" / "general_severity_sam2_v2"

MASK_DIR = RESULTS_DIR / "masks"
OVERLAY_DIR = RESULTS_DIR / "overlays"

CHECKPOINT = (
    ROOT
    / "ai"
    / "severity"
    / "models"
    / "sam2_repo"
    / "checkpoints"
    / "sam2.1_hiera_tiny.pt"
)

CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_t.yaml"


# ============================================================
# SAM2 SETTINGS
# ============================================================

MAX_IMAGE_SIDE = 1024

POINTS_PER_SIDE = 20
PRED_IOU_THRESHOLD = 0.72
STABILITY_SCORE_THRESHOLD = 0.72
MIN_MASK_REGION_AREA = 80

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# GENERAL SETTINGS
# ============================================================

MIN_LEAF_AREA_RATIO = 0.12
MAX_LEAF_AREA_RATIO = 0.92

MIN_PROPOSAL_AREA_RATIO = 0.00015
MAX_PROPOSAL_AREA_RATIO = 0.30

MIN_COMPONENT_PIXELS = 8

# Maximum final pseudo-severity.
# This prevents one catastrophic proposal-selection failure
# from becoming a 95-100% label.
MAX_FINAL_SEVERITY = 70.0

# If the result is extremely high, apply stronger sanity checks.
HIGH_SEVERITY_LIMIT = 45.0

# Small lesions need to survive.
SMALL_LESION_BOOST = 1.20

# Large regions are increasingly penalized.
LARGE_REGION_PENALTY_START = 0.08


# ============================================================
# DISEASE NORMALIZATION
# ============================================================

def normalize_name(text: str) -> str:
    text = str(text)
    text = text.replace("___", " ")
    text = text.replace("_", " ")
    text = text.replace("  ", " ")
    return text.lower().strip()


# ============================================================
# DISEASE PROFILES
# ============================================================

# The profiles intentionally use broad visual evidence rather than
# hard-coded severity percentages.
#
# They are NOT treatment rules and NOT biological ground truth.

DISEASE_PROFILES = {
    "bacterial spot": {
        "colors": ["dark", "brown", "yellow"],
        "small_lesion": 1.30,
        "contrast": 1.15,
        "texture": 1.05,
        "large_penalty": 1.35,
    },
    "early blight": {
        "colors": ["dark", "brown", "yellow"],
        "small_lesion": 1.20,
        "contrast": 1.20,
        "texture": 1.20,
        "large_penalty": 1.35,
    },
    "late blight": {
        "colors": ["dark", "brown", "yellow"],
        "small_lesion": 1.15,
        "contrast": 1.20,
        "texture": 1.15,
        "large_penalty": 1.30,
    },
    "leaf mold": {
        "colors": ["dark", "yellow", "brown"],
        "small_lesion": 1.10,
        "contrast": 1.10,
        "texture": 1.25,
        "large_penalty": 1.30,
    },
    "cercospora leaf spot gray leaf spot": {
        "colors": ["dark", "brown", "yellow"],
        "small_lesion": 1.20,
        "contrast": 1.25,
        "texture": 1.15,
        "large_penalty": 1.40,
    },
    "common rust": {
        "colors": ["brown", "orange", "dark"],
        "small_lesion": 1.45,
        "contrast": 1.25,
        "texture": 1.10,
        "large_penalty": 1.40,
    },
    "northern leaf blight": {
        "colors": ["dark", "brown", "yellow"],
        "small_lesion": 1.15,
        "contrast": 1.25,
        "texture": 1.15,
        "large_penalty": 1.40,
    },
    "apple scab": {
        "colors": ["dark", "brown", "yellow"],
        "small_lesion": 1.30,
        "contrast": 1.20,
        "texture": 1.10,
        "large_penalty": 1.40,
    },
    "black rot": {
        "colors": ["dark", "brown"],
        "small_lesion": 1.20,
        "contrast": 1.25,
        "texture": 1.15,
        "large_penalty": 1.40,
    },
    "cedar apple rust": {
        "colors": ["orange", "brown", "yellow", "dark"],
        "small_lesion": 1.40,
        "contrast": 1.25,
        "texture": 1.10,
        "large_penalty": 1.40,
    },
}


def get_profile(disease: str):
    name = normalize_name(disease)

    if name in DISEASE_PROFILES:
        return DISEASE_PROFILES[name]

    # Handle PlantVillage's exact gray-leaf-spot naming.
    if "gray leaf spot" in name or "cercospora" in name:
        return DISEASE_PROFILES["cercospora leaf spot gray leaf spot"]

    if "rust" in name:
        return DISEASE_PROFILES["common rust"]

    if "early blight" in name:
        return DISEASE_PROFILES["early blight"]

    if "late blight" in name:
        return DISEASE_PROFILES["late blight"]

    if "bacterial spot" in name:
        return DISEASE_PROFILES["bacterial spot"]

    if "leaf mold" in name:
        return DISEASE_PROFILES["leaf mold"]

    if "scab" in name:
        return DISEASE_PROFILES["apple scab"]

    if "black rot" in name:
        return DISEASE_PROFILES["black rot"]

    return DISEASE_PROFILES["early blight"]


# ============================================================
# IMAGE UTILITIES
# ============================================================

def resize_keep_aspect(image, max_side=MAX_IMAGE_SIDE):
    h, w = image.shape[:2]

    scale = min(1.0, max_side / max(h, w))

    if scale == 1.0:
        return image.copy(), 1.0

    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return resized, scale


def normalize01(x):
    x = np.asarray(x, dtype=np.float32)

    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)

    if hi - lo < 1e-6:
        return np.zeros_like(x)

    return np.clip((x - lo) / (hi - lo), 0, 1)


# ============================================================
# LEAF ESTIMATION
# ============================================================

def estimate_leaf_mask(image):
    """
    Conservative leaf/background estimation.

    Uses multiple cues instead of assuming that every green pixel
    is diseased or every non-green pixel is background.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)

    green = (
        (h >= 25)
        & (h <= 95)
        & (s >= 35)
        & (v >= 30)
    )

    yellow_green = (
        (h >= 15)
        & (h < 40)
        & (s >= 30)
        & (v >= 35)
    )

    brown_leaf = (
        (h >= 5)
        & (h <= 30)
        & (s >= 35)
        & (v >= 25)
        & (l >= 35)
        & (l <= 210)
    )

    candidate = green | yellow_green | brown_leaf

    candidate = candidate.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep large connected components.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)

    mask = np.zeros_like(candidate)

    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        order = np.argsort(areas)[::-1]

        total = candidate.shape[0] * candidate.shape[1]

        for idx in order[:5]:
            label = idx + 1
            area = stats[label, cv2.CC_STAT_AREA]

            if area < 0.01 * total:
                continue

            mask[labels == label] = 255

    # Fallback: use foreground from GrabCut if color segmentation is weak.
    ratio = np.count_nonzero(mask) / mask.size

    if ratio < MIN_LEAF_AREA_RATIO or ratio > MAX_LEAF_AREA_RATIO:
        gc = np.zeros(image.shape[:2], np.uint8)
        gc[:] = cv2.GC_PR_BGD

        border = max(2, min(image.shape[:2]) // 30)
        gc[:border, :] = cv2.GC_BGD
        gc[-border:, :] = cv2.GC_BGD
        gc[:, :border] = cv2.GC_BGD
        gc[:, -border:] = cv2.GC_BGD

        # Use non-border pixels as probable foreground.
        gc[border:-border, border:-border] = cv2.GC_PR_FGD

        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(
                image,
                gc,
                None,
                bgd,
                fgd,
                3,
                cv2.GC_INIT_WITH_MASK,
            )

            grab = np.where(
                (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)

            grab = cv2.morphologyEx(
                grab,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=2,
            )

            if np.count_nonzero(grab) > np.count_nonzero(mask):
                mask = grab
        except Exception:
            pass

    # Final cleanup.
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

    return mask > 0


# ============================================================
# DISEASE EVIDENCE
# ============================================================

def disease_pixel_evidence(image, disease):
    """
    Conservative pixel-level disease evidence.

    This is used as one scoring signal, not as a hard disease mask.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    l = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Local darkness / local contrast.
    blur = cv2.GaussianBlur(gray, (0, 0), 7)
    local_dark = normalize01(np.maximum(blur - gray, 0))

    local_std = cv2.GaussianBlur(
        (gray - cv2.GaussianBlur(gray, (0, 0), 5)) ** 2,
        (0, 0),
        3,
    )
    local_texture = normalize01(np.sqrt(np.maximum(local_std, 0)))

    dark = np.clip((135 - v) / 100, 0, 1)

    brown = (
        (h >= 5)
        & (h <= 30)
        & (s >= 35)
        & (v >= 25)
    ).astype(np.float32)

    orange = (
        (h >= 5)
        & (h <= 22)
        & (s >= 70)
        & (v >= 40)
    ).astype(np.float32)

    yellow = (
        (h >= 18)
        & (h <= 42)
        & (s >= 30)
        & (v >= 45)
    ).astype(np.float32)

    green = (
        (h >= 25)
        & (h <= 95)
        & (s >= 35)
        & (v >= 30)
    ).astype(np.float32)

    # Difference from normal green-leaf chroma.
    green_deviation = np.clip(
        np.abs(a - 115) / 45 + np.abs(b - 135) / 55,
        0,
        1,
    )

    name = normalize_name(disease)

    if "common rust" in name or "cedar apple rust" in name:
        color = np.maximum(orange, brown)
        score = (
            0.38 * color
            + 0.24 * local_dark
            + 0.18 * local_texture
            + 0.20 * green_deviation
        )

    elif "bacterial spot" in name:
        color = np.maximum.reduce([dark, brown * 0.9, yellow * 0.6])
        score = (
            0.30 * color
            + 0.30 * local_dark
            + 0.18 * local_texture
            + 0.22 * green_deviation
        )

    elif "leaf mold" in name:
        color = np.maximum.reduce([dark, yellow * 0.9, brown])
        score = (
            0.28 * color
            + 0.25 * local_dark
            + 0.25 * local_texture
            + 0.22 * green_deviation
        )

    else:
        color = np.maximum.reduce([dark, brown, yellow * 0.75])
        score = (
            0.32 * color
            + 0.26 * local_dark
            + 0.20 * local_texture
            + 0.22 * green_deviation
        )

    # Suppress strongly healthy green regions.
    healthy_penalty = np.clip(
        green * (1.0 - green_deviation),
        0,
        1,
    )

    score = score * (1.0 - 0.42 * healthy_penalty)

    return np.clip(score, 0, 1).astype(np.float32)


# ============================================================
# PROPOSAL SCORING
# ============================================================

def bbox_from_mask(mask):
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def mask_iou(a, b):
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)

    if union == 0:
        return 0.0

    return inter / union


def proposal_score(
    proposal,
    leaf_mask,
    disease_score,
    profile,
    image_area,
):
    mask = proposal["segmentation"].astype(bool)

    area = np.count_nonzero(mask)

    if area == 0:
        return -1.0, {}

    leaf_overlap = np.count_nonzero(mask & leaf_mask) / area

    if leaf_overlap < 0.55:
        return -1.0, {}

    disease_values = disease_score[mask]
    evidence = float(np.mean(disease_values))

    # Focus on the strongest pixels within the proposal.
    top_fraction = max(1, int(len(disease_values) * 0.25))
    top_values = np.partition(
        disease_values,
        -top_fraction,
    )[-top_fraction:]

    top_evidence = float(np.mean(top_values))

    # A lesion region should contain stronger evidence than the
    # average image region.
    compactness = min(1.0, top_evidence * 1.25)

    area_ratio = area / image_area

    # Very small regions are not automatically discarded.
    if area_ratio < 0.002:
        small_bonus = SMALL_LESION_BOOST
    else:
        small_bonus = 1.0

    # Large SAM2 regions are dangerous because they often include
    # healthy leaf tissue.
    if area_ratio <= LARGE_REGION_PENALTY_START:
        large_penalty = 1.0
    else:
        excess = (
            area_ratio - LARGE_REGION_PENALTY_START
        ) / max(1e-6, 0.30 - LARGE_REGION_PENALTY_START)

        large_penalty = max(
            0.05,
            1.0 - profile["large_penalty"] * excess,
        )

    sam_quality = (
        0.5 * float(proposal.get("predicted_iou", 0))
        + 0.5 * float(proposal.get("stability_score", 0))
    )

    score = (
        0.34 * evidence
        + 0.26 * top_evidence
        + 0.12 * compactness
        + 0.12 * leaf_overlap
        + 0.16 * sam_quality
    )

    score *= small_bonus
    score *= large_penalty

    details = {
        "area": area,
        "area_ratio": area_ratio,
        "leaf_overlap": leaf_overlap,
        "evidence": evidence,
        "top_evidence": top_evidence,
        "compactness": compactness,
        "sam_quality": sam_quality,
        "large_penalty": large_penalty,
        "score": score,
    }

    return float(score), details


# ============================================================
# SELECT LESION PROPOSALS
# ============================================================

def select_proposals(
    proposals,
    leaf_mask,
    disease_score,
    disease,
):
    profile = get_profile(disease)
    image_area = leaf_mask.size

    ranked = []

    for p in proposals:
        mask = p["segmentation"].astype(bool)

        area = np.count_nonzero(mask)

        if area < MIN_PROPOSAL_AREA_RATIO * image_area:
            continue

        if area > MAX_PROPOSAL_AREA_RATIO * image_area:
            continue

        score, details = proposal_score(
            p,
            leaf_mask,
            disease_score,
            profile,
            image_area,
        )

        if score <= 0:
            continue

        # Minimum disease evidence.
        if details["evidence"] < 0.10:
            continue

        ranked.append((score, p, details))

    ranked.sort(key=lambda x: x[0], reverse=True)

    selected = []
    union = np.zeros_like(leaf_mask, dtype=bool)

    for score, proposal, details in ranked:
        mask = proposal["segmentation"].astype(bool)

        # Reject almost-duplicate proposals.
        max_iou = 0.0

        for old in selected:
            old_mask = old["proposal"]["segmentation"].astype(bool)
            max_iou = max(max_iou, mask_iou(mask, old_mask))

        if max_iou > 0.70:
            continue

        # Do not allow selected proposals to cover most of the leaf.
        proposed_union = union | mask
        union_ratio = np.count_nonzero(
            proposed_union & leaf_mask
        ) / max(1, np.count_nonzero(leaf_mask))

        if union_ratio > 0.45:
            continue

        selected.append(
            {
                "proposal": proposal,
                "score": score,
                "details": details,
            }
        )

        union = proposed_union

        if len(selected) >= 18:
            break

    return selected


# ============================================================
# BUILD FINAL DISEASE MASK
# ============================================================

def build_disease_mask(
    image,
    leaf_mask,
    disease_score,
    selected,
):
    mask = np.zeros_like(leaf_mask, dtype=np.uint8)

    # Proposal masks.
    for item in selected:
        proposal = item["proposal"]
        score = item["score"]

        p = proposal["segmentation"].astype(bool)

        # Only retain pixels that have some disease evidence.
        evidence = disease_score >= 0.22

        # Higher quality proposals get a slightly lower threshold.
        threshold = 0.18 if score > 0.50 else 0.24

        evidence = disease_score >= threshold

        component = p & evidence & leaf_mask

        mask[component] = 255

    # Add conservative high-evidence pixels that were not captured
    # by SAM2, but only when they are surrounded by disease evidence.
    high = (disease_score >= 0.72) & leaf_mask

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    high = cv2.morphologyEx(
        high.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        kernel,
    ) > 0

    # Only add small, isolated high-confidence regions.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        high.astype(np.uint8),
        8,
    )

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]

        if 8 <= area <= 5000:
            mask[labels == i] = 255

    # Clean speckle but preserve small lesions.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2, 2),
        ),
    )

    # Never leave the leaf.
    mask[~leaf_mask] = 0

    return mask > 0


# ============================================================
# OUTLIER PROTECTION
# ============================================================

def robust_severity(
    disease_mask,
    leaf_mask,
    selected,
    disease_score,
):
    leaf_pixels = int(np.count_nonzero(leaf_mask))

    if leaf_pixels == 0:
        return 0.0, "invalid_leaf"

    disease_pixels = int(np.count_nonzero(disease_mask))

    raw = 100.0 * disease_pixels / leaf_pixels

    # If there are no credible proposals and only weak pixel evidence,
    # keep severity conservative.
    if len(selected) == 0 and raw > 5:
        raw = min(raw, 5.0)

    # High severity requires stronger evidence.
    if raw > HIGH_SEVERITY_LIMIT:
        strong_ratio = (
            np.count_nonzero(
                (disease_score >= 0.55) & leaf_mask
            )
            / leaf_pixels
        )

        proposal_area = sum(
            item["details"]["area"]
            for item in selected
        )

        proposal_ratio = proposal_area / max(1, leaf_pixels)

        # If a huge label is not supported by strong evidence,
        # shrink it to a conservative upper bound.
        if strong_ratio < 0.25 and proposal_ratio > 0.30:
            raw = min(raw, 35.0)

        elif strong_ratio < 0.15:
            raw = min(raw, 30.0)

    final = float(np.clip(raw, 0, MAX_FINAL_SEVERITY))

    if final < 5:
        quality = "low_to_moderate"
    elif final < 20:
        quality = "moderate"
    elif final < 40:
        quality = "high"
    else:
        quality = "review_required"

    return final, quality


# ============================================================
# OVERLAY
# ============================================================

def make_overlay(image, leaf_mask, disease_mask, severity):
    overlay = image.copy()

    # Leaf boundary.
    contours, _ = cv2.findContours(
        leaf_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (255, 255, 0),
        2,
    )

    # Disease region.
    color = np.zeros_like(image)
    color[:, :, 2] = 255

    disease_bool = disease_mask.astype(bool)

    overlay[disease_bool] = (
        0.55 * overlay[disease_bool]
        + 0.45 * color[disease_bool]
    ).astype(np.uint8)

    text = f"Pseudo severity: {severity:.2f}%"

    cv2.putText(
        overlay,
        text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return overlay


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    disease,
    mask_generator,
):
    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    image, scale = resize_keep_aspect(image_bgr)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    leaf_mask = estimate_leaf_mask(image)

    disease_score = disease_pixel_evidence(
        image,
        disease,
    )

    # Restrict disease evidence to leaf.
    disease_score = disease_score * leaf_mask.astype(np.float32)

    proposals = mask_generator.generate(rgb)

    selected = select_proposals(
        proposals,
        leaf_mask,
        disease_score,
        disease,
    )

    disease_mask = build_disease_mask(
        image,
        leaf_mask,
        disease_score,
        selected,
    )

    severity, quality = robust_severity(
        disease_mask,
        leaf_mask,
        selected,
        disease_score,
    )

    overlay = make_overlay(
        image,
        leaf_mask,
        disease_mask,
        severity,
    )

    return {
        "image": image,
        "leaf_mask": leaf_mask,
        "disease_mask": disease_mask,
        "overlay": overlay,
        "severity": severity,
        "quality": quality,
        "proposal_count": len(proposals),
        "selected_count": len(selected),
        "leaf_ratio": float(
            np.count_nonzero(leaf_mask) / leaf_mask.size
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_DIR}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found:\n{CHECKPOINT}"
        )

    metadata_path = DATASET_DIR / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found:\n{metadata_path}"
        )

    df = pd.read_csv(metadata_path)

    required = {
        "filename",
        "plant",
        "disease",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing)}"
        )

    print("=" * 70)
    print("SAM2 V2 GENERAL SEVERITY PILOT")
    print("=" * 70)
    print(f"Dataset : {DATASET_DIR}")
    print(f"Images  : {len(df)}")
    print(f"Device  : {DEVICE}")
    print(f"Output  : {RESULTS_DIR}")
    print()

    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    print("\nLoading SAM2...")

    sam2_model = build_sam2(
        CONFIG_NAME,
        str(CHECKPOINT),
        device=DEVICE,
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESHOLD,
        stability_score_thresh=STABILITY_SCORE_THRESHOLD,
        min_mask_region_area=MIN_MASK_REGION_AREA,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )

    print("SAM2 loaded.\n")

    rows = []
    start = time.time()

    for idx, row in enumerate(
        tqdm(
            df.itertuples(index=False),
            total=len(df),
            desc="Processing",
        ),
        start=1,
    ):
        image_path = Path(row.image_path)

        if not image_path.exists():
            # Fall back to general_60 folder structure.
            image_path = (
                DATASET_DIR
                / str(row.plant)
                / str(row.disease)
                / str(row.filename)
            )

        if not image_path.exists():
            print(
                f"\nWARNING: image missing: {row.filename}"
            )
            continue

        try:
            result = process_image(
                image_path,
                row.disease,
                mask_generator,
            )

            stem = Path(row.filename).stem

            safe_class = (
                str(row.disease)
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
            )

            output_stem = f"{safe_class}__{stem}"

            mask_path = MASK_DIR / f"{output_stem}_mask.png"
            overlay_path = (
                OVERLAY_DIR / f"{output_stem}_overlay.jpg"
            )

            cv2.imwrite(
                str(mask_path),
                result["disease_mask"].astype(np.uint8) * 255,
            )

            cv2.imwrite(
                str(overlay_path),
                result["overlay"],
            )

            rows.append(
                {
                    "filename": row.filename,
                    "plant": row.plant,
                    "disease": row.disease,
                    "severity": result["severity"],
                    "quality": result["quality"],
                    "leaf_ratio": result["leaf_ratio"],
                    "proposal_count": result["proposal_count"],
                    "selected_proposals": result["selected_count"],
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                }
            )

        except Exception as exc:
            print(
                f"\nERROR: {row.filename}: {exc}"
            )

            rows.append(
                {
                    "filename": row.filename,
                    "plant": row.plant,
                    "disease": row.disease,
                    "severity": np.nan,
                    "quality": "error",
                    "leaf_ratio": np.nan,
                    "proposal_count": 0,
                    "selected_proposals": 0,
                    "mask_path": "",
                    "overlay_path": "",
                }
            )

    elapsed = time.time() - start

    out_df = pd.DataFrame(rows)

    csv_path = RESULTS_DIR / "severity_labels_sam2_v2.csv"
    out_df.to_csv(csv_path, index=False)

    valid = out_df[
        pd.to_numeric(
            out_df["severity"],
            errors="coerce",
        ).notna()
    ].copy()

    if len(valid):
        severity_values = valid["severity"].astype(float)

        bins = {
            "0-5": int(
                ((severity_values >= 0) & (severity_values < 5)).sum()
            ),
            "5-10": int(
                ((severity_values >= 5) & (severity_values < 10)).sum()
            ),
            "10-20": int(
                ((severity_values >= 10) & (severity_values < 20)).sum()
            ),
            "20-30": int(
                ((severity_values >= 20) & (severity_values < 30)).sum()
            ),
            "30-50": int(
                ((severity_values >= 30) & (severity_values < 50)).sum()
            ),
            "50+": int(
                (severity_values >= 50).sum()
            ),
        }

        disease_summary = (
            valid.groupby(["plant", "disease"])["severity"]
            .agg(
                [
                    "count",
                    "mean",
                    "median",
                    "min",
                    "max",
                ]
            )
            .reset_index()
        )

        disease_summary_path = (
            RESULTS_DIR / "disease_summary.csv"
        )

        disease_summary.to_csv(
            disease_summary_path,
            index=False,
        )

        summary = {
            "pipeline": "SAM2 V2 automatic pseudo-label generator",
            "images_processed": int(len(valid)),
            "images_requested": int(len(df)),
            "runtime_seconds": round(elapsed, 2),
            "device": DEVICE,
            "mean_severity": float(severity_values.mean()),
            "median_severity": float(severity_values.median()),
            "std_severity": float(severity_values.std()),
            "min_severity": float(severity_values.min()),
            "max_severity": float(severity_values.max()),
            "distribution": bins,
            "review_required_count": int(
                (valid["quality"] == "review_required").sum()
            ),
            "warning": (
                "These are automatically generated pseudo-labels. "
                "They are not independently verified ground-truth "
                "disease severity measurements."
            ),
        }

    else:
        summary = {
            "pipeline": "SAM2 V2 automatic pseudo-label generator",
            "images_processed": 0,
            "images_requested": int(len(df)),
            "runtime_seconds": round(elapsed, 2),
            "device": DEVICE,
            "warning": "No valid predictions were produced.",
        }

    summary_path = RESULTS_DIR / "summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Processed : {len(valid)}/{len(df)}")
    print(f"Runtime   : {elapsed:.1f} sec")
    print(f"CSV       : {csv_path}")
    print(f"Summary   : {summary_path}")

    if len(valid):
        print(
            f"Mean      : {valid['severity'].mean():.2f}%"
        )
        print(
            f"Median    : {valid['severity'].median():.2f}%"
        )
        print(
            f"Min       : {valid['severity'].min():.2f}%"
        )
        print(
            f"Max       : {valid['severity'].max():.2f}%"
        )

        print("\nDistribution:")
        for key, value in summary["distribution"].items():
            print(f"  {key:>6}: {value}")

        print("\nDisease summary:")
        print(
            valid.groupby(
                ["plant", "disease"]
            )["severity"]
            .agg(["mean", "median", "min", "max"])
            .round(2)
        )

    print("\nIMPORTANT:")
    print(
        "Do NOT train the final severity model yet. "
        "Inspect the pilot output first."
    )


if __name__ == "__main__":
    main()
