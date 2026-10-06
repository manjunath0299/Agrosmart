"""
QC for 3,000-image SAM2 pseudo-label severity dataset.

Purpose:
    - Do NOT regenerate SAM2 masks.
    - Inspect existing severity CSV.
    - Identify suspicious pseudo-labels.
    - Assign HIGH_CONFIDENCE / REVIEW / REJECT.
    - Create cleaned training CSV.
    - Preserve the original CSV.

Important:
    SAM2 pseudo-labels are NOT ground-truth annotations.
"""

from pathlib import Path
import json
import re

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"E:\smart_agriculture\smart_agriculture")

INPUT_CSV = (
    ROOT
    / "ai"
    / "severity"
    / "results"
    / "general_severity_sam2_3000"
    / "severity_labels_sam2_3000.csv"
)

OUTPUT_DIR = (
    ROOT
    / "ai"
    / "severity"
    / "results"
    / "general_severity_sam2_3000_qc"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# QC THRESHOLDS
# ============================================================

# These are intentionally conservative.

VERY_LOW_SEVERITY = 0.20

LOW_SEVERITY = 1.0

REVIEW_SEVERITY = 45.0

REJECT_SEVERITY = 60.0

EXTREME_SEVERITY = 65.0

MIN_SAM_PROPOSALS = 2

MIN_RANKED_PROPOSALS = 1

MIN_SELECTED_PROPOSALS = 1

# Disease fraction relative to entire image.
LARGE_DISEASE_IMAGE_FRACTION = 0.35

EXTREME_DISEASE_IMAGE_FRACTION = 0.50

# Leaf should normally occupy a meaningful part of the image.
VERY_SMALL_LEAF_FRACTION = 0.08

VERY_LARGE_LEAF_FRACTION = 0.90


# ============================================================
# HELPERS
# ============================================================

def split_flags(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(";")
        if x.strip()
    ]


def has_flag(flags, keyword):
    return any(keyword in flag.lower() for flag in flags)


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def calculate_qc(row):
    """
    Calculate conservative QC score.

    Returns:
        qc_score
        qc_status
        qc_reasons
    """

    severity = safe_float(row.get("severity"))
    leaf_fraction = safe_float(row.get("leaf_area_fraction"))
    disease_image_fraction = safe_float(
        row.get("disease_area_fraction_of_image")
    )

    sam_proposals = safe_float(
        row.get("num_sam_proposals"),
        default=0
    )

    ranked_proposals = safe_float(
        row.get("num_ranked_proposals"),
        default=0
    )

    selected_proposals = safe_float(
        row.get("num_selected_proposals"),
        default=0
    )

    flags = split_flags(row.get("quality_flags"))

    score = 100
    reasons = []

    # --------------------------------------------------------
    # Existing quality flags
    # --------------------------------------------------------

    if has_flag(flags, "severity_pixel_mismatch"):
        score -= 60
        reasons.append("severity_pixel_mismatch")

    if has_flag(flags, "extreme_severity"):
        score -= 30
        reasons.append("extreme_severity")

    if has_flag(flags, "high_severity_review"):
        score -= 20
        reasons.append("high_severity_review")

    if has_flag(flags, "large_disease_image_fraction"):
        score -= 20
        reasons.append("large_disease_image_fraction")

    if has_flag(flags, "low_sam_proposals"):
        score -= 10
        reasons.append("low_sam_proposals")

    if has_flag(flags, "very_low_severity"):
        score -= 8
        reasons.append("very_low_severity")

    if has_flag(flags, "very_small_disease_region"):
        score -= 8
        reasons.append("very_small_disease_region")

    if has_flag(flags, "very_large_leaf_fraction"):
        score -= 8
        reasons.append("very_large_leaf_fraction")

    # --------------------------------------------------------
    # Severity-based checks
    # --------------------------------------------------------

    if not np.isnan(severity):

        if severity >= EXTREME_SEVERITY:
            score -= 35
            reasons.append("extreme_severity_value")

        elif severity >= REJECT_SEVERITY:
            score -= 25
            reasons.append("very_high_severity")

        elif severity >= REVIEW_SEVERITY:
            score -= 15
            reasons.append("high_severity")

        if severity <= VERY_LOW_SEVERITY:
            score -= 5
            reasons.append("near_zero_severity")

    # --------------------------------------------------------
    # Leaf fraction
    # --------------------------------------------------------

    if not np.isnan(leaf_fraction):

        if leaf_fraction < VERY_SMALL_LEAF_FRACTION:
            score -= 20
            reasons.append("very_small_leaf_fraction")

        if leaf_fraction > VERY_LARGE_LEAF_FRACTION:
            score -= 8
            reasons.append("very_large_leaf_fraction")

    # --------------------------------------------------------
    # Disease area relative to whole image
    # --------------------------------------------------------

    if not np.isnan(disease_image_fraction):

        if disease_image_fraction >= EXTREME_DISEASE_IMAGE_FRACTION:
            score -= 35
            reasons.append("extreme_disease_image_fraction")

        elif disease_image_fraction >= LARGE_DISEASE_IMAGE_FRACTION:
            score -= 20
            reasons.append("large_disease_image_fraction")

    # --------------------------------------------------------
    # Proposal counts
    # --------------------------------------------------------

    if sam_proposals < MIN_SAM_PROPOSALS:
        score -= 15
        reasons.append("too_few_sam_proposals")

    if ranked_proposals < MIN_RANKED_PROPOSALS:
        score -= 10
        reasons.append("too_few_ranked_proposals")

    if selected_proposals < MIN_SELECTED_PROPOSALS:
        score -= 15
        reasons.append("no_selected_proposal")

    # --------------------------------------------------------
    # Special consistency checks
    # --------------------------------------------------------

    if (
        not np.isnan(severity)
        and severity >= REVIEW_SEVERITY
        and not np.isnan(disease_image_fraction)
        and disease_image_fraction >= LARGE_DISEASE_IMAGE_FRACTION
    ):
        score -= 15
        reasons.append("high_severity_large_region")

    score = max(0, min(100, score))

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if (
        has_flag(flags, "severity_pixel_mismatch")
        or score < 40
        or severity >= EXTREME_SEVERITY
    ):
        status = "REJECT"

    elif score < 70 or severity >= REVIEW_SEVERITY:
        status = "REVIEW"

    else:
        status = "HIGH_CONFIDENCE"

    # Remove duplicate reasons.
    reasons = list(dict.fromkeys(reasons))

    return score, status, ";".join(reasons)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("GENERAL SEVERITY — SAM2 3000 QC")
print("=" * 80)

print(f"\nInput CSV:")
print(INPUT_CSV)

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Input CSV not found:\n{INPUT_CSV}"
    )

df = pd.read_csv(INPUT_CSV)

print(f"\nRows loaded: {len(df)}")

if len(df) != 3000:
    print(
        f"WARNING: Expected 3000 rows, found {len(df)}"
    )


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "image_path",
    "filename",
    "plant",
    "disease",
    "severity",
    "severity_level",
    "leaf_pixels",
    "disease_pixels",
    "leaf_area_fraction",
    "disease_area_fraction_of_image",
    "num_sam_proposals",
    "num_ranked_proposals",
    "num_selected_proposals",
    "quality_flags",
    "status",
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns:\n{missing}"
    )


# ============================================================
# NUMERIC CLEANUP
# ============================================================

numeric_columns = [
    "severity",
    "leaf_pixels",
    "disease_pixels",
    "leaf_area_fraction",
    "disease_area_fraction_of_image",
    "num_sam_proposals",
    "num_ranked_proposals",
    "num_selected_proposals",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# RUN QC
# ============================================================

print("\nRunning QC...")

qc_results = []

for idx, row in df.iterrows():

    score, status, reasons = calculate_qc(row)

    qc_results.append(
        {
            "qc_score": score,
            "qc_status": status,
            "qc_reasons": reasons,
        }
    )

    if (idx + 1) % 500 == 0:
        print(
            f"Processed {idx + 1}/{len(df)}"
        )


qc_df = pd.DataFrame(qc_results)

df = pd.concat(
    [
        df.reset_index(drop=True),
        qc_df.reset_index(drop=True)
    ],
    axis=1
)


# ============================================================
# SAVE COMPLETE QC DATASET
# ============================================================

all_qc_csv = (
    OUTPUT_DIR
    / "severity_labels_sam2_3000_with_qc.csv"
)

df.to_csv(
    all_qc_csv,
    index=False
)

print(
    f"\nSaved:\n{all_qc_csv}"
)


# ============================================================
# STATUS COUNTS
# ============================================================

status_counts = (
    df["qc_status"]
    .value_counts()
    .reindex(
        [
            "HIGH_CONFIDENCE",
            "REVIEW",
            "REJECT"
        ],
        fill_value=0
    )
)

print("\n" + "=" * 80)
print("QC STATUS")
print("=" * 80)

for status, count in status_counts.items():

    percentage = (
        count / len(df) * 100
        if len(df)
        else 0
    )

    print(
        f"{status:20s}: "
        f"{count:4d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# CLEAN TRAINING DATA
# ============================================================

# IMPORTANT:
# REVIEW and REJECT are NOT used for training.

clean_df = df[
    df["qc_status"] == "HIGH_CONFIDENCE"
].copy()

clean_csv = (
    OUTPUT_DIR
    / "severity_labels_sam2_3000_clean.csv"
)

clean_df.to_csv(
    clean_csv,
    index=False
)

print(
    f"\nClean training rows: {len(clean_df)}"
)

print(
    f"Saved clean dataset:\n{clean_csv}"
)


# ============================================================
# REVIEW DATASET
# ============================================================

review_df = df[
    df["qc_status"] == "REVIEW"
].copy()

review_csv = (
    OUTPUT_DIR
    / "severity_labels_sam2_3000_review.csv"
)

review_df.to_csv(
    review_csv,
    index=False
)


# ============================================================
# REJECT DATASET
# ============================================================

reject_df = df[
    df["qc_status"] == "REJECT"
].copy()

reject_csv = (
    OUTPUT_DIR
    / "severity_labels_sam2_3000_reject.csv"
)

reject_df.to_csv(
    reject_csv,
    index=False
)


# ============================================================
# DISEASE-WISE STATISTICS
# ============================================================

disease_stats = []

for disease, group in df.groupby("disease"):

    clean_group = group[
        group["qc_status"] == "HIGH_CONFIDENCE"
    ]

    disease_stats.append(
        {
            "disease": disease,
            "total": len(group),

            "high_confidence": len(clean_group),

            "review": int(
                (group["qc_status"] == "REVIEW").sum()
            ),

            "reject": int(
                (group["qc_status"] == "REJECT").sum()
            ),

            "mean_severity_all":
                group["severity"].mean(),

            "median_severity_all":
                group["severity"].median(),

            "mean_severity_clean":
                clean_group["severity"].mean()
                if len(clean_group)
                else np.nan,

            "median_severity_clean":
                clean_group["severity"].median()
                if len(clean_group)
                else np.nan,

            "min_severity":
                group["severity"].min(),

            "max_severity":
                group["severity"].max(),
        }
    )

disease_stats_df = pd.DataFrame(
    disease_stats
)

disease_stats_csv = (
    OUTPUT_DIR
    / "disease_wise_qc_statistics.csv"
)

disease_stats_df.to_csv(
    disease_stats_csv,
    index=False
)


# ============================================================
# FLAG STATISTICS
# ============================================================

flag_counter = {}

for value in df["quality_flags"]:

    for flag in split_flags(value):

        flag_counter[flag] = (
            flag_counter.get(flag, 0) + 1
        )

flag_df = pd.DataFrame(
    [
        {
            "flag": flag,
            "count": count,
        }
        for flag, count
        in sorted(
            flag_counter.items(),
            key=lambda x: x[1],
            reverse=True
        )
    ]
)

flag_csv = (
    OUTPUT_DIR
    / "quality_flag_statistics.csv"
)

flag_df.to_csv(
    flag_csv,
    index=False
)


# ============================================================
# SEVERITY DISTRIBUTION
# ============================================================

bins = [
    0,
    5,
    10,
    20,
    30,
    40,
    50,
    60,
    70.0001
]

labels = [
    "0-5",
    "5-10",
    "10-20",
    "20-30",
    "30-40",
    "40-50",
    "50-60",
    "60-70"
]

df["severity_bin"] = pd.cut(
    df["severity"],
    bins=bins,
    labels=labels,
    right=False
)

distribution = (
    df["severity_bin"]
    .value_counts()
    .reindex(labels, fill_value=0)
)

distribution_df = pd.DataFrame(
    {
        "severity_range": distribution.index,
        "count": distribution.values,
    }
)

distribution_df["percentage"] = (
    distribution_df["count"]
    / len(df)
    * 100
)

distribution_csv = (
    OUTPUT_DIR
    / "severity_distribution.csv"
)

distribution_df.to_csv(
    distribution_csv,
    index=False
)


# ============================================================
# WORST / MOST SUSPICIOUS EXAMPLES
# ============================================================

worst_df = (
    df.sort_values(
        [
            "qc_score",
            "severity"
        ],
        ascending=[
            True,
            False
        ]
    )
    .head(300)
)

worst_csv = (
    OUTPUT_DIR
    / "most_suspicious_300.csv"
)

worst_df.to_csv(
    worst_csv,
    index=False
)


# ============================================================
# HIGH SEVERITY REVIEW LIST
# ============================================================

high_severity_df = df[
    df["severity"] >= REVIEW_SEVERITY
].sort_values(
    "severity",
    ascending=False
)

high_severity_csv = (
    OUTPUT_DIR
    / "high_severity_review_list.csv"
)

high_severity_df.to_csv(
    high_severity_csv,
    index=False
)


# ============================================================
# VERY LOW SEVERITY REVIEW LIST
# ============================================================

very_low_df = df[
    df["severity"] <= VERY_LOW_SEVERITY
].copy()

very_low_csv = (
    OUTPUT_DIR
    / "very_low_severity_list.csv"
)

very_low_df.to_csv(
    very_low_csv,
    index=False
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "input_rows": int(len(df)),

    "high_confidence": int(
        status_counts["HIGH_CONFIDENCE"]
    ),

    "review": int(
        status_counts["REVIEW"]
    ),

    "reject": int(
        status_counts["REJECT"]
    ),

    "clean_training_rows": int(
        len(clean_df)
    ),

    "clean_percentage": float(
        len(clean_df) / len(df) * 100
    ),

    "severity_all": {
        "mean": float(
            df["severity"].mean()
        ),
        "median": float(
            df["severity"].median()
        ),
        "min": float(
            df["severity"].min()
        ),
        "max": float(
            df["severity"].max()
        ),
    },

    "severity_clean": {
        "mean": float(
            clean_df["severity"].mean()
        )
        if len(clean_df)
        else None,

        "median": float(
            clean_df["severity"].median()
        )
        if len(clean_df)
        else None,

        "min": float(
            clean_df["severity"].min()
        )
        if len(clean_df)
        else None,

        "max": float(
            clean_df["severity"].max()
        )
        if len(clean_df)
        else None,
    },

    "note": (
        "SAM2 outputs are automatically generated "
        "pseudo-labels and are not independently "
        "verified ground truth."
    )
}

summary_json = (
    OUTPUT_DIR
    / "qc_summary.json"
)

with open(
    summary_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("QC COMPLETE")
print("=" * 80)

print(
    f"\nHIGH_CONFIDENCE : "
    f"{status_counts['HIGH_CONFIDENCE']}"
)

print(
    f"REVIEW          : "
    f"{status_counts['REVIEW']}"
)

print(
    f"REJECT          : "
    f"{status_counts['REJECT']}"
)

print(
    f"\nCLEAN TRAINING DATA: "
    f"{len(clean_df)} images"
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nImportant:")
print(
    "Do NOT delete the original SAM2 dataset."
)

print(
    "Do NOT train until the disease-wise QC "
    "statistics and suspicious examples are inspected."
)

print("\nFiles created:")

for p in [
    all_qc_csv,
    clean_csv,
    review_csv,
    reject_csv,
    disease_stats_csv,
    flag_csv,
    distribution_csv,
    worst_csv,
    high_severity_csv,
    very_low_csv,
    summary_json,
]:

    print(" -", p.name)

print("\nDone.")