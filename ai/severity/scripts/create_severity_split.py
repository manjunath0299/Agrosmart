from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "severity_109_v3"
    / "severity_labels_109_v3.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "dataset"
    / "split_109_v3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SEVERITY DATASET SPLIT")
print("=" * 70)

df = pd.read_csv(INPUT_CSV)

print()
print(f"Total images: {len(df)}")


# ============================================================
# CHECK REQUIRED COLUMN
# ============================================================

if "severity_percent" not in df.columns:
    raise RuntimeError(
        "severity_percent column not found."
    )


# ============================================================
# CREATE SEVERITY STRATA
# ============================================================
#
# We use broad severity bins rather than exact percentages.
#
# 0  - <5      Very Low
# 5  - <10     Low
# 10 - <20     Moderate
# 20 - <40     High
# >=40         Very High
#
# These are ONLY used for balanced splitting.
# They are NOT the final treatment classes.
# ============================================================

bins = [
    -np.inf,
    5,
    10,
    20,
    40,
    np.inf
]

labels = [
    "very_low",
    "low",
    "moderate",
    "high",
    "very_high"
]

df["severity_bin"] = pd.cut(
    df["severity_percent"],
    bins=bins,
    labels=labels,
    right=False
)


# ============================================================
# PRINT DISTRIBUTION
# ============================================================

print()
print("Overall severity distribution:")
print()

print(
    df["severity_bin"]
    .value_counts()
    .sort_index()
)


# ============================================================
# FIRST SPLIT
# TRAIN = 70%
# TEMP  = 30%
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=(1 - TRAIN_RATIO),
    random_state=RANDOM_SEED,
    stratify=df["severity_bin"]
)


# ============================================================
# SECOND SPLIT
#
# TEMP = 30%
#
# Half validation
# Half test
#
# Therefore:
# validation ≈ 15%
# test       ≈ 15%
# ============================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_df["severity_bin"]
)


# ============================================================
# REMOVE STRATIFICATION COLUMN
# ============================================================

train_df = train_df.drop(
    columns=["severity_bin"]
)

val_df = val_df.drop(
    columns=["severity_bin"]
)

test_df = test_df.drop(
    columns=["severity_bin"]
)


# ============================================================
# SAVE SPLITS
# ============================================================

train_path = (
    OUTPUT_DIR
    / "train.csv"
)

val_path = (
    OUTPUT_DIR
    / "val.csv"
)

test_path = (
    OUTPUT_DIR
    / "test.csv"
)

train_df.to_csv(
    train_path,
    index=False
)

val_df.to_csv(
    val_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def print_summary(
    name,
    data
):

    print()
    print(
        "-" * 60
    )

    print(
        f"{name}: {len(data)} images"
    )

    print(
        f"Mean severity: "
        f"{data['severity_percent'].mean():.2f}%"
    )

    print(
        f"Median severity: "
        f"{data['severity_percent'].median():.2f}%"
    )

    print(
        f"Min severity: "
        f"{data['severity_percent'].min():.2f}%"
    )

    print(
        f"Max severity: "
        f"{data['severity_percent'].max():.2f}%"
    )

    print()

    temp_bins = pd.cut(
        data["severity_percent"],
        bins=bins,
        labels=labels,
        right=False
    )

    print(
        temp_bins.value_counts()
        .sort_index()
    )


# ============================================================
# PRINT SUMMARIES
# ============================================================

print_summary(
    "TRAIN",
    train_df
)

print_summary(
    "VALIDATION",
    val_df
)

print_summary(
    "TEST",
    test_df
)


# ============================================================
# CHECK DUPLICATES
# ============================================================

train_images = set(
    train_df["image"]
)

val_images = set(
    val_df["image"]
)

test_images = set(
    test_df["image"]
)

train_val_overlap = (
    train_images
    & val_images
)

train_test_overlap = (
    train_images
    & test_images
)

val_test_overlap = (
    val_images
    & test_images
)


print()
print("=" * 70)
print("OVERLAP CHECK")
print("=" * 70)

print(
    f"Train ∩ Validation: {len(train_val_overlap)}"
)

print(
    f"Train ∩ Test:       {len(train_test_overlap)}"
)

print(
    f"Validation ∩ Test:  {len(val_test_overlap)}"
)


if (
    train_val_overlap
    or train_test_overlap
    or val_test_overlap
):

    raise RuntimeError(
        "DATA LEAKAGE DETECTED!"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print()
print(
    f"Train CSV:      {train_path}"
)

print(
    f"Validation CSV: {val_path}"
)

print(
    f"Test CSV:       {test_path}"
)

print()
print(
    "No image overlap detected."
)

print()
print("=" * 70)