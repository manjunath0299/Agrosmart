from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# PlantVillage color dataset
DATASET_DIR = PROJECT_ROOT / "PlantVillage-Dataset" / "raw" / "color"

# Where we save the split information
OUTPUT_DIR = PROJECT_ROOT / "ai" / "disease" / "dataset"
OUTPUT_FILE = OUTPUT_DIR / "splits.csv"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# Image extensions
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
}


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found:\n{DATASET_DIR}"
    )

print("=" * 70)
print("PLANTVILLAGE DATASET PREPARATION")
print("=" * 70)

print(f"\nDataset directory:")
print(DATASET_DIR)


# ============================================================
# FIND CLASSES
# ============================================================

class_dirs = sorted([
    directory
    for directory in DATASET_DIR.iterdir()
    if directory.is_dir()
])

print(f"\nNumber of classes: {len(class_dirs)}")

if len(class_dirs) == 0:
    raise RuntimeError("No class folders found.")


# ============================================================
# COLLECT IMAGE PATHS
# ============================================================

records = []

print("\nCollecting images...")

for class_index, class_dir in enumerate(class_dirs):

    class_name = class_dir.name

    image_files = [
        file
        for file in class_dir.rglob("*")
        if file.is_file() and file.suffix in IMAGE_EXTENSIONS
    ]

    print(
        f"{class_index:02d} | "
        f"{class_name:<55} | "
        f"{len(image_files)} images"
    )

    for image_path in image_files:

        records.append({
            "path": str(image_path.resolve()),
            "class_name": class_name,
            "label": class_index
        })


df = pd.DataFrame(records)

print("\n" + "=" * 70)
print(f"Total images: {len(df)}")
print(f"Total classes: {df['class_name'].nunique()}")
print("=" * 70)


# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

class_counts = (
    df["class_name"]
    .value_counts()
    .sort_index()
)

print(class_counts.to_string())


# ============================================================
# TRAIN / TEMP SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=(VAL_RATIO + TEST_RATIO),
    stratify=df["label"],
    random_state=RANDOM_SEED
)


# ============================================================
# VALIDATION / TEST SPLIT
# ============================================================

# temp is 30% of the full dataset.
# We need 15% validation + 15% test.
# Therefore divide temp equally.

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=RANDOM_SEED
)


# ============================================================
# ADD SPLIT COLUMN
# ============================================================

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"


# ============================================================
# COMBINE
# ============================================================

final_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

# Shuffle rows
final_df = final_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET SPLIT COMPLETE")
print("=" * 70)

print(f"\nTrain images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"Test images: {len(test_df)}")
print(f"Total images: {len(final_df)}")

print("\nSplit percentages:")

print(
    f"Train: {len(train_df) / len(final_df) * 100:.2f}%"
)

print(
    f"Validation: {len(val_df) / len(final_df) * 100:.2f}%"
)

print(
    f"Test: {len(test_df) / len(final_df) * 100:.2f}%"
)

print("\nSaved split file:")
print(OUTPUT_FILE)

print("\nDone.")