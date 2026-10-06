from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path(r"C:\smart_agriculture")

SPLITS_FILE = (
    PROJECT
    / "ai"
    / "disease"
    / "dataset"
    / "splits.csv"
)

PLANTDOC_DIR = (
    PROJECT
    / "ai"
    / "disease"
    / "external_train"
)

OUTPUT_DIR = (
    PROJECT
    / "ai"
    / "disease"
    / "dataset"
    / "domain_adaptation"
)

SEED = 42
PLANTDOC_VAL_RATIO = 0.15

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. LOAD ORIGINAL PLANTVILLAGE SPLIT
# ============================================================

print("=" * 70)
print("LOADING PLANTVILLAGE DATASET")
print("=" * 70)

pv = pd.read_csv(SPLITS_FILE)

required_columns = {
    "path",
    "class_name",
    "label",
    "split",
}

missing_columns = (
    required_columns - set(pv.columns)
)

if missing_columns:
    raise ValueError(
        "Missing required columns in splits.csv: "
        f"{sorted(missing_columns)}"
    )

print(f"Total PlantVillage images : {len(pv)}")
print(
    "PlantVillage classes       : "
    f"{pv['class_name'].nunique()}"
)


# ============================================================
# 2. CREATE CLASS NAME -> NUMERIC LABEL MAPPING
# ============================================================

class_mapping = (
    pv[
        ["class_name", "label"]
    ]
    .drop_duplicates()
    .copy()
)

class_mapping["class_name"] = (
    class_mapping["class_name"].astype(str)
)

class_mapping["label"] = (
    class_mapping["label"].astype(int)
)

# Make sure each class has exactly one label
mapping_counts = (
    class_mapping
    .groupby("class_name")["label"]
    .nunique()
)

if (mapping_counts > 1).any():

    problematic = (
        mapping_counts[
            mapping_counts > 1
        ]
        .index
        .tolist()
    )

    raise ValueError(
        "Some PlantVillage classes have "
        "multiple numeric labels:\n"
        + "\n".join(problematic)
    )

class_name_to_label = dict(
    zip(
        class_mapping["class_name"],
        class_mapping["label"]
    )
)


print()
print("PlantVillage class mapping:")

for class_name, label in sorted(
    class_name_to_label.items(),
    key=lambda x: x[1]
):
    print(
        f"{label:2d} -> {class_name}"
    )


# ============================================================
# 3. SELECT PLANTVILLAGE TRAINING DATA
# ============================================================

pv_train = pv[
    pv["split"].astype(str).str.lower() == "train"
].copy()

print()
print(
    "PlantVillage training images : "
    f"{len(pv_train)}"
)


# ============================================================
# 4. SELECT PLANTVILLAGE VALIDATION DATA
# ============================================================

pv_val = pv[
    pv["split"].astype(str).str.lower() == "val"
].copy()

print(
    "PlantVillage validation images : "
    f"{len(pv_val)}"
)


# ============================================================
# 5. LOAD PLANTDOC TRAINING IMAGES
# ============================================================

print()
print("=" * 70)
print("LOADING PLANTDOC TRAINING DATA")
print("=" * 70)

plantdoc_rows = []

for class_dir in sorted(
    PLANTDOC_DIR.iterdir()
):

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    for image_path in sorted(
        class_dir.rglob("*")
    ):

        if not image_path.is_file():
            continue

        if (
            image_path.suffix.lower()
            not in VALID_EXTENSIONS
        ):
            continue

        plantdoc_rows.append(
            {
                "path": str(
                    image_path.resolve()
                ),
                "class_name": class_name,
                "source": "PlantDoc",
            }
        )


plantdoc = pd.DataFrame(
    plantdoc_rows
)

if plantdoc.empty:

    raise ValueError(
        "No PlantDoc training images were found."
    )

print(
    "PlantDoc training images : "
    f"{len(plantdoc)}"
)

print(
    "PlantDoc classes         : "
    f"{plantdoc['class_name'].nunique()}"
)


# ============================================================
# 6. VERIFY PLANTDOC CLASSES
# ============================================================

plantdoc_classes = set(
    plantdoc["class_name"].unique()
)

plantvillage_classes = set(
    class_name_to_label.keys()
)

unknown_classes = (
    plantdoc_classes
    - plantvillage_classes
)

if unknown_classes:

    print()
    print(
        "ERROR: Unknown PlantDoc classes:"
    )

    for name in sorted(
        unknown_classes
    ):
        print(
            f"  {name}"
        )

    raise ValueError(
        "PlantDoc contains classes that "
        "are not present in PlantVillage."
    )


print()
print(
    "All PlantDoc classes match "
    "PlantVillage classes."
)


# ============================================================
# 7. CONVERT PLANTDOC CLASS NAMES TO NUMERIC LABELS
# ============================================================

plantdoc["label"] = (
    plantdoc["class_name"]
    .map(class_name_to_label)
    .astype(int)
)

print()
print(
    "Successfully mapped "
    f"{len(plantdoc)} PlantDoc images "
    "to PlantVillage numeric labels."
)


# ============================================================
# 8. ADD LABELS / SOURCE TO PLANTVILLAGE
# ============================================================

pv_train["label"] = (
    pv_train["label"].astype(int)
)

pv_val["label"] = (
    pv_val["label"].astype(int)
)

pv_train["source"] = "PlantVillage"
pv_val["source"] = "PlantVillage"


# ============================================================
# 9. SPLIT PLANTDOC INTO TRAIN / VALIDATION
# ============================================================

print()
print(
    "Creating PlantDoc train/validation split..."
)

plantdoc_train, plantdoc_val = train_test_split(
    plantdoc,
    test_size=PLANTDOC_VAL_RATIO,
    random_state=SEED,
    stratify=plantdoc["label"],
)

plantdoc_train = plantdoc_train.copy()
plantdoc_val = plantdoc_val.copy()

plantdoc_train["split"] = "train"
plantdoc_val["split"] = "val"


# ============================================================
# 10. ADD SPLIT COLUMN TO PLANTVILLAGE
# ============================================================

pv_train["split"] = "train"
pv_val["split"] = "val"


# ============================================================
# 11. CREATE COMBINED TRAINING DATASET
# ============================================================

domain_train = pd.concat(
    [
        pv_train[
            [
                "path",
                "class_name",
                "label",
                "source",
                "split",
            ]
        ],

        plantdoc_train[
            [
                "path",
                "class_name",
                "label",
                "source",
                "split",
            ]
        ],
    ],
    ignore_index=True,
)


# ============================================================
# 12. CREATE COMBINED VALIDATION DATASET
# ============================================================

domain_val = pd.concat(
    [
        pv_val[
            [
                "path",
                "class_name",
                "label",
                "source",
                "split",
            ]
        ],

        plantdoc_val[
            [
                "path",
                "class_name",
                "label",
                "source",
                "split",
            ]
        ],
    ],
    ignore_index=True,
)


# ============================================================
# 13. SAVE DATASETS
# ============================================================

train_file = (
    OUTPUT_DIR
    / "domain_train.csv"
)

val_file = (
    OUTPUT_DIR
    / "domain_val.csv"
)

summary_file = (
    OUTPUT_DIR
    / "dataset_summary.csv"
)

domain_train.to_csv(
    train_file,
    index=False
)

domain_val.to_csv(
    val_file,
    index=False
)


# ============================================================
# 14. CREATE DATASET SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "dataset": [
            "PlantVillage train",
            "PlantDoc train",
            "Combined train",
            "PlantVillage validation",
            "PlantDoc validation",
            "Combined validation",
        ],

        "images": [
            len(pv_train),
            len(plantdoc_train),
            len(domain_train),
            len(pv_val),
            len(plantdoc_val),
            len(domain_val),
        ],
    }
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 15. PRINT FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("DOMAIN ADAPTATION DATASET")
print("=" * 70)

print()
print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# SOURCE DISTRIBUTION
# ============================================================

print()
print("Training images by source:")

print(
    domain_train["source"]
    .value_counts()
    .to_string()
)

print()
print("Validation images by source:")

print(
    domain_val["source"]
    .value_counts()
    .to_string()
)


# ============================================================
# PLANTDOC CLASS DISTRIBUTION
# ============================================================

print()
print("PlantDoc training images by class:")

print(
    plantdoc_train[
        "class_name"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# FINAL FILE LOCATIONS
# ============================================================

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    f"Training : {train_file}"
)

print(
    f"Validation : {val_file}"
)

print(
    f"Summary : {summary_file}"
)


# ============================================================
# SAFETY CHECK
# ============================================================

print()
print("=" * 70)
print("SAFETY CHECK")
print("=" * 70)

print(
    "PlantDoc external_test was NOT accessed."
)

print(
    "The 134-image external test set remains "
    "completely untouched."
)

print()
print("Dataset preparation completed successfully.")