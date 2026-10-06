from pathlib import Path
import random
import shutil
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"E:\smart_agriculture\smart_agriculture")

PV_ROOT = (
    ROOT
    / "PlantVillage-Dataset"
    / "raw"
    / "color"
)

OUT_ROOT = (
    ROOT
    / "ai"
    / "severity"
    / "dataset"
    / "general_3000"
)

IMAGE_DIR = OUT_ROOT / "images"

METADATA_PATH = OUT_ROOT / "metadata.csv"

SEED = 42

IMAGES_PER_CLASS = 250


# ============================================================
# SELECTED DISEASE CLASSES
# ============================================================

selected_classes = {

    "Tomato": [
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
    ],

    "Potato": [
        "Potato___Early_blight",
        "Potato___Late_blight",
    ],

    "Corn": [
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
    ],

    "Apple": [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
    ],
}


# ============================================================
# SETUP
# ============================================================

random.seed(SEED)

IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
}


records = []


print("=" * 70)
print("GENERAL SEVERITY DATASET BUILDER")
print("=" * 70)


# ============================================================
# COLLECT IMAGES
# ============================================================

for plant, disease_classes in selected_classes.items():

    print(f"\n{plant}")
    print("-" * 60)

    for disease_class in disease_classes:

        source_dir = PV_ROOT / disease_class

        if not source_dir.exists():

            print(
                f"WARNING: missing folder: {disease_class}"
            )

            continue

        files = [
            p
            for p in source_dir.iterdir()
            if p.is_file()
            and p.suffix in extensions
        ]

        files = sorted(files)

        random.shuffle(files)

        selected = files[
            :min(
                IMAGES_PER_CLASS,
                len(files)
            )
        ]

        print(
            f"{disease_class}: "
            f"{len(files)} available -> "
            f"{len(selected)} selected"
        )


        # ----------------------------------------------------
        # Copy images
        # ----------------------------------------------------

        safe_name = (
            disease_class
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        class_output = IMAGE_DIR / safe_name

        class_output.mkdir(
            parents=True,
            exist_ok=True
        )


        for index, src in enumerate(selected):

            # Preserve original extension
            dst_name = (
                f"{safe_name}"
                f"_{index:04d}"
                f"{src.suffix.lower()}"
            )

            dst = class_output / dst_name

            shutil.copy2(
                src,
                dst
            )

            records.append({
                "image_path": str(dst),
                "filename": dst.name,
                "plant": plant,
                "disease": disease_class,
                "source_path": str(src),
                "severity": None,
                "severity_source": None,
            })


# ============================================================
# SAVE METADATA
# ============================================================

df = pd.DataFrame(records)

df.to_csv(
    METADATA_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("DATASET CREATED")
print("=" * 70)

print(
    f"Total images: {len(df)}"
)

print(
    f"Metadata: {METADATA_PATH}"
)

print("\nImages by plant:")

print(
    df.groupby("plant")
      .size()
      .to_string()
)

print("\nImages by disease:")

print(
    df.groupby("disease")
      .size()
      .to_string()
)

print("\nDone.")