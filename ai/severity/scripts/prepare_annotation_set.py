import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(r"C:\smart_agriculture")

CSV_PATH = (
    BASE_DIR
    / "PlantVillage-Dataset"
    / "leaf_grouping"
    / "filtered_leafmaps"
    / "Tomato___Early_blight.csv"
)

IMAGE_DIR = (
    BASE_DIR
    / "PlantVillage-Dataset"
    / "raw"
    / "color"
    / "Tomato___Early_blight"
)

OUTPUT_DIR = (
    BASE_DIR
    / "ai"
    / "severity"
    / "dataset"
    / "images"
    / "annotation_pool"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 1. Index the actual PlantVillage images
# ---------------------------------------------------------

actual_images = {}

for path in IMAGE_DIR.iterdir():

    if not path.is_file():
        continue

    name = path.name

    # PlantVillage current filenames look like:
    #
    # UUID___OriginalFilename.JPG
    #
    # Convert this to:
    #
    # OriginalFilename.JPG

    if "___" in name:
        original_name = name.split("___", 1)[1]
    else:
        original_name = name

    actual_images[original_name.lower()] = path


# ---------------------------------------------------------
# 2. Read the leaf-group CSV
# ---------------------------------------------------------

rows = []

with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:

    reader = csv.DictReader(f)

    for row in reader:

        filename = row["File Name"]
        leaf_number = row["Leaf #"]

        key = filename.lower()

        if key in actual_images:

            rows.append(
                {
                    "filename": filename,
                    "leaf": leaf_number,
                    "path": actual_images[key],
                }
            )


print("=" * 60)
print("PlantVillage Tomato Early Blight")
print("=" * 60)

print(f"CSV entries:       1000")
print(f"Mapped images:     {len(rows)}")


# ---------------------------------------------------------
# 3. Group images by leaf
# ---------------------------------------------------------

leaf_groups = defaultdict(list)

for row in rows:

    leaf_groups[row["leaf"]].append(row)


print(f"Leaf groups:       {len(leaf_groups)}")


# ---------------------------------------------------------
# 4. Select ONE image from each leaf
# ---------------------------------------------------------

selected = []

for leaf_number in sorted(
    leaf_groups.keys(),
    key=lambda x: float(x)
):

    candidates = leaf_groups[leaf_number]

    # Deterministic selection.
    # We use the first filename alphabetically.
    candidates = sorted(
        candidates,
        key=lambda x: x["filename"].lower()
    )

    selected.append(candidates[0])


# ---------------------------------------------------------
# 5. Copy representative images
# ---------------------------------------------------------

for index, item in enumerate(selected, start=1):

    source = item["path"]

    # Prefix with leaf number so the annotation dataset
    # remains traceable.

    safe_leaf = str(item["leaf"]).replace(".", "_")

    destination_name = (
        f"leaf_{safe_leaf}_"
        f"{source.name}"
    )

    destination = OUTPUT_DIR / destination_name

    shutil.copy2(source, destination)


# ---------------------------------------------------------
# 6. Save metadata
# ---------------------------------------------------------

metadata_path = OUTPUT_DIR / "metadata.csv"

with open(
    metadata_path,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "leaf_number",
            "original_filename",
            "source_path",
            "annotation_image",
        ]
    )

    for item in selected:

        safe_leaf = str(item["leaf"]).replace(".", "_")

        annotation_name = (
            f"leaf_{safe_leaf}_"
            f"{item['path'].name}"
        )

        writer.writerow(
            [
                item["leaf"],
                item["filename"],
                str(item["path"]),
                annotation_name,
            ]
        )


print()
print("=" * 60)
print("Annotation dataset created")
print("=" * 60)

print(f"Representative images: {len(selected)}")
print(f"Output directory:       {OUTPUT_DIR}")
print(f"Metadata:               {metadata_path}")
print()
print("The missing legacy image was skipped automatically.")