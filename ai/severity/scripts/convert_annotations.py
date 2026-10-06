import json
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


BASE_DIR = Path(r"C:\smart_agriculture")

ANNOTATION_DIR = (
    BASE_DIR
    / "ai"
    / "severity"
    / "dataset"
    / "images"
    / "annotation_pool"
)

MASK_DIR = (
    BASE_DIR
    / "ai"
    / "severity"
    / "dataset"
    / "masks"
)

OVERLAY_DIR = (
    BASE_DIR
    / "ai"
    / "severity"
    / "dataset"
    / "overlays"
)

MASK_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)


def polygon_mask(width, height, shapes, label):
    """
    Create a binary mask for all polygons having the
    requested LabelMe label.
    """

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for shape in shapes:

        if shape.get("label", "").lower() != label:
            continue

        points = shape.get("points", [])

        if len(points) < 3:
            continue

        polygon = [
            (int(round(x)), int(round(y)))
            for x, y in points
        ]

        draw.polygon(polygon, fill=255)

    return mask


results = []

json_files = sorted(ANNOTATION_DIR.glob("*.json"))

print("=" * 60)
print("LabelMe Annotation Conversion")
print("=" * 60)
print(f"JSON files found: {len(json_files)}")
print()


for json_path in json_files:

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_width = data.get("imageWidth")
    image_height = data.get("imageHeight")

    if image_width is None or image_height is None:

        image_path = ANNOTATION_DIR / data["imagePath"]

        with Image.open(image_path) as im:
            image_width, image_height = im.size

    shapes = data.get("shapes", [])

    labels = {
        shape.get("label", "").lower()
        for shape in shapes
    }

    if "leaf" not in labels:
        print(f"WARNING: {json_path.name} has no leaf annotation")

    if "disease" not in labels:
        print(f"WARNING: {json_path.name} has no disease annotation")

    # Create masks
    leaf_mask = polygon_mask(
        image_width,
        image_height,
        shapes,
        "leaf",
    )

    disease_mask = polygon_mask(
        image_width,
        image_height,
        shapes,
        "disease",
    )

    # Convert to NumPy
    leaf_array = np.array(leaf_mask) > 0
    disease_array = np.array(disease_mask) > 0

    # Disease should only exist inside the leaf
    disease_array = disease_array & leaf_array

    leaf_pixels = int(leaf_array.sum())
    disease_pixels = int(disease_array.sum())

    if leaf_pixels == 0:
        severity = 0.0
    else:
        severity = (
            disease_pixels
            / leaf_pixels
            * 100.0
        )

    # Save masks
    stem = json_path.stem

    leaf_mask_path = MASK_DIR / f"{stem}_leaf.png"
    disease_mask_path = MASK_DIR / f"{stem}_disease.png"

    leaf_mask.save(leaf_mask_path)
    Image.fromarray(
        (disease_array * 255).astype(np.uint8)
    ).save(disease_mask_path)

    # --------------------------------------------------
    # Create verification overlay
    # --------------------------------------------------

    image_path = ANNOTATION_DIR / data["imagePath"]

    if not image_path.exists():

        # LabelMe may store only the filename
        image_path = ANNOTATION_DIR / Path(
            data["imagePath"]
        ).name

    if image_path.exists():

        image = Image.open(image_path).convert("RGB")

        overlay = image.copy()
        overlay_array = np.array(overlay)

        # Leaf boundary visualization
        # Disease regions are highlighted.
        disease_pixels_mask = disease_array

        overlay_array[disease_pixels_mask] = [
            255,
            0,
            0,
        ]

        overlay = Image.fromarray(overlay_array)

        overlay_path = (
            OVERLAY_DIR
            / f"{stem}_overlay.png"
        )

        overlay.save(overlay_path)

    results.append(
        {
            "annotation_file": json_path.name,
            "image_file": data.get("imagePath", ""),
            "leaf_pixels": leaf_pixels,
            "disease_pixels": disease_pixels,
            "severity_percent": round(severity, 2),
            "leaf_mask": str(leaf_mask_path),
            "disease_mask": str(disease_mask_path),
        }
    )

    print(
        f"{json_path.name[:45]:45s} "
        f"Severity: {severity:6.2f}%"
    )


# ------------------------------------------------------
# Save CSV
# ------------------------------------------------------

csv_path = (
    BASE_DIR
    / "ai"
    / "severity"
    / "dataset"
    / "severity_labels.csv"
)

with open(
    csv_path,
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "annotation_file",
            "image_file",
            "leaf_pixels",
            "disease_pixels",
            "severity_percent",
            "leaf_mask",
            "disease_mask",
        ],
    )

    writer.writeheader()
    writer.writerows(results)


print()
print("=" * 60)
print("Conversion complete")
print("=" * 60)

print(f"Annotations: {len(results)}")
print(f"CSV:         {csv_path}")
print(f"Masks:       {MASK_DIR}")
print(f"Overlays:    {OVERLAY_DIR}")