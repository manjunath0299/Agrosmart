from pathlib import Path

ROOT = Path(r"E:\smart_agriculture\smart_agriculture")
PV = ROOT / "PlantVillage-Dataset" / "raw" / "color"

plant_prefixes = {
    "Tomato": "Tomato___",
    "Potato": "Potato___",
    "Corn": "Corn_(maize)___",
    "Apple": "Apple___",
}

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
}

print("=" * 70)
print("PLANTVILLAGE DISEASE IMAGE COUNTS")
print("=" * 70)

for plant, prefix in plant_prefixes.items():

    print(f"\n{plant}")
    print("-" * 50)

    folders = sorted(
        p for p in PV.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    )

    total = 0

    for folder in folders:

        count = sum(
            1
            for f in folder.iterdir()
            if f.is_file()
            and f.suffix in image_extensions
        )

        total += count

        print(
            f"{folder.name}: {count} images"
        )

    print(f"TOTAL {plant}: {total}")


print("\nDone.")