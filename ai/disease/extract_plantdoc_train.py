from pathlib import Path
import subprocess
import json
import re

REPO = Path(r"C:\smart_agriculture\PlantDoc-Dataset")
OUTPUT = Path(r"C:\smart_agriculture\ai\disease\external_train")

OUTPUT.mkdir(parents=True, exist_ok=True)

CLASS_MAPPING = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "grape leaf black rot": "Grape___Black_rot",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
}

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def sanitize_filename(name):
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.rstrip(". ")
    return name


def get_train_entries():

    print("Reading Git tree...")

    result = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD"],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    lines = result.stdout.decode(
        "utf-8",
        errors="replace"
    ).splitlines()

    entries = []

    for line in lines:

        # Example:
        # 100644 blob SHA    train/Apple Scab Leaf/image.jpg

        parts = line.split(maxsplit=3)

        if len(parts) != 4:
            continue

        mode, obj_type, sha, path = parts

        if obj_type != "blob":
            continue

        if not path.startswith("train/"):
            continue

        entries.append((sha, path))

    return entries


def get_blob(sha):

    result = subprocess.run(
        ["git", "cat-file", "blob", sha],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    return result.stdout


def main():

    entries = get_train_entries()

    print(f"Total Git train entries: {len(entries)}")

    extracted = 0
    skipped = 0
    failed = 0

    class_counts = {}

    for sha, relative_path in entries:

        parts = relative_path.split("/", 2)

        if len(parts) != 3:
            skipped += 1
            continue

        _, class_name, filename = parts

        # Only use our unambiguous classes
        if class_name not in CLASS_MAPPING:
            skipped += 1
            continue

        suffix = Path(filename).suffix.lower()

        if suffix not in VALID_EXTENSIONS:
            skipped += 1
            continue

        target_class = CLASS_MAPPING[class_name]

        target_dir = OUTPUT / target_class
        target_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        safe_name = sanitize_filename(filename)

        if not safe_name:
            skipped += 1
            continue

        target_file = target_dir / safe_name

        # Prevent overwriting files
        if target_file.exists():

            stem = target_file.stem
            extension = target_file.suffix

            counter = 1

            while target_file.exists():

                target_file = (
                    target_dir
                    / f"{stem}_{counter}{extension}"
                )

                counter += 1

        try:

            data = get_blob(sha)

            target_file.write_bytes(data)

            extracted += 1

            class_counts[target_class] = (
                class_counts.get(target_class, 0) + 1
            )

        except Exception as e:

            failed += 1

            print(
                f"FAILED: {relative_path}"
            )

            print(e)

    # Save mapping
    mapping_file = OUTPUT / "class_mapping.json"

    with open(
        mapping_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            CLASS_MAPPING,
            f,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("==============================")
    print("PlantDoc TRAIN extraction")
    print("==============================")

    print(f"Extracted : {extracted}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")

    print()
    print("Class counts:")

    for class_name, count in sorted(
        class_counts.items()
    ):

        print(
            f"{class_name}: {count}"
        )


if __name__ == "__main__":
    main()