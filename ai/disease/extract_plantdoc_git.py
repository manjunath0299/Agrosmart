import subprocess
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPO = PROJECT_ROOT / "PlantDoc-Dataset"
OUTPUT = PROJECT_ROOT / "ai" / "disease" / "external_test" / "images"

OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PLANTDOC DIRECT GIT EXTRACTION")
print("=" * 70)

print(f"Repository : {REPO}")
print(f"Output     : {OUTPUT}")
print()

if not REPO.exists():
    raise FileNotFoundError(f"Repository not found: {REPO}")


def safe_name(name):
    """
    Make a filename safe for Windows.
    """
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.rstrip(". ")
    return name


# ------------------------------------------------------------
# Get all files in test/ directly from the Git tree.
# -z is important because some filenames contain special chars.
# ------------------------------------------------------------

result = subprocess.run(
    [
        "git",
        "-C",
        str(REPO),
        "ls-tree",
        "-r",
        "-z",
        "HEAD",
        "test"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=True
)

entries = result.stdout.split(b"\x00")

files = []

for entry in entries:
    if not entry:
        continue

    # Format:
    # 100644 blob SHA<TAB>test/class/file.jpg
    try:
        metadata, path_bytes = entry.split(b"\t", 1)
        parts = metadata.split()

        if len(parts) != 3:
            continue

        mode, object_type, sha = parts

        if object_type != b"blob":
            continue

        path = path_bytes.decode("utf-8", errors="replace")

        if not path.startswith("test/"):
            continue

        path_parts = path.split("/")

        if len(path_parts) < 3:
            continue

        class_name = path_parts[1]
        filename = "/".join(path_parts[2:])

        extension = Path(filename).suffix.lower()

        if extension not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        }:
            continue

        files.append(
            {
                "sha": sha.decode(),
                "class_name": class_name,
                "filename": filename
            }
        )

    except Exception as e:
        print("Could not parse entry:", e)


print(f"Image files found in Git test tree: {len(files)}")
print()

# ------------------------------------------------------------
# Extract each Git blob.
#
# We use `git cat-file blob SHA`.
# This completely bypasses Windows filename restrictions.
# ------------------------------------------------------------

extracted = 0
failed = 0

for index, item in enumerate(files, start=1):

    class_name = safe_name(item["class_name"])
    original_filename = Path(item["filename"]).name
    filename = safe_name(original_filename)

    output_dir = OUTPUT / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / filename

    # Avoid collisions after sanitizing names.
    if output_file.exists():
        stem = output_file.stem
        suffix = output_file.suffix

        counter = 1

        while output_file.exists():
            output_file = (
                output_dir /
                f"{stem}_{counter}{suffix}"
            )
            counter += 1

    try:

        with open(output_file, "wb") as f:

            process = subprocess.run(
                [
                    "git",
                    "-C",
                    str(REPO),
                    "cat-file",
                    "blob",
                    item["sha"]
                ],
                stdout=f,
                stderr=subprocess.PIPE
            )

        if process.returncode != 0:
            failed += 1
            print(
                f"[FAILED] {item['filename']}"
            )
            continue

        extracted += 1

        if index % 100 == 0:
            print(
                f"Progress: {index}/{len(files)} "
                f"| Extracted: {extracted} "
                f"| Failed: {failed}"
            )

    except Exception as e:
        failed += 1
        print(
            f"[FAILED] {item['filename']} "
            f"-> {e}"
        )


print()
print("=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)
print(f"Images found    : {len(files)}")
print(f"Images extracted: {extracted}")
print(f"Images failed   : {failed}")
print(f"Output          : {OUTPUT}")
print("=" * 70)