from pathlib import Path
from PIL import Image

ROOT = Path(r"C:\smart_agriculture\ai\disease\external_train")

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

files = [
    p for p in ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in EXTENSIONS
]

good = 0
bad = 0

for path in files:
    try:
        with Image.open(path) as img:
            img.verify()
        good += 1

    except Exception as e:
        bad += 1
        print(f"BAD: {path}")
        print(f"     {e}")

print()
print("==============================")
print("IMAGE VALIDATION")
print("==============================")
print(f"Total : {len(files)}")
print(f"Good  : {good}")
print(f"Bad   : {bad}")