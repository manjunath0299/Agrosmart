from pathlib import Path

from PIL import Image

from severity_engine import severity_engine


IMAGE = Path(
    r"E:\smart_agriculture\smart_agriculture\ai\severity\dataset\human_gt_120\images"
)

image_files = list(IMAGE.glob("*.jpg")) + list(IMAGE.glob("*.png"))

if not image_files:
    raise RuntimeError("No test images found.")

image_path = image_files[0]

print("=" * 60)
print("SEVERITY INFERENCE TEST")
print("=" * 60)
print("Image:", image_path)

image = Image.open(image_path)

result = severity_engine.predict(image)

print()
print("RESULT")
print("-" * 60)

for key, value in result.items():

    if key != "mask":
        print(f"{key}: {value}")

print("-" * 60)
print("Severity inference completed.")