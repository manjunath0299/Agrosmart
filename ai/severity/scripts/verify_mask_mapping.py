import os
import pandas as pd
from PIL import Image

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

CSV_PATH = (
    r"ai\severity\results\general_severity_sam2_3000_qc"
    r"\severity_labels_sam2_3000_clean.csv"
)

MASK_DIR = (
    r"ai\severity\results\general_severity_sam2_3000"
    r"\masks"
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

mask_files = [
    f for f in os.listdir(MASK_DIR)
    if f.lower().endswith(".png")
]

print("=" * 60)
print("MASK MAPPING VERIFICATION")
print("=" * 60)

print(f"Clean CSV rows : {len(df)}")
print(f"Mask files     : {len(mask_files)}")


# ------------------------------------------------------------
# CREATE LOOKUP
# ------------------------------------------------------------

mask_lookup = {}

for filename in mask_files:

    stem = os.path.splitext(filename)[0].lower()

    mask_lookup[stem] = filename


# ------------------------------------------------------------
# MATCH IMAGES TO MASKS
# ------------------------------------------------------------

matched = []

missing = []

ambiguous = []


for _, row in df.iterrows():

    image_filename = str(
        row["filename"]
    )

    image_stem = os.path.splitext(
        image_filename
    )[0].lower()

    # Exact ending match.
    # SAM2 filenames contain a normalized disease prefix,
    # followed by the original image filename.
    candidates = [
        filename
        for filename in mask_files
        if filename.lower().endswith(
            image_stem + ".png"
        )
    ]

    if len(candidates) == 1:

        matched.append({
            "image_filename":
                image_filename,

            "mask_filename":
                candidates[0]
        })

    elif len(candidates) == 0:

        missing.append(
            image_filename
        )

    else:

        ambiguous.append({
            "image_filename":
                image_filename,

            "candidates":
                candidates
        })


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print()
print("=" * 60)
print("MATCHING RESULTS")
print("=" * 60)

print(
    f"Exact matches       : {len(matched)}"
)

print(
    f"Missing masks       : {len(missing)}"
)

print(
    f"Ambiguous matches   : {len(ambiguous)}"
)


# ------------------------------------------------------------
# SHOW PROBLEMS
# ------------------------------------------------------------

if missing:

    print()
    print("First missing masks:")

    for filename in missing[:20]:

        print(
            f"  {filename}"
        )


if ambiguous:

    print()
    print("First ambiguous matches:")

    for item in ambiguous[:10]:

        print(
            f"\nImage: {item['image_filename']}"
        )

        for candidate in item["candidates"]:

            print(
                f"  -> {candidate}"
            )


# ------------------------------------------------------------
# INSPECT FIRST MATCHES
# ------------------------------------------------------------

print()
print("=" * 60)
print("FIRST 10 IMAGE → MASK MAPPINGS")
print("=" * 60)

for item in matched[:10]:

    print(
        f"{item['image_filename']}"
        f"  ->  "
        f"{item['mask_filename']}"
    )


# ------------------------------------------------------------
# INSPECT MASKS
# ------------------------------------------------------------

print()
print("=" * 60)
print("MASK INSPECTION")
print("=" * 60)

shapes = set()

values = set()

invalid_masks = []


for item in matched:

    mask_path = os.path.join(
        MASK_DIR,
        item["mask_filename"]
    )

    try:

        mask = Image.open(
            mask_path
        )

        shapes.add(
            mask.size
        )

        mask_values = set(
            mask.getdata()
        )

        values.update(
            mask_values
        )

    except Exception as e:

        invalid_masks.append({
            "file":
                item["mask_filename"],

            "error":
                str(e)
        })


print(
    f"Mask shapes: {shapes}"
)

print(
    f"Unique mask pixel values: "
    f"{sorted(values)}"
)

print(
    f"Invalid masks: "
    f"{len(invalid_masks)}"
)


# ------------------------------------------------------------
# SAVE MAPPING
# ------------------------------------------------------------

mapping_df = pd.DataFrame(
    matched
)

mapping_path = (
    r"ai\severity\results"
    r"\general_severity_sam2_3000_qc"
    r"\clean_image_mask_mapping.csv"
)

mapping_df.to_csv(
    mapping_path,
    index=False
)


print()
print(
    f"Mapping saved to:"
)

print(
    mapping_path
)


# ------------------------------------------------------------
# FINAL STATUS
# ------------------------------------------------------------

print()
print("=" * 60)

if (
    len(matched) == len(df)
    and len(missing) == 0
    and len(ambiguous) == 0
    and len(invalid_masks) == 0
):

    print(
        "STATUS: PASS"
    )

    print(
        "All clean images have exactly "
        "one valid mask."
    )

else:

    print(
        "STATUS: CHECK REQUIRED"
    )

print("=" * 60)