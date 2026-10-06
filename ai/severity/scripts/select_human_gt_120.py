# ============================================================
# SELECT HUMAN GROUND-TRUTH DATASET
# ============================================================
#
# Purpose:
#   Select 10 representative images per disease
#   from the existing 2382 QC-clean dataset.
#
# Total:
#   12 diseases x 10 images = 120 images
#
# Selection strategy:
#   - Spread samples across the available severity range
#   - Avoid selecting only low/medium samples
#   - Prefer diverse severity values
#   - Deterministic / reproducible
#
# IMPORTANT:
#   The selected images will be manually annotated later.
#
# Human annotation:
#   Class 1 = leaf
#   Class 2 = disease
#
# ============================================================

import os
import shutil
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

ROOT = r"E:\smart_agriculture\smart_agriculture"

# Existing clean dataset
CLEAN_CSV = os.path.join(
    ROOT,
    r"ai\severity\results"
    r"\general_severity_sam2_3000_qc",
    "severity_labels_sam2_3000_clean.csv"
)

# Existing image-mask mapping
MAPPING_CSV = os.path.join(
    ROOT,
    r"ai\severity\results"
    r"\general_severity_sam2_3000_qc",
    "clean_image_mask_mapping.csv"
)

# Output
OUTPUT_DIR = os.path.join(
    ROOT,
    r"ai\severity\dataset\human_gt_120"
)

IMAGE_DIR = os.path.join(
    OUTPUT_DIR,
    "images"
)

ANNOTATION_DIR = os.path.join(
    OUTPUT_DIR,
    "annotations"
)

MASK_DIR = os.path.join(
    OUTPUT_DIR,
    "masks"
)

OVERLAY_DIR = os.path.join(
    OUTPUT_DIR,
    "overlays"
)

# Number of images per disease
IMAGES_PER_DISEASE = 10

# Reproducibility
SEED = 42

# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    IMAGE_DIR,
    exist_ok=True
)

os.makedirs(
    ANNOTATION_DIR,
    exist_ok=True
)

os.makedirs(
    MASK_DIR,
    exist_ok=True
)

os.makedirs(
    OVERLAY_DIR,
    exist_ok=True
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("HUMAN GROUND-TRUTH DATASET SELECTION")
print("=" * 70)

if not os.path.exists(CLEAN_CSV):

    raise FileNotFoundError(
        f"\nClean CSV not found:\n{CLEAN_CSV}"
    )

df = pd.read_csv(
    CLEAN_CSV
)

print(
    "\nClean dataset rows:",
    len(df)
)

# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "image_path",
    "filename",
    "plant",
    "disease",
    "severity"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise RuntimeError(
        "Missing required columns:\n"
        + str(missing_columns)
    )

# ============================================================
# REMOVE INVALID IMAGE PATHS
# ============================================================

df["image_exists"] = (
    df["image_path"]
    .apply(
        os.path.exists
    )
)

invalid_count = (
    (~df["image_exists"])
    .sum()
)

print(
    "Invalid image paths:",
    invalid_count
)

if invalid_count > 0:

    print(
        "\nFirst invalid paths:"
    )

    print(
        df.loc[
            ~df["image_exists"],
            "image_path"
        ].head(10).to_string(
            index=False
        )
    )

df = df[
    df["image_exists"]
].copy()

df.reset_index(
    drop=True,
    inplace=True
)

# ============================================================
# DISEASE LIST
# ============================================================

diseases = sorted(
    df["disease"]
    .unique()
)

print(
    "\nDiseases found:",
    len(diseases)
)

for i, disease in enumerate(
    diseases,
    start=1
):

    count = (
        df["disease"]
        .eq(disease)
        .sum()
    )

    print(
        f"{i:02d}. {disease} "
        f"({count} images)"
    )

# ============================================================
# CHECK EXPECTED 12 DISEASES
# ============================================================

if len(diseases) != 12:

    print(
        "\nWARNING:"
    )

    print(
        "Expected 12 diseases, "
        f"but found {len(diseases)}."
    )

    print(
        "The script will continue using "
        "all diseases found."
    )

# ============================================================
# SELECTION STRATEGY
# ============================================================
#
# We select images around evenly spaced severity quantiles.
#
# Example for 10 images:
#
#   0%
#   11%
#   22%
#   33%
#   44%
#   56%
#   67%
#   78%
#   89%
#   100%
#
# This gives coverage across the available severity range.
#
# We do NOT force >40 or >60 values because some diseases
# do not contain such samples.
#
# ============================================================

rng = np.random.default_rng(
    SEED
)

selected_rows = []

selection_records = []

print(
    "\n"
    + "=" * 70
)

print(
    "SELECTING REPRESENTATIVE IMAGES"
)

print(
    "=" * 70
)

for disease in diseases:

    disease_df = (
        df[
            df["disease"]
            == disease
        ]
        .copy()
        .reset_index(drop=True)
    )

    disease_df = disease_df.sort_values(
        "severity"
    ).reset_index(
        drop=True
    )

    n = len(
        disease_df
    )

    if n < IMAGES_PER_DISEASE:

        raise RuntimeError(
            f"Disease '{disease}' has only "
            f"{n} images."
        )

    # --------------------------------------------------------
    # Target quantile positions
    # --------------------------------------------------------

    quantiles = np.linspace(
        0.0,
        1.0,
        IMAGES_PER_DISEASE
    )

    selected_indices = []

    used_indices = set()

    for q in quantiles:

        target_position = (
            q * (n - 1)
        )

        # Search near target position
        center = int(
            round(
                target_position
            )
        )

        candidate_offsets = [
            0,
            -1,
            1,
            -2,
            2,
            -3,
            3,
            -4,
            4,
            -5,
            5
        ]

        selected_index = None

        for offset in candidate_offsets:

            candidate = (
                center
                + offset
            )

            if (
                candidate >= 0
                and candidate < n
                and candidate
                not in used_indices
            ):

                selected_index = candidate

                break

        if selected_index is None:

            # Random unused fallback
            available = [
                i
                for i in range(n)
                if i not in used_indices
            ]

            selected_index = int(
                rng.choice(
                    available
                )
            )

        used_indices.add(
            selected_index
        )

        selected_indices.append(
            selected_index
        )

    # --------------------------------------------------------
    # Save selected rows
    # --------------------------------------------------------

    selected = disease_df.iloc[
        selected_indices
    ].copy()

    selected = selected.sort_values(
        "severity"
    ).reset_index(
        drop=True
    )

    for rank, (_, row) in enumerate(
        selected.iterrows(),
        start=1
    ):

        selected_rows.append(
            row
        )

        selection_records.append(
            {
                "selection_rank":
                    rank,

                "disease":
                    row["disease"],

                "plant":
                    row["plant"],

                "original_filename":
                    row["filename"],

                "original_image_path":
                    row["image_path"],

                "sam2_severity":
                    float(
                        row["severity"]
                    )
            }
        )

    # --------------------------------------------------------
    # Print disease selection
    # --------------------------------------------------------

    print(
        f"\n{disease}"
    )

    print(
        "Available:",
        n
    )

    print(
        "Selected severities:"
    )

    print(
        [
            round(
                float(x),
                2
            )
            for x in selected[
                "severity"
            ].tolist()
        ]
    )

# ============================================================
# BUILD SELECTION DATAFRAME
# ============================================================

selected_df = pd.DataFrame(
    selected_rows
).reset_index(
    drop=True
)

selection_df = pd.DataFrame(
    selection_records
)

# ============================================================
# VERIFY COUNT
# ============================================================

expected_count = (
    len(diseases)
    *
    IMAGES_PER_DISEASE
)

actual_count = len(
    selected_df
)

print(
    "\n"
    + "=" * 70
)

print(
    "SELECTION SUMMARY"
)

print(
    "=" * 70
)

print(
    "Diseases:",
    len(diseases)
)

print(
    "Images per disease:",
    IMAGES_PER_DISEASE
)

print(
    "Expected images:",
    expected_count
)

print(
    "Selected images:",
    actual_count
)

if actual_count != expected_count:

    raise RuntimeError(
        "Selected image count is incorrect."
    )

# ============================================================
# COPY IMAGES
# ============================================================

print(
    "\nCopying selected images..."
)

metadata_rows = []

for global_index, row in selected_df.iterrows():

    disease = str(
        row["disease"]
    )

    plant = str(
        row["plant"]
    )

    original_filename = str(
        row["filename"]
    )

    original_path = str(
        row["image_path"]
    )

    # --------------------------------------------------------
    # Create safe disease folder/name
    # --------------------------------------------------------

    disease_safe = (
        disease
        .replace(
            " ",
            "_"
        )
        .replace(
            "/",
            "_"
        )
        .replace(
            "\\",
            "_"
        )
        .replace(
            ":",
            "_"
        )
    )

    original_ext = os.path.splitext(
        original_filename
    )[1]

    if not original_ext:

        original_ext = ".jpg"

    new_filename = (
        f"{global_index + 1:03d}_"
        f"{disease_safe}_"
        f"{original_filename}"
    )

    destination = os.path.join(
        IMAGE_DIR,
        new_filename
    )

    shutil.copy2(
        original_path,
        destination
    )

    # --------------------------------------------------------
    # Determine selection rank
    # --------------------------------------------------------

    disease_records = selection_df[
        selection_df["disease"]
        == disease
    ]

    matching_rank = disease_records[
        disease_records[
            "original_filename"
        ]
        == original_filename
    ]

    if len(matching_rank) > 0:

        selection_rank = int(
            matching_rank.iloc[0][
                "selection_rank"
            ]
        )

    else:

        selection_rank = -1

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_rows.append(
        {
            "id":
                global_index + 1,

            "new_filename":
                new_filename,

            "image_path":
                destination,

            "original_filename":
                original_filename,

            "original_image_path":
                original_path,

            "plant":
                plant,

            "disease":
                disease,

            "sam2_severity":
                float(
                    row["severity"]
                ),

            "selection_rank":
                selection_rank,

            "human_annotation_status":
                "NOT_ANNOTATED",

            "human_severity":
                np.nan,

            "human_leaf_mask":
                "",

            "human_disease_mask":
                ""
        }
    )

# ============================================================
# SAVE METADATA
# ============================================================

metadata_df = pd.DataFrame(
    metadata_rows
)

metadata_path = os.path.join(
    OUTPUT_DIR,
    "metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)

# ============================================================
# SAVE SELECTION DETAILS
# ============================================================

selection_path = os.path.join(
    OUTPUT_DIR,
    "selection_details.csv"
)

selection_df.to_csv(
    selection_path,
    index=False
)

# ============================================================
# VERIFY COPIED FILES
# ============================================================

copied_count = 0

missing_copied = []

for path in metadata_df[
    "image_path"
]:

    if os.path.exists(path):

        copied_count += 1

    else:

        missing_copied.append(
            path
        )

print(
    "\nCopied images:",
    copied_count
)

print(
    "Missing copied images:",
    len(missing_copied)
)

if missing_copied:

    print(
        "\nMissing:"
    )

    for path in missing_copied[:10]:

        print(
            path
        )

    raise RuntimeError(
        "Some selected images were not copied."
    )

# ============================================================
# FINAL DISEASE COUNTS
# ============================================================

print(
    "\nFinal disease counts:"
)

final_counts = (
    metadata_df[
        "disease"
    ]
    .value_counts()
    .sort_index()
)

print(
    final_counts.to_string()
)

# ============================================================
# FINAL SEVERITY TABLE
# ============================================================

print(
    "\nSelected severity values:"
)

severity_table = (
    metadata_df[
        [
            "disease",
            "sam2_severity"
        ]
    ]
    .groupby(
        "disease"
    )[
        "sam2_severity"
    ]
    .agg(
        [
            "min",
            "max",
            "mean"
        ]
    )
    .round(2)
)

print(
    severity_table.to_string()
)

# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SELECTION COMPLETE"
)

print(
    "=" * 70
)

print(
    "Total selected:",
    len(metadata_df)
)

print(
    "Image folder:"
)

print(
    IMAGE_DIR
)

print(
    "\nMetadata:"
)

print(
    metadata_path
)

print(
    "\nSelection details:"
)

print(
    selection_path
)

print(
    "\nNext step:"
)

print(
    "Open the images with LabelMe and annotate:"
)

print(
    "  1. leaf"
)

print(
    "  2. disease"
)

print(
    "\nDo NOT use the SAM2 mask as a guide while annotating."
)

print(
    "\nDone."
)

print(
    "=" * 70
)