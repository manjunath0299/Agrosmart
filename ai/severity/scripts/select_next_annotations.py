from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

POOL_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "dataset"
    / "images"
    / "annotation_pool"
)

SAM2_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "sam2_annotations"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "next_annotation_candidates"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TOP_HIGH = 30
TOP_MEDIUM = 30
RANDOM_DIVERSE = 20


# ============================================================
# HELPERS
# ============================================================

UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}"
)


def extract_uuid(filename):
    match = UUID_PATTERN.search(filename)

    if match:
        return match.group(0).lower()

    return None


def is_already_annotated(image_path):
    """
    Match by UUID because filenames contain UUIDs.
    """
    image_uuid = extract_uuid(image_path.name)

    if image_uuid is None:
        return False

    for mask_path in SAM2_DIR.glob("*_disease_mask.png"):
        if image_uuid in mask_path.name.lower():
            return True

    return False


def lesion_candidate_score(image_path):
    """
    Rough triage score only.

    This is NOT a disease-severity label.

    Higher score means the image contains more pixels with
    reddish/brown/yellow-green characteristics inside the
    approximate leaf region.

    The result is only used to prioritize visual inspection.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        return np.nan

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Approximate leaf/background separation.
    # Green leaf pixels generally have H around 30-90 in OpenCV HSV.
    green = (
        (h >= 25)
        & (h <= 95)
        & (s >= 35)
        & (v >= 35)
    )

    # Approximate suspicious brown/red/yellow tissue.
    red_brown = (
        (
            ((h <= 15) | (h >= 165))
            & (s >= 50)
            & (v >= 35)
        )
        |
        (
            (h >= 8)
            & (h <= 30)
            & (s >= 45)
            & (v >= 30)
        )
    )

    leaf_pixels = np.sum(green)

    if leaf_pixels == 0:
        return np.nan

    suspicious_pixels = np.sum(
        red_brown & green
    )

    score = (
        suspicious_pixels
        / leaf_pixels
        * 100.0
    )

    return float(score)


# ============================================================
# FIND POOL
# ============================================================

extensions = {
    ".jpg",
    ".jpeg",
    ".png"
}

images = sorted(
    [
        p
        for p in POOL_DIR.iterdir()
        if p.suffix.lower() in extensions
    ],
    key=lambda p: p.name.lower()
)

print("=" * 70)
print("NEXT ANNOTATION CANDIDATE SELECTION")
print("=" * 70)

print(f"Annotation pool images: {len(images)}")


# ============================================================
# REMOVE ALREADY ANNOTATED
# ============================================================

remaining = []

for image_path in images:

    if not is_already_annotated(image_path):
        remaining.append(image_path)


print(f"Already annotated:      {len(images) - len(remaining)}")
print(f"Remaining candidates:   {len(remaining)}")
print()


# ============================================================
# SCORE REMAINING IMAGES
# ============================================================

records = []

for i, image_path in enumerate(remaining, start=1):

    score = lesion_candidate_score(image_path)

    records.append({
        "image": image_path.name,
        "path": str(image_path),
        "triage_score": score,
    })

    if i % 25 == 0:
        print(f"Processed {i}/{len(remaining)}")


df = pd.DataFrame(records)

df = df.dropna(
    subset=["triage_score"]
)

df = df.sort_values(
    "triage_score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# SAVE ALL RANKINGS
# ============================================================

all_path = RESULT_DIR / "all_remaining_ranked.csv"

df.to_csv(
    all_path,
    index=False
)


# ============================================================
# SELECT HIGH / MEDIUM / RANDOM
# ============================================================

high = df.head(TOP_HIGH).copy()

# Medium candidates from the middle of the ranking
start_medium = TOP_HIGH
end_medium = min(
    TOP_HIGH + TOP_MEDIUM,
    len(df)
)

medium = df.iloc[
    start_medium:end_medium
].copy()


# Random candidates from the remainder
remainder = df.iloc[
    end_medium:
].copy()

random_count = min(
    RANDOM_DIVERSE,
    len(remainder)
)

if random_count > 0:

    random_candidates = remainder.sample(
        n=random_count,
        random_state=42
    ).copy()

else:

    random_candidates = pd.DataFrame(
        columns=df.columns
    )


high["selection"] = "HIGH_TRIAGE"
medium["selection"] = "MEDIUM_TRIAGE"
random_candidates["selection"] = "RANDOM_DIVERSITY"


selected = pd.concat(
    [
        high,
        medium,
        random_candidates
    ],
    ignore_index=True
)


selected_path = (
    RESULT_DIR
    / "selected_next_candidates.csv"
)

selected.to_csv(
    selected_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SELECTION COMPLETE")
print("=" * 70)

print(f"Remaining images:       {len(df)}")
print(f"High candidates:        {len(high)}")
print(f"Medium candidates:      {len(medium)}")
print(f"Random candidates:      {len(random_candidates)}")
print(f"Total selected:         {len(selected)}")

print()
print("All ranked:")
print(all_path)

print()
print("Selected candidates:")
print(selected_path)

print()
print("Top 15 candidates:")
print(
    df[
        [
            "image",
            "triage_score"
        ]
    ].head(15).to_string(index=False)
)

print()
print("=" * 70)
print("IMPORTANT:")
print("Triage score is NOT disease severity.")
print("Use it only to prioritize visual inspection and SAM2 annotation.")
print("=" * 70)