from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_CSV = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "next_annotation_candidates"
    / "selected_next_candidates.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "next_annotation_candidates"
)

OUTPUT_CSV = OUTPUT_DIR / "selected_60_for_annotation.csv"


# Candidate numbers from the contact sheet
selected_numbers = [
    # High
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 12, 13, 14, 16,
    17, 19, 21, 22, 23,

    # Medium
    31, 32, 33, 35, 36,
    37, 38, 39, 42, 44,
    45, 46, 48, 49, 50,
    51, 52, 53, 54, 55,

    # Diversity / lower
    61, 62, 63, 64, 65,
    66, 67, 68, 69, 70,
    71, 72, 73, 74, 75,
    76, 77, 78, 79, 80,
]


df = pd.read_csv(INPUT_CSV)

# The contact-sheet number corresponds to the CSV row + 1.
df["candidate_number"] = range(1, len(df) + 1)

selected = df[
    df["candidate_number"].isin(selected_numbers)
].copy()

selected["annotation_order"] = range(
    1,
    len(selected) + 1
)

selected.to_csv(
    OUTPUT_CSV,
    index=False
)

print("=" * 70)
print("60-IMAGE ANNOTATION LIST")
print("=" * 70)

print(f"Total selected: {len(selected)}")
print()

if len(selected) != 60:
    print("WARNING: Expected exactly 60 images.")
else:
    print("Exactly 60 images selected.")

print()
print("Output:")
print(OUTPUT_CSV)

print()
print("Selected candidates:")
print(
    selected[
        [
            "annotation_order",
            "candidate_number",
            "image",
            "selection",
            "triage_score",
        ]
    ].to_string(index=False)
)

print()
print("=" * 70)