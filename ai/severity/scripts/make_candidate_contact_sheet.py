from pathlib import Path
import cv2
import pandas as pd
import math


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CSV_PATH = (
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "contact_sheet.jpg"


# ============================================================
# SETTINGS
# ============================================================

THUMB_W = 220
THUMB_H = 220

LABEL_H = 65

COLS = 5

MARGIN = 15

BACKGROUND = (35, 35, 35)


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("CANDIDATE CONTACT SHEET")
print("=" * 70)

print(f"Candidates: {len(df)}")


# ============================================================
# LOAD IMAGES
# ============================================================

items = []

for idx, row in df.iterrows():

    image_path = Path(row["path"])

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"WARNING: Could not read {image_path}")
        continue

    image = cv2.resize(
        image,
        (THUMB_W, THUMB_H),
        interpolation=cv2.INTER_AREA
    )

    items.append({
        "number": idx + 1,
        "image": image,
        "filename": row["image"],
        "score": float(row["triage_score"]),
        "selection": row["selection"],
    })


# ============================================================
# CREATE SHEET
# ============================================================

ROWS = math.ceil(len(items) / COLS)

CELL_W = THUMB_W + MARGIN
CELL_H = THUMB_H + LABEL_H + MARGIN

sheet_w = COLS * CELL_W + MARGIN
sheet_h = ROWS * CELL_H + MARGIN

sheet = (
    cv2
    .Mat
    if False
    else None
)

sheet = __import__("numpy").full(
    (sheet_h, sheet_w, 3),
    BACKGROUND,
    dtype="uint8"
)


# ============================================================
# DRAW ITEMS
# ============================================================

for i, item in enumerate(items):

    row = i // COLS
    col = i % COLS

    x = MARGIN + col * CELL_W
    y = MARGIN + row * CELL_H

    # Image
    sheet[
        y:y + THUMB_H,
        x:x + THUMB_W
    ] = item["image"]

    # Candidate number
    cv2.putText(
        sheet,
        f"#{item['number']}",
        (x + 5, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Selection type
    cv2.putText(
        sheet,
        str(item["selection"]),
        (x + 5, y + THUMB_H + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # Score
    cv2.putText(
        sheet,
        f"score: {item['score']:.1f}",
        (x + 5, y + THUMB_H + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


# ============================================================
# SAVE
# ============================================================

cv2.imwrite(
    str(OUTPUT_PATH),
    sheet,
    [cv2.IMWRITE_JPEG_QUALITY, 95]
)

print()
print(f"Created:")
print(OUTPUT_PATH)

print()
print("=" * 70)