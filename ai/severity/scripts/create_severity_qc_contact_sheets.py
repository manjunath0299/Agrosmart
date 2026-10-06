"""
Create visual contact sheets for inspecting SAM2-generated
severity pseudo-labels.

Purpose
-------
Create easy-to-review contact sheets showing:

    Original image
    SAM2 severity overlay
    Severity %
    QC status
    QC flags

The script DOES NOT:
    - rerun SAM2
    - modify masks
    - modify original labels
    - delete images
    - train a model

It only creates visual QC sheets.

Input
-----
ai/severity/results/general_severity_sam2_3000/
    severity_labels_sam2_3000.csv
    overlays/

ai/severity/results/general_severity_sam2_3000_qc/
    qc_all_3000.csv

Output
------
ai/severity/results/general_severity_visual_qc/
    all/
    high_severity/
    by_disease/
    review/

Each contact sheet contains:
    Original | SAM2 Overlay
    Disease
    Severity
    QC Status
    QC Score
    QC Flags
"""

from pathlib import Path
import math
import re

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "general_severity_sam2_3000"
)

QC_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "general_severity_sam2_3000_qc"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "general_severity_visual_qc"
)

OVERLAY_DIR = RESULTS_DIR / "overlays"

CSV_PATH = QC_DIR / "qc_all_3000.csv"


# ============================================================
# SETTINGS
# ============================================================

# Number of images per disease in the main contact sheet
SAMPLES_PER_DISEASE = 12

# Number of suspicious/high severity samples
HIGH_SEVERITY_SAMPLES = 30

# Number of review samples
REVIEW_SAMPLES = 30

# Images per contact-sheet page
IMAGES_PER_PAGE = 6

# Maximum image display size
IMAGE_WIDTH = 500
IMAGE_HEIGHT = 380

# Contact sheet layout
PAGE_COLUMNS = 2
PAGE_ROWS = 3

# Text area below each image pair
TEXT_HEIGHT = 145

# JPEG quality
JPEG_QUALITY = 95


# ============================================================
# FONT
# ============================================================

def get_font(size=22, bold=False):

    candidates = []

    if bold:
        candidates = [
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf",
        ]
    else:
        candidates = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
        ]

    for path in candidates:

        if Path(path).exists():

            try:
                return ImageFont.truetype(
                    path,
                    size
                )
            except Exception:
                pass

    return ImageFont.load_default()


FONT_TITLE = get_font(28, bold=True)
FONT_NORMAL = get_font(20)
FONT_SMALL = get_font(16)
FONT_SEVERITY = get_font(26, bold=True)


# ============================================================
# HELPERS
# ============================================================

def clean_filename(text):

    text = str(text)

    text = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        text
    )

    return text[:180]


def safe_float(value):

    try:
        if pd.isna(value):
            return np.nan

        return float(value)

    except Exception:

        return np.nan


def find_overlay(row):

    """
    Try to locate the SAM2 overlay image.

    Several possible naming conventions are supported.
    """

    filename = str(row["filename"])

    image_path = Path(
        str(row["image_path"])
    )

    candidates = [

        # Same filename
        OVERLAY_DIR / filename,

        # JPG/PNG alternatives
        OVERLAY_DIR / (
            Path(filename).stem + ".jpg"
        ),

        OVERLAY_DIR / (
            Path(filename).stem + ".png"
        ),

        OVERLAY_DIR / (
            Path(filename).stem + ".jpeg"
        ),

        # Recursive search
        OVERLAY_DIR / image_path.name,
    ]

    for candidate in candidates:

        if candidate.exists():

            return candidate

    # Recursive fallback
    matches = list(
        OVERLAY_DIR.rglob(
            image_path.name
        )
    )

    if matches:

        return matches[0]

    return None


def find_original(row):

    """
    Locate original image.

    Prefer image_path from the CSV.
    """

    path = Path(
        str(row["image_path"])
    )

    if path.exists():

        return path

    # Try source_path if available
    if "source_path" in row:

        source = Path(
            str(row["source_path"])
        )

        if source.exists():

            return source

    return None


def resize_keep_aspect(
    image,
    max_width,
    max_height
):

    image = image.copy()

    image.thumbnail(
        (
            max_width,
            max_height
        ),
        Image.Resampling.LANCZOS
    )

    return image


def add_image_to_box(
    canvas,
    image,
    x,
    y,
    box_width,
    box_height
):

    image = image.convert("RGB")

    image = resize_keep_aspect(
        image,
        box_width,
        box_height
    )

    paste_x = (
        x
        + (box_width - image.width) // 2
    )

    paste_y = (
        y
        + (box_height - image.height) // 2
    )

    canvas.paste(
        image,
        (
            paste_x,
            paste_y
        )
    )


def wrap_text(
    draw,
    text,
    font,
    max_width
):

    words = str(text).split()

    lines = []

    current = ""

    for word in words:

        candidate = (
            word
            if not current
            else current + " " + word
        )

        bbox = draw.textbbox(
            (0, 0),
            candidate,
            font=font
        )

        width = bbox[2] - bbox[0]

        if width <= max_width:

            current = candidate

        else:

            if current:

                lines.append(current)

            current = word

    if current:

        lines.append(current)

    return lines


# ============================================================
# CREATE ONE CARD
# ============================================================

def create_card(row):

    card_width = (
        IMAGE_WIDTH * 2
    )

    card_height = (
        IMAGE_HEIGHT
        + TEXT_HEIGHT
    )

    card = Image.new(
        "RGB",
        (
            card_width,
            card_height
        ),
        "white"
    )

    draw = ImageDraw.Draw(card)

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    original = find_original(row)

    if original is not None:

        try:

            image = Image.open(
                original
            ).convert("RGB")

            add_image_to_box(
                card,
                image,
                0,
                0,
                IMAGE_WIDTH,
                IMAGE_HEIGHT
            )

        except Exception:

            draw.text(
                (20, 30),
                "Original image error",
                fill="black",
                font=FONT_NORMAL
            )

    else:

        draw.text(
            (20, 30),
            "Original image not found",
            fill="black",
            font=FONT_NORMAL
        )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay_path = find_overlay(row)

    if overlay_path is not None:

        try:

            image = Image.open(
                overlay_path
            ).convert("RGB")

            add_image_to_box(
                card,
                image,
                IMAGE_WIDTH,
                0,
                IMAGE_WIDTH,
                IMAGE_HEIGHT
            )

        except Exception:

            draw.text(
                (
                    IMAGE_WIDTH + 20,
                    30
                ),
                "Overlay error",
                fill="black",
                font=FONT_NORMAL
            )

    else:

        draw.text(
            (
                IMAGE_WIDTH + 20,
                30
            ),
            "SAM2 overlay not found",
            fill="black",
            font=FONT_NORMAL
        )

    # --------------------------------------------------------
    # Labels above images
    # --------------------------------------------------------

    draw.text(
        (
            15,
            10
        ),
        "ORIGINAL",
        fill="white",
        stroke_width=2,
        stroke_fill="black",
        font=FONT_NORMAL
    )

    draw.text(
        (
            IMAGE_WIDTH + 15,
            10
        ),
        "SAM2 OVERLAY",
        fill="white",
        stroke_width=2,
        stroke_fill="black",
        font=FONT_NORMAL
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    info_y = IMAGE_HEIGHT + 10

    disease = str(
        row.get(
            "disease",
            ""
        )
    )

    severity = safe_float(
        row.get("severity")
    )

    qc_status = str(
        row.get(
            "qc_status",
            ""
        )
    )

    qc_score = safe_float(
        row.get("qc_score")
    )

    flags = str(
        row.get(
            "qc_flags",
            ""
        )
    )

    filename = str(
        row.get(
            "filename",
            ""
        )
    )

    # Disease
    draw.text(
        (
            15,
            info_y
        ),
        disease,
        fill="black",
        font=FONT_NORMAL
    )

    # Severity
    severity_text = (
        f"Severity: {severity:.2f}%"
        if np.isfinite(severity)
        else "Severity: UNKNOWN"
    )

    draw.text(
        (
            15,
            info_y + 28
        ),
        severity_text,
        fill="black",
        font=FONT_SEVERITY
    )

    # QC status
    qc_text = (
        f"QC: {qc_status}"
    )

    draw.text(
        (
            15,
            info_y + 62
        ),
        qc_text,
        fill="black",
        font=FONT_NORMAL
    )

    # QC score
    if np.isfinite(qc_score):

        score_text = (
            f"QC score: {qc_score:.0f}/100"
        )

        draw.text(
            (
                300,
                info_y + 62
            ),
            score_text,
            fill="black",
            font=FONT_NORMAL
        )

    # Flags
    flag_text = (
        "Flags: "
        + flags
        if flags
        else "Flags: none"
    )

    flag_lines = wrap_text(
        draw,
        flag_text,
        FONT_SMALL,
        card_width - 30
    )

    y = info_y + 92

    for line in flag_lines[:2]:

        draw.text(
            (
                15,
                y
            ),
            line,
            fill="black",
            font=FONT_SMALL
        )

        y += 20

    return card


# ============================================================
# CONTACT SHEET
# ============================================================

def create_contact_sheet(
    dataframe,
    output_directory,
    sheet_name
):

    if len(dataframe) == 0:

        return []

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    pages = []

    total_pages = math.ceil(
        len(dataframe)
        / IMAGES_PER_PAGE
    )

    page_width = (
        PAGE_COLUMNS
        * IMAGE_WIDTH
        * 2
    )

    page_card_width = (
        IMAGE_WIDTH * 2
    )

    page_card_height = (
        IMAGE_HEIGHT
        + TEXT_HEIGHT
    )

    page_height = (
        PAGE_ROWS
        * page_card_height
    )

    for page_index in range(
        total_pages
    ):

        start = (
            page_index
            * IMAGES_PER_PAGE
        )

        end = min(
            start + IMAGES_PER_PAGE,
            len(dataframe)
        )

        page_df = dataframe.iloc[
            start:end
        ]

        sheet = Image.new(
            "RGB",
            (
                page_width,
                page_height
            ),
            "white"
        )

        for position, (_, row) in enumerate(
            page_df.iterrows()
        ):

            card = create_card(
                row
            )

            row_index = (
                position
                // PAGE_COLUMNS
            )

            column_index = (
                position
                % PAGE_COLUMNS
            )

            x = (
                column_index
                * page_card_width
            )

            y = (
                row_index
                * page_card_height
            )

            sheet.paste(
                card,
                (
                    x,
                    y
                )
            )

        output_path = (
            output_directory
            / (
                f"{sheet_name}_"
                f"page_{page_index + 1:03d}.jpg"
            )
        )

        sheet.save(
            output_path,
            quality=JPEG_QUALITY
        )

        pages.append(
            output_path
        )

    return pages


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("SAM2 GENERAL SEVERITY — VISUAL QC CONTACT SHEETS")
    print("=" * 75)

    print(
        f"\nQC CSV:\n{CSV_PATH}"
    )

    print(
        f"\nOverlay directory:\n{OVERLAY_DIR}"
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"\nQC CSV not found:\n{CSV_PATH}"
        )

    if not OVERLAY_DIR.exists():

        raise FileNotFoundError(
            f"\nOverlay directory not found:\n{OVERLAY_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"\nLoaded {len(df):,} records."
    )

    # Numeric severity
    df["severity_numeric"] = pd.to_numeric(
        df["severity"],
        errors="coerce"
    )

    # ========================================================
    # 1. ALL DISEASES — REPRESENTATIVE SAMPLES
    # ========================================================

    print(
        "\nCreating disease-wise representative sheets..."
    )

    disease_output = (
        OUTPUT_DIR
        / "by_disease"
    )

    for disease, group in df.groupby(
        "disease"
    ):

        group = group.copy()

        # ----------------------------------------------------
        # We intentionally include a mixture:
        #
        # low severity
        # medium severity
        # high severity
        # review
        #
        # This is much better than only inspecting extreme
        # images.
        # ----------------------------------------------------

        group = group.sort_values(
            "severity_numeric"
        )

        n = len(group)

        if n <= SAMPLES_PER_DISEASE:

            selected = group

        else:

            indices = np.linspace(
                0,
                n - 1,
                SAMPLES_PER_DISEASE,
                dtype=int
            )

            selected = group.iloc[
                indices
            ]

        safe_name = clean_filename(
            disease
        )

        create_contact_sheet(
            selected,
            disease_output / safe_name,
            safe_name
        )

        print(
            f"  {disease}: "
            f"{len(selected)} samples"
        )

    # ========================================================
    # 2. HIGH SEVERITY
    # ========================================================

    print(
        "\nCreating high-severity review sheets..."
    )

    high_severity = df[
        df["severity_numeric"] >= 45.0
    ].copy()

    high_severity = high_severity.sort_values(
        "severity_numeric",
        ascending=False
    )

    high_severity = high_severity.head(
        HIGH_SEVERITY_SAMPLES
    )

    create_contact_sheet(
        high_severity,
        OUTPUT_DIR / "high_severity",
        "high_severity"
    )

    print(
        f"  High severity samples: "
        f"{len(high_severity)}"
    )

    # ========================================================
    # 3. REVIEW REQUIRED
    # ========================================================

    print(
        "\nCreating QC-review sheets..."
    )

    review = df[
        df["qc_status"] == "REVIEW"
    ].copy()

    review = review.sort_values(
        "severity_numeric",
        ascending=False
    )

    review = review.head(
        REVIEW_SAMPLES
    )

    create_contact_sheet(
        review,
        OUTPUT_DIR / "review",
        "review"
    )

    print(
        f"  Review samples: "
        f"{len(review)}"
    )

    # ========================================================
    # 4. LOW SEVERITY
    # ========================================================

    print(
        "\nCreating low-severity sheets..."
    )

    low = df[
        df["severity_numeric"] < 5
    ].copy()

    low = low.sort_values(
        "severity_numeric"
    )

    low = low.head(
        30
    )

    create_contact_sheet(
        low,
        OUTPUT_DIR / "low_severity",
        "low_severity"
    )

    # ========================================================
    # 5. HIGH CONFIDENCE ONLY
    # ========================================================

    print(
        "\nCreating HIGH_CONFIDENCE samples..."
    )

    high_conf = df[
        df["qc_status"]
        == "HIGH_CONFIDENCE"
    ].copy()

    high_conf = high_conf.sort_values(
        "severity_numeric",
        ascending=False
    )

    high_conf = high_conf.head(
        30
    )

    create_contact_sheet(
        high_conf,
        OUTPUT_DIR / "high_confidence",
        "high_confidence"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 75)
    print("VISUAL QC CONTACT SHEETS CREATED")
    print("=" * 75)

    print(
        f"\nOutput:\n{OUTPUT_DIR}"
    )

    print(
        "\nImportant folders:"
    )

    print(
        f"\n1. Disease-wise:"
        f"\n   {OUTPUT_DIR / 'by_disease'}"
    )

    print(
        f"\n2. High severity:"
        f"\n   {OUTPUT_DIR / 'high_severity'}"
    )

    print(
        f"\n3. Review:"
        f"\n   {OUTPUT_DIR / 'review'}"
    )

    print(
        f"\n4. Low severity:"
        f"\n   {OUTPUT_DIR / 'low_severity'}"
    )

    print(
        f"\n5. High confidence:"
        f"\n   {OUTPUT_DIR / 'high_confidence'}"
    )

    print("\n")
    print("=" * 75)

    print(
        "\nWHAT TO CHECK"
    )

    print("=" * 75)

    print(
        """
For each image, compare:

ORIGINAL
    ↓
Does the visible disease actually exist where
the red SAM2 region appears?

SAM2 OVERLAY
    ↓
Is red restricted to diseased tissue?

Check specifically for:

[1] RED ON HEALTHY LEAF
    Bad pseudo-label.

[2] RED ON SHADOW
    Bad pseudo-label.

[3] RED ON GROUND/BACKGROUND
    Very bad pseudo-label.

[4] DISEASE MISSED
    Bad pseudo-label.

[5] PARTIAL DISEASE ONLY
    Potentially underestimated severity.

[6] RED COVERING MOST OF THE LEAF
    Check whether the leaf is genuinely severely diseased.

[7] CLEAN/REASONABLE LESION MASK
    Good pseudo-label.

DO NOT manually edit the images yet.

We first want to understand how often each failure mode
occurs before changing the labeling pipeline.
"""
    )

    print("=" * 75)


if __name__ == "__main__":
    main()