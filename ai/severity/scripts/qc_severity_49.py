from pathlib import Path
import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SEVERITY_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "severity_49"
)

CSV_PATH = SEVERITY_DIR / "severity_labels_49.csv"
DISEASE_MASK_DIR = SEVERITY_DIR / "disease_masks"
LEAF_MASK_DIR = SEVERITY_DIR / "leaf_masks"
OVERLAY_DIR = SEVERITY_DIR / "overlays"

QC_DIR = SEVERITY_DIR / "qc"
FLAGGED_DIR = QC_DIR / "flagged"

QC_DIR.mkdir(parents=True, exist_ok=True)
FLAGGED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# THRESHOLDS
# ============================================================

# These are QC warning thresholds, NOT biological definitions.

VERY_SMALL_DISEASE_PERCENT = 0.5
VERY_LARGE_DISEASE_PERCENT = 45.0

MIN_LEAF_PIXELS = 1000

MAX_COMPONENTS_TO_REPORT = 50


# ============================================================
# HELPERS
# ============================================================

def find_mask(directory: Path, stem: str):
    """
    Find a mask whose filename starts with the requested stem.
    """
    matches = list(directory.glob(f"{stem}*"))

    if not matches:
        return None

    return matches[0]


def load_binary_mask(path: Path):
    """
    Load an image as a binary mask.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return None

    return mask > 0


def connected_components(mask):
    """
    Return number of connected foreground components.
    """
    mask_uint8 = (mask.astype(np.uint8) * 255)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8
    )

    # label 0 is background
    component_count = max(0, num_labels - 1)

    return component_count, stats[1:] if num_labels > 1 else np.empty((0, 5))


def save_flagged_overlay(
    stem,
    leaf_mask,
    disease_mask,
    flags,
    output_path
):
    """
    Create a simple QC visualization.

    Green = leaf
    Red   = disease
    Yellow = disease outside leaf
    """

    h, w = leaf_mask.shape

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Leaf boundary / area
    canvas[leaf_mask] = (0, 180, 0)

    # Disease inside leaf
    disease_inside = disease_mask & leaf_mask
    canvas[disease_inside] = (0, 0, 255)

    # Disease outside leaf
    disease_outside = disease_mask & (~leaf_mask)
    canvas[disease_outside] = (0, 255, 255)

    # Put text on image
    text = " | ".join(flags)

    scale = max(0.4, min(1.0, w / 1200))

    cv2.putText(
        canvas,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(str(output_path), canvas)


# ============================================================
# LOAD CSV
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Severity CSV not found:\n{CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

required_columns = {
    "image",
    "severity_percent"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"CSV is missing required columns: {sorted(missing_columns)}\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# QC
# ============================================================

records = []

print("=" * 70)
print("Severity Dataset QC")
print("=" * 70)

print(f"CSV rows: {len(df)}")
print(f"Disease mask directory: {DISEASE_MASK_DIR}")
print(f"Leaf mask directory:    {LEAF_MASK_DIR}")
print()


for _, row in df.iterrows():

    image_name = str(row["image"])
    severity_csv = float(row["severity_percent"])

    stem = Path(image_name).stem

    disease_path = find_mask(
        DISEASE_MASK_DIR,
        stem
    )

    leaf_path = find_mask(
        LEAF_MASK_DIR,
        stem
    )

    flags = []

    # --------------------------------------------------------
    # Missing masks
    # --------------------------------------------------------

    if disease_path is None:
        flags.append("MISSING_DISEASE_MASK")

    if leaf_path is None:
        flags.append("MISSING_LEAF_MASK")

    if flags:
        records.append({
            "image": image_name,
            "severity_csv": severity_csv,
            "severity_recomputed": np.nan,
            "leaf_pixels": np.nan,
            "disease_pixels": np.nan,
            "disease_outside_leaf_pixels": np.nan,
            "disease_outside_leaf_percent": np.nan,
            "connected_components": np.nan,
            "flag": ";".join(flags),
        })

        continue

    # --------------------------------------------------------
    # Load masks
    # --------------------------------------------------------

    leaf_mask = load_binary_mask(leaf_path)
    disease_mask = load_binary_mask(disease_path)

    if leaf_mask is None:
        flags.append("INVALID_LEAF_MASK")

    if disease_mask is None:
        flags.append("INVALID_DISEASE_MASK")

    if flags:
        records.append({
            "image": image_name,
            "severity_csv": severity_csv,
            "severity_recomputed": np.nan,
            "leaf_pixels": np.nan,
            "disease_pixels": np.nan,
            "disease_outside_leaf_pixels": np.nan,
            "disease_outside_leaf_percent": np.nan,
            "connected_components": np.nan,
            "flag": ";".join(flags),
        })

        continue

    # --------------------------------------------------------
    # Shape check
    # --------------------------------------------------------

    if leaf_mask.shape != disease_mask.shape:
        flags.append("MASK_SHAPE_MISMATCH")

        records.append({
            "image": image_name,
            "severity_csv": severity_csv,
            "severity_recomputed": np.nan,
            "leaf_pixels": np.sum(leaf_mask),
            "disease_pixels": np.sum(disease_mask),
            "disease_outside_leaf_pixels": np.nan,
            "disease_outside_leaf_percent": np.nan,
            "connected_components": np.nan,
            "flag": ";".join(flags),
        })

        continue

    # --------------------------------------------------------
    # Pixel statistics
    # --------------------------------------------------------

    leaf_pixels = int(np.sum(leaf_mask))

    disease_pixels = int(np.sum(disease_mask))

    disease_inside = disease_mask & leaf_mask

    disease_inside_pixels = int(
        np.sum(disease_inside)
    )

    disease_outside = disease_mask & (~leaf_mask)

    disease_outside_pixels = int(
        np.sum(disease_outside)
    )

    # --------------------------------------------------------
    # Severity recomputation
    # --------------------------------------------------------

    if leaf_pixels > 0:
        severity_recomputed = (
            disease_inside_pixels
            / leaf_pixels
            * 100.0
        )
    else:
        severity_recomputed = np.nan
        flags.append("EMPTY_LEAF_MASK")

    # --------------------------------------------------------
    # Outside-leaf percentage
    # --------------------------------------------------------

    if disease_pixels > 0:
        outside_percent = (
            disease_outside_pixels
            / disease_pixels
            * 100.0
        )
    else:
        outside_percent = 0.0

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    component_count, component_stats = connected_components(
        disease_inside
    )

    # --------------------------------------------------------
    # QC checks
    # --------------------------------------------------------

    if leaf_pixels < MIN_LEAF_PIXELS:
        flags.append("VERY_SMALL_LEAF_MASK")

    if severity_recomputed < VERY_SMALL_DISEASE_PERCENT:
        flags.append("VERY_LOW_SEVERITY")

    if severity_recomputed > VERY_LARGE_DISEASE_PERCENT:
        flags.append("VERY_HIGH_SEVERITY")

    if disease_outside_pixels > 0:
        flags.append("DISEASE_OUTSIDE_LEAF")

    # --------------------------------------------------------
    # Compare CSV severity with recomputed severity
    # --------------------------------------------------------

    severity_difference = abs(
        severity_csv - severity_recomputed
    )

    if severity_difference > 0.5:
        flags.append("CSV_SEVERITY_MISMATCH")

    # --------------------------------------------------------
    # Save flagged visualization
    # --------------------------------------------------------

    if flags:

        output_path = (
            FLAGGED_DIR
            / f"{stem}_qc.jpg"
        )

        save_flagged_overlay(
            stem,
            leaf_mask,
            disease_mask,
            flags,
            output_path
        )

    # --------------------------------------------------------
    # Record
    # --------------------------------------------------------

    records.append({
        "image": image_name,
        "severity_csv": round(severity_csv, 4),
        "severity_recomputed": round(
            severity_recomputed, 4
        ),
        "severity_difference": round(
            severity_difference, 4
        ),
        "leaf_pixels": leaf_pixels,
        "disease_pixels": disease_pixels,
        "disease_inside_leaf_pixels": disease_inside_pixels,
        "disease_outside_leaf_pixels": disease_outside_pixels,
        "disease_outside_leaf_percent": round(
            outside_percent, 4
        ),
        "connected_components": component_count,
        "flag": ";".join(flags),
    })


# ============================================================
# SAVE REPORT
# ============================================================

qc_df = pd.DataFrame(records)

report_path = QC_DIR / "severity_qc_report.csv"

qc_df.to_csv(
    report_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

flagged = qc_df[
    qc_df["flag"].fillna("").astype(str).str.len() > 0
]

clean = qc_df[
    qc_df["flag"].fillna("").astype(str).str.len() == 0
]


print()
print("=" * 70)
print("QC COMPLETE")
print("=" * 70)

print(f"Total samples:       {len(qc_df)}")
print(f"Clean samples:       {len(clean)}")
print(f"Flagged samples:     {len(flagged)}")

print()
print(f"QC report:")
print(report_path)

print()
print(f"Flagged visualizations:")
print(FLAGGED_DIR)

print()
print("Flag summary:")

if len(flagged) == 0:
    print("No QC flags detected.")
else:

    all_flags = []

    for value in flagged["flag"]:
        if isinstance(value, str):
            all_flags.extend(
                value.split(";")
            )

    for flag_name in sorted(set(all_flags)):
        count = all_flags.count(flag_name)
        print(
            f"  {flag_name}: {count}"
        )

print()
print("=" * 70)