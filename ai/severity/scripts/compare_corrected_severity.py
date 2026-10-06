from pathlib import Path
import cv2
import pandas as pd
import numpy as np

ROOT = Path(r"E:\smart_agriculture\smart_agriculture")

V2_CSV = (
    ROOT / "ai" / "severity" / "results"
    / "severity_250_v2" / "severity_labels_250_v2.csv"
)

V2_LEAF_MASK_DIR = (
    ROOT / "ai" / "severity" / "results"
    / "severity_250_v2" / "leaf_masks"
)

CORRECTED_DIR = (
    ROOT / "ai" / "severity" / "results"
    / "corrected_masks"
)

OUT_DIR = (
    ROOT / "ai" / "severity" / "results"
    / "corrected_severity_check"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load original V2 severity CSV
# ---------------------------------------------------------
df = pd.read_csv(V2_CSV)

# Normalize filename column
filename_col = None
for col in ["filename", "image", "File Name", "file_name", "image_name"]:
    if col in df.columns:
        filename_col = col
        break

if filename_col is None:
    raise ValueError(
        f"Could not find filename column. Available columns: {list(df.columns)}"
    )

# Normalize severity column
severity_col = None
for col in ["severity_percent", "severity", "Severity", "severity_pct"]:
    if col in df.columns:
        severity_col = col
        break

if severity_col is None:
    raise ValueError(
        f"Could not find severity column. Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------
# Six corrected files
# ---------------------------------------------------------
corrected_files = sorted(
    CORRECTED_DIR.glob("*_corrected_mask.png")
)

print(f"Corrected masks found: {len(corrected_files)}")

if len(corrected_files) != 6:
    print("WARNING: Expected 6 corrected masks.")


results = []


# ---------------------------------------------------------
# Match corrected masks with original records
# ---------------------------------------------------------
for corrected_path in corrected_files:

    stem = corrected_path.name.replace("_corrected_mask.png", "")

    # Find matching original row
    matches = df[
        df[filename_col].astype(str).str.contains(
            stem,
            case=False,
            regex=False
        )
    ]

    # If exact stem matching fails, try UUID/image-name fragments
    if len(matches) == 0:
        matches = df[
            df[filename_col].astype(str).apply(
                lambda x: stem in Path(x).stem
            )
        ]

    if len(matches) == 0:
        print(f"NOT FOUND IN CSV: {stem}")
        continue

    row = matches.iloc[0]

    original_filename = str(row[filename_col])
    original_severity = float(row[severity_col])

    # -----------------------------------------------------
    # Find corresponding leaf mask
    # -----------------------------------------------------
    possible_leaf_masks = list(
        V2_LEAF_MASK_DIR.glob(f"*{stem}*.png")
    )

    if not possible_leaf_masks:
        # fallback: search all masks
        possible_leaf_masks = [
            p for p in V2_LEAF_MASK_DIR.glob("*.png")
            if stem in p.name
        ]

    if not possible_leaf_masks:
        print(f"LEAF MASK NOT FOUND: {stem}")
        continue

    leaf_mask_path = possible_leaf_masks[0]

    # -----------------------------------------------------
    # Read masks
    # -----------------------------------------------------
    disease_mask = cv2.imread(
        str(corrected_path),
        cv2.IMREAD_GRAYSCALE
    )

    leaf_mask = cv2.imread(
        str(leaf_mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if disease_mask is None:
        print(f"Could not read disease mask: {corrected_path}")
        continue

    if leaf_mask is None:
        print(f"Could not read leaf mask: {leaf_mask_path}")
        continue

    # Ensure same size
    if disease_mask.shape != leaf_mask.shape:
        disease_mask = cv2.resize(
            disease_mask,
            (leaf_mask.shape[1], leaf_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # Binary masks
    leaf = leaf_mask > 0
    disease = disease_mask > 0

    # Disease cannot exist outside leaf
    disease = disease & leaf

    leaf_pixels = np.sum(leaf)
    disease_pixels = np.sum(disease)

    if leaf_pixels == 0:
        print(f"ZERO LEAF PIXELS: {stem}")
        continue

    corrected_severity = (
        disease_pixels / leaf_pixels
    ) * 100.0

    difference = corrected_severity - original_severity

    abs_difference = abs(difference)

    results.append({
        "image": original_filename,
        "original_v2_severity": original_severity,
        "corrected_severity": corrected_severity,
        "difference": difference,
        "absolute_difference": abs_difference,
        "leaf_pixels": int(leaf_pixels),
        "disease_pixels": int(disease_pixels),
    })


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------
result_df = pd.DataFrame(results)

if len(result_df) == 0:
    raise RuntimeError("No corrected images could be matched.")

result_df = result_df.sort_values(
    "corrected_severity",
    ascending=False
)

output_csv = OUT_DIR / "corrected_severity_comparison.csv"

result_df.to_csv(
    output_csv,
    index=False
)

# ---------------------------------------------------------
# Print clean report
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("CORRECTED SEVERITY COMPARISON")
print("=" * 70)

for _, r in result_df.iterrows():

    print(f"\nImage: {r['image']}")
    print(
        f"  Original V2 : {r['original_v2_severity']:.2f}%"
    )
    print(
        f"  Corrected   : {r['corrected_severity']:.2f}%"
    )
    print(
        f"  Difference  : {r['difference']:+.2f} percentage points"
    )

print("\n" + "-" * 70)

print(
    f"Mean original severity : "
    f"{result_df['original_v2_severity'].mean():.2f}%"
)

print(
    f"Mean corrected severity: "
    f"{result_df['corrected_severity'].mean():.2f}%"
)

print(
    f"Mean absolute change   : "
    f"{result_df['absolute_difference'].mean():.2f} percentage points"
)

print("\nSaved:")
print(output_csv)