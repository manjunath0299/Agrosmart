from pathlib import Path
import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "severity_109_v3"
)

CSV_PATH = (
    RESULTS_DIR
    / "severity_labels_109_v3.csv"
)

LEAF_DIR = (
    RESULTS_DIR
    / "leaf_masks"
)

DISEASE_DIR = (
    RESULTS_DIR
    / "disease_masks"
)

QC_DIR = (
    RESULTS_DIR
    / "qc"
)

FLAGGED_DIR = (
    QC_DIR
    / "flagged"
)

QC_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FLAGGED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 70)
print("SEVERITY V3 DATASET QC")
print("=" * 70)


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(
    CSV_PATH
)

print()
print(
    f"CSV rows: {len(df)}"
)


# ============================================================
# QC RULES
# ============================================================

flags = []


for _, row in df.iterrows():

    image_name = row["image"]

    severity = float(
        row["severity_percent"]
    )

    leaf_area = float(
        row["leaf_area_percent_of_image"]
    )

    disease_pixels = int(
        row["disease_pixels"]
    )

    leaf_pixels = int(
        row["leaf_pixels"]
    )

    problems = []

    # --------------------------------------------------------
    # Basic numeric checks
    # --------------------------------------------------------

    if not (
        0 <= severity <= 100
    ):
        problems.append(
            "severity_out_of_range"
        )

    if leaf_pixels <= 0:
        problems.append(
            "zero_leaf_pixels"
        )

    if disease_pixels < 0:
        problems.append(
            "negative_disease_pixels"
        )

    if disease_pixels > leaf_pixels:
        problems.append(
            "disease_larger_than_leaf"
        )

    # --------------------------------------------------------
    # Leaf area sanity
    # --------------------------------------------------------

    if leaf_area < 10:
        problems.append(
            "very_small_leaf_area"
        )

    if leaf_area > 75:
        problems.append(
            "very_large_leaf_area"
        )

    # --------------------------------------------------------
    # Severity extremes
    # --------------------------------------------------------

    if severity >= 90:
        problems.append(
            "extreme_severity"
        )

    # --------------------------------------------------------
    # File existence
    # --------------------------------------------------------

    leaf_mask_name = row[
        "leaf_mask"
    ]

    disease_mask_name = row[
        "disease_mask"
    ]

    leaf_path = (
        LEAF_DIR
        / leaf_mask_name
    )

    disease_path = (
        DISEASE_DIR
        / disease_mask_name
    )

    if not leaf_path.exists():
        problems.append(
            "missing_leaf_mask"
        )

    if not disease_path.exists():
        problems.append(
            "missing_disease_mask"
        )

    # --------------------------------------------------------
    # Mask geometry
    # --------------------------------------------------------

    if (
        leaf_path.exists()
        and disease_path.exists()
    ):

        leaf_mask = cv2.imread(
            str(leaf_path),
            cv2.IMREAD_GRAYSCALE
        )

        disease_mask = cv2.imread(
            str(disease_path),
            cv2.IMREAD_GRAYSCALE
        )

        if leaf_mask is None:
            problems.append(
                "leaf_mask_read_error"
            )

        if disease_mask is None:
            problems.append(
                "disease_mask_read_error"
            )

        if (
            leaf_mask is not None
            and disease_mask is not None
        ):

            if (
                leaf_mask.shape
                != disease_mask.shape
            ):
                problems.append(
                    "mask_shape_mismatch"
                )

            leaf_bool = (
                leaf_mask > 0
            )

            disease_bool = (
                disease_mask > 0
            )

            # Disease should never exist
            # outside the final leaf mask.
            outside = (
                disease_bool
                & ~leaf_bool
            )

            if np.any(outside):
                problems.append(
                    "disease_outside_leaf"
                )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    flags.append({
        "image": image_name,
        "severity_percent": severity,
        "leaf_area_percent": leaf_area,
        "leaf_pixels": leaf_pixels,
        "disease_pixels": disease_pixels,
        "flag_count": len(problems),
        "flags": ";".join(problems)
    })


# ============================================================
# SAVE REPORT
# ============================================================

qc_df = pd.DataFrame(
    flags
)

report_path = (
    QC_DIR
    / "severity_qc_report_v3.csv"
)

qc_df.to_csv(
    report_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

flagged = qc_df[
    qc_df["flag_count"] > 0
]

clean = qc_df[
    qc_df["flag_count"] == 0
]


print()
print("=" * 70)
print("QC COMPLETE")
print("=" * 70)

print(
    f"Total samples:   {len(qc_df)}"
)

print(
    f"Clean samples:    {len(clean)}"
)

print(
    f"Flagged samples:  {len(flagged)}"
)

print()
print(
    "QC report:"
)

print(
    report_path
)


# ============================================================
# SHOW FLAGS
# ============================================================

if len(flagged) > 0:

    print()
    print("Flagged samples:")
    print()

    print(
        flagged[
            [
                "image",
                "severity_percent",
                "leaf_area_percent",
                "flags"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print()
    print(
        "No automated QC flags detected."
    )


print()
print("=" * 70)