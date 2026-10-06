# ============================================================
# HUMAN GT 120 - LABELME TO MASK + SEVERITY VALIDATION
# ============================================================

import os
import json
import cv2
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

ROOT = r"E:\smart_agriculture\smart_agriculture"

GT_ROOT = os.path.join(
    ROOT,
    r"ai\severity\dataset\human_gt_120"
)

IMAGE_DIR = os.path.join(
    GT_ROOT,
    "images"
)

ANNOTATION_DIR = os.path.join(
    GT_ROOT,
    "annotations"
)

MASK_DIR = os.path.join(
    GT_ROOT,
    "masks"
)

OVERLAY_DIR = os.path.join(
    GT_ROOT,
    "overlays"
)

METADATA_CSV = os.path.join(
    GT_ROOT,
    "metadata.csv"
)

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    MASK_DIR,
    exist_ok=True
)

os.makedirs(
    OVERLAY_DIR,
    exist_ok=True
)

# ============================================================
# SEVERITY FUNCTION
# ============================================================

def severity_level(severity):

    if severity <= 10:
        return "LOW"

    elif severity <= 40:
        return "MEDIUM"

    elif severity <= 60:
        return "HIGH"

    else:
        return "SEVERE"


# ============================================================
# LABELME POLYGON → MASK
# ============================================================

def create_mask_from_shapes(
    shapes,
    image_height,
    image_width,
    target_label
):

    mask = np.zeros(
        (image_height, image_width),
        dtype=np.uint8
    )

    found = False

    for shape in shapes:

        label = str(
            shape.get("label", "")
        ).strip().lower()

        if label != target_label:
            continue

        points = np.array(
            shape["points"],
            dtype=np.float32
        )

        if len(points) < 3:
            continue

        # Round coordinates and clip
        points[:, 0] = np.clip(
            np.round(points[:, 0]),
            0,
            image_width - 1
        )

        points[:, 1] = np.clip(
            np.round(points[:, 1]),
            0,
            image_height - 1
        )

        points = points.astype(
            np.int32
        )

        cv2.fillPoly(
            mask,
            [points],
            255
        )

        found = True

    return mask, found


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    image,
    leaf_mask,
    disease_mask
):

    overlay = image.copy()

    # Leaf boundary
    leaf_binary = (
        leaf_mask > 0
    ).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        leaf_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    # Disease region
    disease_binary = (
        disease_mask > 0
    ).astype(np.uint8)

    # Semi-transparent disease overlay
    disease_layer = overlay.copy()

    disease_layer[
        disease_binary > 0
    ] = (0, 0, 255)

    overlay = cv2.addWeighted(
        overlay,
        0.70,
        disease_layer,
        0.30,
        0
    )

    # Disease boundary
    disease_uint8 = (
        disease_binary * 255
    ).astype(np.uint8)

    disease_contours, _ = cv2.findContours(
        disease_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        disease_contours,
        -1,
        (0, 0, 255),
        2
    )

    return overlay


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 75)
print("HUMAN GROUND-TRUTH VALIDATION")
print("=" * 75)

if not os.path.exists(METADATA_CSV):

    raise FileNotFoundError(
        f"\nMetadata not found:\n{METADATA_CSV}"
    )

metadata = pd.read_csv(
    METADATA_CSV
)

print(
    "\nMetadata rows:",
    len(metadata)
)

# ============================================================
# FIND JSON FILES
# ============================================================

json_files = sorted(
    [
        f
        for f in os.listdir(
            ANNOTATION_DIR
        )
        if f.lower().endswith(".json")
    ]
)

print(
    "LabelMe JSON files:",
    len(json_files)
)

if len(json_files) == 0:

    raise RuntimeError(
        "\nNo LabelMe JSON files found.\n"
        "Annotate the images first."
    )

# ============================================================
# PROCESS
# ============================================================

results = []

errors = []

print(
    "\n"
    + "=" * 75
)

print(
    "PROCESSING HUMAN ANNOTATIONS"
)

print(
    "=" * 75
)

for index, json_name in enumerate(
    json_files,
    start=1
):

    json_path = os.path.join(
        ANNOTATION_DIR,
        json_name
    )

    try:

        # ----------------------------------------------------
        # Read LabelMe JSON
        # ----------------------------------------------------

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        image_filename = data.get(
            "imagePath",
            ""
        )

        # ----------------------------------------------------
        # Locate image
        # ----------------------------------------------------

        image_path = os.path.join(
            IMAGE_DIR,
            os.path.basename(
                image_filename
            )
        )

        # If imagePath doesn't match,
        # try JSON basename
        if not os.path.exists(
            image_path
        ):

            base = os.path.splitext(
                json_name
            )[0]

            candidates = [
                f
                for f in os.listdir(
                    IMAGE_DIR
                )
                if os.path.splitext(f)[0] == base
            ]

            if len(candidates) == 1:

                image_path = os.path.join(
                    IMAGE_DIR,
                    candidates[0]
                )

        if not os.path.exists(
            image_path
        ):

            raise FileNotFoundError(
                f"Image not found for {json_name}"
            )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise RuntimeError(
                "OpenCV could not read image."
            )

        height, width = image.shape[:2]

        # ----------------------------------------------------
        # Create masks
        # ----------------------------------------------------

        shapes = data.get(
            "shapes",
            []
        )

        leaf_mask, leaf_found = (
            create_mask_from_shapes(
                shapes,
                height,
                width,
                "leaf"
            )
        )

        disease_mask, disease_found = (
            create_mask_from_shapes(
                shapes,
                height,
                width,
                "disease"
            )
        )

        if not leaf_found:

            raise RuntimeError(
                "No 'leaf' annotation found."
            )

        if not disease_found:

            raise RuntimeError(
                "No 'disease' annotation found."
            )

        # ----------------------------------------------------
        # Ensure disease is inside leaf
        # ----------------------------------------------------

        disease_mask = cv2.bitwise_and(
            disease_mask,
            leaf_mask
        )

        # ----------------------------------------------------
        # Pixel counts
        # ----------------------------------------------------

        leaf_pixels = int(
            np.count_nonzero(
                leaf_mask
            )
        )

        disease_pixels = int(
            np.count_nonzero(
                disease_mask
            )
        )

        if leaf_pixels == 0:

            raise RuntimeError(
                "Leaf mask contains zero pixels."
            )

        # ----------------------------------------------------
        # TRUE HUMAN SEVERITY
        # ----------------------------------------------------

        human_severity = (
            disease_pixels
            /
            leaf_pixels
            *
            100.0
        )

        human_severity = float(
            np.clip(
                human_severity,
                0,
                100
            )
        )

        level = severity_level(
            human_severity
        )

        # ----------------------------------------------------
        # Existing SAM2 severity
        # ----------------------------------------------------

        matching = metadata[
            metadata[
                "new_filename"
            ]
            == os.path.basename(
                image_path
            )
        ]

        if len(matching) == 0:

            # fallback by filename
            matching = metadata[
                metadata[
                    "original_filename"
                ]
                == os.path.basename(
                    image_path
                )
            ]

        if len(matching) > 0:

            sam2_severity = float(
                matching.iloc[0][
                    "sam2_severity"
                ]
            )

            disease = matching.iloc[0][
                "disease"
            ]

            plant = matching.iloc[0][
                "plant"
            ]

        else:

            sam2_severity = np.nan
            disease = "UNKNOWN"
            plant = "UNKNOWN"

        # ----------------------------------------------------
        # Difference
        # ----------------------------------------------------

        if not np.isnan(
            sam2_severity
        ):

            severity_error = (
                human_severity
                -
                sam2_severity
            )

            absolute_error = abs(
                severity_error
            )

        else:

            severity_error = np.nan
            absolute_error = np.nan

        # ----------------------------------------------------
        # Save masks
        # ----------------------------------------------------

        base_name = os.path.splitext(
            os.path.basename(
                image_path
            )
        )[0]

        leaf_mask_path = os.path.join(
            MASK_DIR,
            base_name + "_leaf.png"
        )

        disease_mask_path = os.path.join(
            MASK_DIR,
            base_name + "_disease.png"
        )

        overlay_path = os.path.join(
            OVERLAY_DIR,
            base_name + "_overlay.jpg"
        )

        cv2.imwrite(
            leaf_mask_path,
            leaf_mask
        )

        cv2.imwrite(
            disease_mask_path,
            disease_mask
        )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        overlay = create_overlay(
            image,
            leaf_mask,
            disease_mask
        )

        cv2.imwrite(
            overlay_path,
            overlay
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append(
            {
                "json_file":
                    json_name,

                "image_file":
                    os.path.basename(
                        image_path
                    ),

                "plant":
                    plant,

                "disease":
                    disease,

                "leaf_pixels":
                    leaf_pixels,

                "disease_pixels":
                    disease_pixels,

                "human_severity":
                    human_severity,

                "human_severity_level":
                    level,

                "sam2_severity":
                    sam2_severity,

                "severity_error":
                    severity_error,

                "absolute_severity_error":
                    absolute_error,

                "leaf_mask":
                    leaf_mask_path,

                "disease_mask":
                    disease_mask_path,

                "overlay":
                    overlay_path
            }
        )

        print(
            f"[{index:03d}/{len(json_files):03d}] "
            f"{disease} | "
            f"Human={human_severity:.2f}% | "
            f"SAM2={sam2_severity:.2f}%"
        )

    except Exception as e:

        errors.append(
            {
                "json_file":
                    json_name,

                "error":
                    str(e)
            }
        )

        print(
            f"[ERROR] {json_name}: {e}"
        )

# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

results_csv = os.path.join(
    GT_ROOT,
    "human_gt_results.csv"
)

results_df.to_csv(
    results_csv,
    index=False
)

# ============================================================
# ERROR REPORT
# ============================================================

errors_csv = os.path.join(
    GT_ROOT,
    "annotation_errors.csv"
)

pd.DataFrame(
    errors
).to_csv(
    errors_csv,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "ANNOTATION PROCESSING SUMMARY"
)

print(
    "=" * 75
)

print(
    "JSON files:",
    len(json_files)
)

print(
    "Successfully processed:",
    len(results_df)
)

print(
    "Errors:",
    len(errors)
)

# ============================================================
# HUMAN SEVERITY DISTRIBUTION
# ============================================================

if len(results_df) > 0:

    print(
        "\nHuman severity distribution:"
    )

    distribution = (
        results_df[
            "human_severity_level"
        ]
        .value_counts()
        .reindex(
            [
                "LOW",
                "MEDIUM",
                "HIGH",
                "SEVERE"
            ],
            fill_value=0
        )
    )

    print(
        distribution.to_string()
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(
        "\nHuman severity statistics:"
    )

    print(
        results_df[
            "human_severity"
        ].describe().round(3).to_string()
    )

    # --------------------------------------------------------
    # SAM2 comparison
    # --------------------------------------------------------

    comparison = results_df[
        [
            "human_severity",
            "sam2_severity",
            "absolute_severity_error"
        ]
    ].dropna()

    if len(comparison) > 0:

        print(
            "\n"
            + "=" * 75
        )

        print(
            "SAM2 vs HUMAN SEVERITY"
        )

        print(
            "=" * 75
        )

        mae = (
            comparison[
                "absolute_severity_error"
            ]
            .mean()
        )

        bias = (
            comparison[
                "sam2_severity"
            ]
            -
            comparison[
                "human_severity"
            ]
        ).mean()

        rmse = np.sqrt(
            np.mean(
                (
                    comparison[
                        "sam2_severity"
                    ]
                    -
                    comparison[
                        "human_severity"
                    ]
                ) ** 2
            )
        )

        print(
            "Samples:",
            len(comparison)
        )

        print(
            f"SAM2 Severity MAE: "
            f"{mae:.4f}%"
        )

        print(
            f"SAM2 Severity RMSE: "
            f"{rmse:.4f}%"
        )

        print(
            f"SAM2 Mean Bias: "
            f"{bias:.4f}%"
        )

        # ----------------------------------------------------
        # Pearson correlation
        # ----------------------------------------------------

        if len(comparison) >= 2:

            correlation = (
                comparison[
                    "human_severity"
                ]
                .corr(
                    comparison[
                        "sam2_severity"
                    ]
                )
            )

            print(
                f"SAM2/Human correlation: "
                f"{correlation:.4f}"
            )

# ============================================================
# DISEASE-WISE RESULTS
# ============================================================

if (
    len(results_df) > 0
    and "disease" in results_df.columns
):

    disease_rows = []

    for disease, group in results_df.groupby(
        "disease"
    ):

        row = {
            "disease":
                disease,

            "samples":
                len(group),

            "human_mean":
                group[
                    "human_severity"
                ].mean(),

            "human_median":
                group[
                    "human_severity"
                ].median(),

            "human_min":
                group[
                    "human_severity"
                ].min(),

            "human_max":
                group[
                    "human_severity"
                ].max()
        }

        valid = group[
            [
                "human_severity",
                "sam2_severity"
            ]
        ].dropna()

        if len(valid) > 0:

            row[
                "sam2_mae"
            ] = np.mean(
                np.abs(
                    valid[
                        "human_severity"
                    ]
                    -
                    valid[
                        "sam2_severity"
                    ]
                )
            )

            row[
                "sam2_rmse"
            ] = np.sqrt(
                np.mean(
                    (
                        valid[
                            "human_severity"
                        ]
                        -
                        valid[
                            "sam2_severity"
                        ]
                    ) ** 2
                )
            )

            if len(valid) >= 2:

                row[
                    "correlation"
                ] = valid[
                    "human_severity"
                ].corr(
                    valid[
                        "sam2_severity"
                    ]
                )

            else:

                row[
                    "correlation"
                ] = np.nan

        else:

            row["sam2_mae"] = np.nan
            row["sam2_rmse"] = np.nan
            row["correlation"] = np.nan

        disease_rows.append(
            row
        )

    disease_df = pd.DataFrame(
        disease_rows
    )

    disease_csv = os.path.join(
        GT_ROOT,
        "human_gt_disease_statistics.csv"
    )

    disease_df.to_csv(
        disease_csv,
        index=False
    )

    print(
        "\nDisease-wise comparison:"
    )

    print(
        disease_df.round(
            3
        ).to_string(
            index=False
        )
    )

# ============================================================
# FINAL
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "HUMAN GT PROCESSING COMPLETE"
)

print(
    "=" * 75
)

print(
    "Results:"
)

print(
    results_csv
)

print(
    "\nMasks:"
)

print(
    MASK_DIR
)

print(
    "\nOverlays:"
)

print(
    OVERLAY_DIR
)

print(
    "\nErrors:"
)

print(
    errors_csv
)

print(
    "\nIMPORTANT:"
)

print(
    "Human severity = disease pixels / leaf pixels × 100"
)

print(
    "These human annotations are the independent reference."
)

print(
    "=" * 75
)