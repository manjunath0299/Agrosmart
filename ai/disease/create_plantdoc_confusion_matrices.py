# ============================================================
# create_plantdoc_confusion_matrices.py
#
# Creates matched PlantDoc confusion matrices for:
#   1. Original EfficientNetV2-B0
#   2. Domain-Adapted EfficientNetV2-B0
#
# Uses the prediction CSV files already generated.
#
# PlantDoc test set:
#   134 images
#   16 mapped true classes
#
# The model has 38 output classes.
# Predictions outside the 16 PlantDoc classes are grouped into:
#   "Other (22 classes)"
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    BASE_DIR
    / "ai"
    / "disease"
    / "results"
)

BASELINE_DIR = (
    RESULTS_DIR
    / "baseline_plantdoc"
)

ADAPTED_DIR = (
    RESULTS_DIR
    / "domain_adaptation"
)

FINAL_DIR = (
    RESULTS_DIR
    / "final_comparison"
)

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. PREDICTION FILES
# ============================================================

BASELINE_PREDICTIONS = (
    BASELINE_DIR
    / "baseline_plantdoc_predictions.csv"
)

ADAPTED_PREDICTIONS = (
    ADAPTED_DIR
    / "plantdoc_predictions.csv"
)


# ============================================================
# 3. CHECK FILES
# ============================================================

if not BASELINE_PREDICTIONS.exists():

    raise FileNotFoundError(
        "Baseline prediction file not found:\n"
        f"{BASELINE_PREDICTIONS}"
    )


if not ADAPTED_PREDICTIONS.exists():

    raise FileNotFoundError(
        "Domain-adapted prediction file not found:\n"
        f"{ADAPTED_PREDICTIONS}"
    )


# ============================================================
# 4. THE 16 PLANTDOC MAPPED CLASSES
#
# These are the actual classes represented in the external
# PlantDoc test set.
# ============================================================

PLANTDOC_CLASSES = [

    "Apple___Apple_scab",

    "Apple___Cedar_apple_rust",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",

    "Corn_(maize)___Common_rust_",

    "Grape___Black_rot",

    "Pepper,_bell___Bacterial_spot",

    "Potato___Early_blight",

    "Potato___Late_blight",

    "Squash___Powdery_mildew",

    "Tomato___Bacterial_spot",

    "Tomato___Early_blight",

    "Tomato___Late_blight",

    "Tomato___Leaf_Mold",

    "Tomato___Septoria_leaf_spot",

    "Tomato___Tomato_mosaic_virus",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]


OTHER_LABEL = (
    "Other (22 PlantVillage classes)"
)


# ============================================================
# 5. LOAD PREDICTIONS
# ============================================================

baseline_df = pd.read_csv(
    BASELINE_PREDICTIONS
)

adapted_df = pd.read_csv(
    ADAPTED_PREDICTIONS
)


print("=" * 75)
print("PLANTDOC CONFUSION MATRIX ANALYSIS")
print("=" * 75)

print()

print(
    f"Baseline predictions: "
    f"{len(baseline_df)}"
)

print(
    f"Adapted predictions: "
    f"{len(adapted_df)}"
)

print()


# ============================================================
# 6. VALIDATE TEST SIZE
# ============================================================

if len(baseline_df) != 134:

    raise RuntimeError(
        "Baseline prediction file does not contain "
        f"134 images. Found {len(baseline_df)}."
    )


if len(adapted_df) != 134:

    raise RuntimeError(
        "Adapted prediction file does not contain "
        f"134 images. Found {len(adapted_df)}."
    )


# ============================================================
# 7. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = {
    "true_class",
    "predicted_class",
}


for name, df in [
    ("Baseline", baseline_df),
    ("Domain-Adapted", adapted_df),
]:

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            f"{name} prediction file is missing "
            f"columns: {missing}"
        )


# ============================================================
# 8. CONVERT PREDICTIONS INTO 16 + OTHER CLASSES
# ============================================================

def prepare_labels(df):

    true_labels = []
    predicted_labels = []


    for _, row in df.iterrows():

        true_class = row[
            "true_class"
        ]

        predicted_class = row[
            "predicted_class"
        ]


        # ----------------------------------------------------
        # True labels must belong to the 16 PlantDoc classes.
        # ----------------------------------------------------

        if true_class not in PLANTDOC_CLASSES:

            raise RuntimeError(
                "Unexpected true class found:\n"
                f"{true_class}"
            )


        true_labels.append(
            true_class
        )


        # ----------------------------------------------------
        # If prediction belongs to one of the 16 mapped
        # classes, keep it.
        #
        # Otherwise classify it as "Other".
        # ----------------------------------------------------

        if predicted_class in PLANTDOC_CLASSES:

            predicted_labels.append(
                predicted_class
            )

        else:

            predicted_labels.append(
                OTHER_LABEL
            )


    return (
        true_labels,
        predicted_labels
    )


# ============================================================
# 9. CREATE CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(
    true_labels,
    predicted_labels,
):

    matrix_labels = (
        PLANTDOC_CLASSES
        + [OTHER_LABEL]
    )


    matrix = np.zeros(
        (
            len(PLANTDOC_CLASSES),
            len(matrix_labels)
        ),
        dtype=int
    )


    label_to_column = {
        label: index
        for index, label
        in enumerate(matrix_labels)
    }


    for true_label, predicted_label in zip(
        true_labels,
        predicted_labels,
    ):

        row_index = (
            PLANTDOC_CLASSES.index(
                true_label
            )
        )


        column_index = (
            label_to_column[
                predicted_label
            ]
        )


        matrix[
            row_index,
            column_index
        ] += 1


    matrix_df = pd.DataFrame(
        matrix,

        index=PLANTDOC_CLASSES,

        columns=matrix_labels,
    )


    return matrix_df


# ============================================================
# 10. LOAD AND PROCESS BOTH MODELS
# ============================================================

baseline_true, baseline_pred = (
    prepare_labels(
        baseline_df
    )
)


adapted_true, adapted_pred = (
    prepare_labels(
        adapted_df
    )
)


baseline_matrix = (
    create_confusion_matrix(
        baseline_true,
        baseline_pred
    )
)


adapted_matrix = (
    create_confusion_matrix(
        adapted_true,
        adapted_pred
    )
)


# ============================================================
# 11. VALIDATE TOTALS
# ============================================================

if baseline_matrix.values.sum() != 134:

    raise RuntimeError(
        "Baseline confusion matrix does not "
        "contain exactly 134 predictions."
    )


if adapted_matrix.values.sum() != 134:

    raise RuntimeError(
        "Adapted confusion matrix does not "
        "contain exactly 134 predictions."
    )


# ============================================================
# 12. SAVE RAW MATRICES
# ============================================================

baseline_csv = (
    FINAL_DIR
    / "baseline_plantdoc_confusion_matrix.csv"
)

adapted_csv = (
    FINAL_DIR
    / "adapted_plantdoc_confusion_matrix.csv"
)


baseline_matrix.to_csv(
    baseline_csv
)

adapted_matrix.to_csv(
    adapted_csv
)


# ============================================================
# 13. NORMALIZED MATRICES
# ============================================================

def normalize_matrix(matrix_df):

    matrix = (
        matrix_df
        .to_numpy(dtype=float)
    )


    row_totals = (
        matrix.sum(
            axis=1,
            keepdims=True
        )
    )


    normalized = np.divide(
        matrix,
        row_totals,

        out=np.zeros_like(matrix),

        where=row_totals != 0
    )


    return pd.DataFrame(
        normalized,

        index=matrix_df.index,

        columns=matrix_df.columns
    )


baseline_normalized = (
    normalize_matrix(
        baseline_matrix
    )
)


adapted_normalized = (
    normalize_matrix(
        adapted_matrix
    )
)


# ============================================================
# 14. SAVE NORMALIZED MATRICES
# ============================================================

baseline_normalized_csv = (
    FINAL_DIR
    / "baseline_plantdoc_confusion_matrix_normalized.csv"
)

adapted_normalized_csv = (
    FINAL_DIR
    / "adapted_plantdoc_confusion_matrix_normalized.csv"
)


baseline_normalized.to_csv(
    baseline_normalized_csv
)

adapted_normalized.to_csv(
    adapted_normalized_csv
)


# ============================================================
# 15. SHORT CLASS NAMES FOR GRAPH
# ============================================================

SHORT_NAMES = {

    "Apple___Apple_scab":
        "Apple Scab",

    "Apple___Cedar_apple_rust":
        "Apple Cedar Rust",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        "Corn Gray Leaf Spot",

    "Corn_(maize)___Common_rust_":
        "Corn Common Rust",

    "Grape___Black_rot":
        "Grape Black Rot",

    "Pepper,_bell___Bacterial_spot":
        "Pepper Bacterial Spot",

    "Potato___Early_blight":
        "Potato Early Blight",

    "Potato___Late_blight":
        "Potato Late Blight",

    "Squash___Powdery_mildew":
        "Squash Powdery Mildew",

    "Tomato___Bacterial_spot":
        "Tomato Bacterial Spot",

    "Tomato___Early_blight":
        "Tomato Early Blight",

    "Tomato___Late_blight":
        "Tomato Late Blight",

    "Tomato___Leaf_Mold":
        "Tomato Leaf Mold",

    "Tomato___Septoria_leaf_spot":
        "Tomato Septoria",

    "Tomato___Tomato_mosaic_virus":
        "Tomato Mosaic Virus",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "Tomato Yellow Leaf Curl Virus",

    OTHER_LABEL:
        "Other",
}


# ============================================================
# 16. PLOT FUNCTION
# ============================================================

def plot_confusion_matrix(
    matrix_df,
    output_path,
    title,
    normalized=False,
):

    display_matrix = (
        matrix_df.to_numpy()
    )


    if normalized:

        display_values = (
            display_matrix
            * 100
        )

        value_format = ".1f"

        colorbar_label = (
            "Percentage (%)"
        )

    else:

        display_values = (
            display_matrix
        )

        value_format = "d"

        colorbar_label = (
            "Number of Images"
        )


    y_labels = [
        SHORT_NAMES[label]
        for label
        in matrix_df.index
    ]


    x_labels = [
        SHORT_NAMES[label]
        for label
        in matrix_df.columns
    ]


    fig, ax = plt.subplots(
        figsize=(15, 11)
    )


    image = ax.imshow(
        display_values,

        aspect="auto"
    )


    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        colorbar_label
    )


    ax.set_xticks(
        range(
            len(x_labels)
        )
    )

    ax.set_yticks(
        range(
            len(y_labels)
        )
    )


    ax.set_xticklabels(
        x_labels,

        rotation=70,

        ha="right"
    )


    ax.set_yticklabels(
        y_labels
    )


    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "True Class"
    )


    ax.set_title(
        title
    )


    # --------------------------------------------------------
    # Add values inside cells
    # --------------------------------------------------------

    for row in range(
        display_values.shape[0]
    ):

        for col in range(
            display_values.shape[1]
        ):

            value = (
                display_values[
                    row,
                    col
                ]
            )


            if normalized:

                text = (
                    f"{value:.1f}"
                )

            else:

                text = (
                    f"{int(value)}"
                )


            ax.text(
                col,
                row,
                text,

                ha="center",

                va="center",

                fontsize=8
            )


    plt.tight_layout()


    plt.savefig(
        output_path,

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 17. GENERATE BASELINE CONFUSION MATRICES
# ============================================================

baseline_graph = (
    FINAL_DIR
    / "baseline_plantdoc_confusion_matrix.png"
)

baseline_normalized_graph = (
    FINAL_DIR
    / "baseline_plantdoc_confusion_matrix_normalized.png"
)


plot_confusion_matrix(
    baseline_matrix,

    baseline_graph,

    "PlantDoc Confusion Matrix - "
    "Original EfficientNetV2-B0",

    normalized=False
)


plot_confusion_matrix(
    baseline_normalized,

    baseline_normalized_graph,

    "PlantDoc Normalized Confusion Matrix - "
    "Original EfficientNetV2-B0",

    normalized=True
)


# ============================================================
# 18. GENERATE ADAPTED CONFUSION MATRICES
# ============================================================

adapted_graph = (
    FINAL_DIR
    / "adapted_plantdoc_confusion_matrix.png"
)

adapted_normalized_graph = (
    FINAL_DIR
    / "adapted_plantdoc_confusion_matrix_normalized.png"
)


plot_confusion_matrix(
    adapted_matrix,

    adapted_graph,

    "PlantDoc Confusion Matrix - "
    "Domain-Adapted EfficientNetV2-B0",

    normalized=False
)


plot_confusion_matrix(
    adapted_normalized,

    adapted_normalized_graph,

    "PlantDoc Normalized Confusion Matrix - "
    "Domain-Adapted EfficientNetV2-B0",

    normalized=True
)


# ============================================================
# 19. CREATE SUMMARY OF CORRECT PREDICTIONS
# ============================================================

summary_rows = []


for class_name in PLANTDOC_CLASSES:

    baseline_total = int(
        baseline_matrix.loc[
            class_name
        ].sum()
    )


    adapted_total = int(
        adapted_matrix.loc[
            class_name
        ].sum()
    )


    baseline_correct = int(
        baseline_matrix.loc[
            class_name,
            class_name
        ]
    )


    adapted_correct = int(
        adapted_matrix.loc[
            class_name,
            class_name
        ]
    )


    baseline_accuracy = (
        baseline_correct
        / baseline_total
        * 100
        if baseline_total > 0
        else 0
    )


    adapted_accuracy = (
        adapted_correct
        / adapted_total
        * 100
        if adapted_total > 0
        else 0
    )


    summary_rows.append(
        {
            "class_name":
                class_name,

            "test_images":
                baseline_total,

            "baseline_correct":
                baseline_correct,

            "baseline_accuracy_percent":
                baseline_accuracy,

            "adapted_correct":
                adapted_correct,

            "adapted_accuracy_percent":
                adapted_accuracy,

            "improvement_percentage_points":
                (
                    adapted_accuracy
                    - baseline_accuracy
                ),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df = summary_df.sort_values(
    "improvement_percentage_points",
    ascending=False
)


summary_csv = (
    FINAL_DIR
    / "confusion_matrix_comparison_summary.csv"
)


summary_df.to_csv(
    summary_csv,
    index=False
)


# ============================================================
# 20. PRINT IMPORTANT CONFUSIONS
# ============================================================

def print_top_confusions(
    matrix_df,
    model_name,
    number=10
):

    rows = []


    for true_class in PLANTDOC_CLASSES:

        for predicted_class in matrix_df.columns:

            if (
                predicted_class
                == true_class
            ):

                continue


            count = int(
                matrix_df.loc[
                    true_class,
                    predicted_class
                ]
            )


            if count > 0:

                rows.append(
                    {
                        "true":
                            true_class,

                        "predicted":
                            predicted_class,

                        "count":
                            count,
                    }
                )


    confusion_df = pd.DataFrame(
        rows
    )


    if confusion_df.empty:

        print(
            f"\n{model_name}: "
            "No incorrect predictions."
        )

        return


    confusion_df = (
        confusion_df
        .sort_values(
            "count",
            ascending=False
        )
        .head(number)
    )


    print()
    print(
        "-" * 75
    )

    print(
        f"TOP CONFUSIONS - {model_name}"
    )

    print(
        "-" * 75
    )


    for _, row in confusion_df.iterrows():

        print(
            f"{SHORT_NAMES[row['true']]} "
            f"-> "
            f"{SHORT_NAMES[row['predicted']]} "
            f": "
            f"{row['count']} image(s)"
        )


print_top_confusions(
    baseline_matrix,
    "ORIGINAL MODEL"
)


print_top_confusions(
    adapted_matrix,
    "DOMAIN-ADAPTED MODEL"
)


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print()
print("=" * 75)
print("CONFUSION MATRIX ANALYSIS COMPLETE")
print("=" * 75)

print()

print(
    "Baseline matrix:"
)

print(
    baseline_graph
)

print()

print(
    "Baseline normalized matrix:"
)

print(
    baseline_normalized_graph
)

print()

print(
    "Adapted matrix:"
)

print(
    adapted_graph
)

print()

print(
    "Adapted normalized matrix:"
)

print(
    adapted_normalized_graph
)

print()

print(
    "Comparison CSV:"
)

print(
    summary_csv
)

print()

print(
    "All files saved in:"
)

print(
    FINAL_DIR
)

print()

print("=" * 75)