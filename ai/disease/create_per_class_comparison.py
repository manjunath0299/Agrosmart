# ============================================================
# create_per_class_comparison.py
#
# Compares disease-wise PlantDoc performance:
# Original model vs Domain-Adapted model
# ============================================================

from pathlib import Path

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

BASELINE_FILE = (
    RESULTS_DIR
    / "baseline_plantdoc"
    / "baseline_plantdoc_per_class_accuracy.csv"
)

ADAPTED_FILE = (
    RESULTS_DIR
    / "domain_adaptation"
    / "plantdoc_per_class_accuracy.csv"
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
# 2. CHECK FILES
# ============================================================

if not BASELINE_FILE.exists():

    raise FileNotFoundError(
        f"Baseline file not found:\n"
        f"{BASELINE_FILE}"
    )


if not ADAPTED_FILE.exists():

    raise FileNotFoundError(
        f"Domain-adapted file not found:\n"
        f"{ADAPTED_FILE}"
    )


# ============================================================
# 3. LOAD RESULTS
# ============================================================

baseline = pd.read_csv(
    BASELINE_FILE
)

adapted = pd.read_csv(
    ADAPTED_FILE
)


print("=" * 75)
print("PER-CLASS PLANTDOC COMPARISON")
print("=" * 75)

print()

print(
    "Baseline rows:",
    len(baseline)
)

print(
    "Adapted rows:",
    len(adapted)
)


# ============================================================
# 4. RENAME ACCURACY COLUMNS
# ============================================================

baseline = baseline[
    [
        "label",
        "class_name",
        "total_images",
        "correct",
        "accuracy",
    ]
].copy()

adapted = adapted[
    [
        "label",
        "class_name",
        "total_images",
        "correct",
        "accuracy",
    ]
].copy()


baseline = baseline.rename(
    columns={
        "total_images":
            "baseline_total",

        "correct":
            "baseline_correct",

        "accuracy":
            "baseline_accuracy",
    }
)


adapted = adapted.rename(
    columns={
        "total_images":
            "adapted_total",

        "correct":
            "adapted_correct",

        "accuracy":
            "adapted_accuracy",
    }
)


# ============================================================
# 5. MERGE
# ============================================================

comparison = pd.merge(
    baseline,
    adapted,

    on=[
        "label",
        "class_name",
    ],

    how="outer"
)


# ============================================================
# 6. CONVERT TO PERCENTAGES
# ============================================================

comparison[
    "baseline_accuracy_percent"
] = (
    comparison[
        "baseline_accuracy"
    ] * 100
)


comparison[
    "adapted_accuracy_percent"
] = (
    comparison[
        "adapted_accuracy"
    ] * 100
)


# ============================================================
# 7. CALCULATE IMPROVEMENT
# ============================================================

comparison[
    "improvement_percentage_points"
] = (
    comparison[
        "adapted_accuracy_percent"
    ]
    -
    comparison[
        "baseline_accuracy_percent"
    ]
)


# ============================================================
# 8. SORT BY IMPROVEMENT
# ============================================================

comparison = comparison.sort_values(
    "improvement_percentage_points",
    ascending=False
)


# ============================================================
# 9. SAVE COMPLETE CSV
# ============================================================

output_csv = (
    FINAL_DIR
    / "plantdoc_per_class_comparison.csv"
)

comparison.to_csv(
    output_csv,
    index=False
)


# ============================================================
# 10. PRINT TABLE
# ============================================================

display_columns = [
    "class_name",
    "baseline_accuracy_percent",
    "adapted_accuracy_percent",
    "improvement_percentage_points",
]


print()

print("=" * 75)
print("DISEASE-WISE ACCURACY COMPARISON")
print("=" * 75)

print()

print(
    comparison[
        display_columns
    ].to_string(
        index=False,
        formatters={
            "baseline_accuracy_percent":
                "{:.2f}%".format,

            "adapted_accuracy_percent":
                "{:.2f}%".format,

            "improvement_percentage_points":
                "{:+.2f}".format,
        }
    )
)


# ============================================================
# 11. GRAPH — PER-CLASS ACCURACY
# ============================================================

# Sort alphabetically for graph readability

plot_df = comparison.sort_values(
    "class_name"
).copy()


classes = plot_df[
    "class_name"
].tolist()


baseline_values = plot_df[
    "baseline_accuracy_percent"
].tolist()


adapted_values = plot_df[
    "adapted_accuracy_percent"
].tolist()


x = range(
    len(classes)
)


width = 0.38


plt.figure(
    figsize=(16, 8)
)


plt.bar(
    [
        i - width / 2
        for i in x
    ],

    baseline_values,

    width,

    label="Original EfficientNetV2-B0"
)


plt.bar(
    [
        i + width / 2
        for i in x
    ],

    adapted_values,

    width,

    label="Domain-Adapted EfficientNetV2-B0"
)


plt.xticks(
    list(x),
    classes,
    rotation=75,
    ha="right"
)


plt.ylabel(
    "Accuracy (%)"
)


plt.xlabel(
    "PlantDoc Disease Class"
)


plt.title(
    "Per-Class PlantDoc Accuracy:\n"
    "Original vs Domain-Adapted Model"
)


plt.ylim(
    0,
    105
)


plt.legend()


plt.tight_layout()


graph_path = (
    FINAL_DIR
    / "plantdoc_per_class_comparison.png"
)


plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 12. TOP IMPROVEMENTS
# ============================================================

top_improvements = comparison.head(
    5
)


print()

print("=" * 75)
print("TOP 5 IMPROVEMENTS")
print("=" * 75)

for _, row in top_improvements.iterrows():

    print(
        f"{row['class_name']}: "
        f"{row['baseline_accuracy_percent']:.2f}% "
        f"-> "
        f"{row['adapted_accuracy_percent']:.2f}% "
        f"("
        f"{row['improvement_percentage_points']:+.2f} pp)"
    )


# ============================================================
# 13. CLASSES WITH NEGATIVE CHANGE
# ============================================================

negative = comparison[
    comparison[
        "improvement_percentage_points"
    ] < 0
]


print()

print("=" * 75)
print("CLASSES WITH DECREASED ACCURACY")
print("=" * 75)

if len(negative) == 0:

    print(
        "No classes showed decreased accuracy."
    )

else:

    for _, row in negative.iterrows():

        print(
            f"{row['class_name']}: "
            f"{row['baseline_accuracy_percent']:.2f}% "
            f"-> "
            f"{row['adapted_accuracy_percent']:.2f}% "
            f"("
            f"{row['improvement_percentage_points']:+.2f} pp)"
        )


# ============================================================
# 14. FINAL
# ============================================================

print()

print("=" * 75)
print("PER-CLASS COMPARISON COMPLETE")
print("=" * 75)

print()

print(
    "CSV:"
)

print(
    output_csv
)

print()

print(
    "Graph:"
)

print(
    graph_path
)

print()

print("=" * 75)