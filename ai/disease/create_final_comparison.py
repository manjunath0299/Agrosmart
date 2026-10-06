# ============================================================
# create_final_comparison.py
#
# Final comparison:
# Original EfficientNetV2-B0
#              VS
# Domain-Adapted EfficientNetV2-B0
#
# Uses the MATCHED evaluation on the same 134 PlantDoc images.
# ============================================================

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    BASE_DIR
    / "ai"
    / "disease"
    / "results"
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
# 2. VERIFIED EXPERIMENT RESULTS
# ============================================================

# ------------------------------------------------------------
# Original EfficientNetV2-B0
# ------------------------------------------------------------

ORIGINAL_PLANTVILLAGE_ACCURACY = 99.75

ORIGINAL_PLANTDOC_ACCURACY = 14.18

ORIGINAL_PLANTDOC_PRECISION = 35.66

ORIGINAL_PLANTDOC_RECALL = 14.18

ORIGINAL_PLANTDOC_F1 = 18.71

ORIGINAL_PLANTDOC_MACRO_F1 = 19.96


# ------------------------------------------------------------
# Domain-Adapted EfficientNetV2-B0
# ------------------------------------------------------------

ADAPTED_PLANTVILLAGE_ACCURACY = 99.63

ADAPTED_PLANTDOC_ACCURACY = 60.45

ADAPTED_PLANTDOC_PRECISION = 67.26

ADAPTED_PLANTDOC_RECALL = 60.45

ADAPTED_PLANTDOC_F1 = 62.73

ADAPTED_PLANTDOC_MACRO_F1 = 62.96


# ------------------------------------------------------------
# Evaluation information
# ------------------------------------------------------------

PLANTDOC_TEST_IMAGES = 134

PLANTDOC_MAPPED_CLASSES = 16


# ============================================================
# 3. CALCULATE IMPROVEMENTS
# ============================================================

plantdoc_accuracy_improvement = (
    ADAPTED_PLANTDOC_ACCURACY
    - ORIGINAL_PLANTDOC_ACCURACY
)

plantdoc_f1_improvement = (
    ADAPTED_PLANTDOC_F1
    - ORIGINAL_PLANTDOC_F1
)

plantdoc_macro_f1_improvement = (
    ADAPTED_PLANTDOC_MACRO_F1
    - ORIGINAL_PLANTDOC_MACRO_F1
)

plantvillage_accuracy_change = (
    ADAPTED_PLANTVILLAGE_ACCURACY
    - ORIGINAL_PLANTVILLAGE_ACCURACY
)


# Relative improvement in PlantDoc accuracy
relative_accuracy_improvement = (
    (
        ADAPTED_PLANTDOC_ACCURACY
        - ORIGINAL_PLANTDOC_ACCURACY
    )
    / ORIGINAL_PLANTDOC_ACCURACY
) * 100


# ============================================================
# 4. CREATE FINAL COMPARISON DATAFRAME
# ============================================================

comparison_df = pd.DataFrame(
    [
        {
            "Model":
                "Original EfficientNetV2-B0",

            "PlantVillage Test Accuracy (%)":
                ORIGINAL_PLANTVILLAGE_ACCURACY,

            "PlantDoc Accuracy (%)":
                ORIGINAL_PLANTDOC_ACCURACY,

            "PlantDoc Weighted Precision (%)":
                ORIGINAL_PLANTDOC_PRECISION,

            "PlantDoc Weighted Recall (%)":
                ORIGINAL_PLANTDOC_RECALL,

            "PlantDoc Weighted F1 (%)":
                ORIGINAL_PLANTDOC_F1,

            "PlantDoc Macro F1 (%)":
                ORIGINAL_PLANTDOC_MACRO_F1,
        },

        {
            "Model":
                "Domain-Adapted EfficientNetV2-B0",

            "PlantVillage Test Accuracy (%)":
                ADAPTED_PLANTVILLAGE_ACCURACY,

            "PlantDoc Accuracy (%)":
                ADAPTED_PLANTDOC_ACCURACY,

            "PlantDoc Weighted Precision (%)":
                ADAPTED_PLANTDOC_PRECISION,

            "PlantDoc Weighted Recall (%)":
                ADAPTED_PLANTDOC_RECALL,

            "PlantDoc Weighted F1 (%)":
                ADAPTED_PLANTDOC_F1,

            "PlantDoc Macro F1 (%)":
                ADAPTED_PLANTDOC_MACRO_F1,
        },
    ]
)


# ============================================================
# 5. SAVE COMPARISON CSV
# ============================================================

comparison_csv = (
    FINAL_DIR
    / "model_comparison.csv"
)

comparison_df.to_csv(
    comparison_csv,
    index=False
)


# ============================================================
# 6. PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 75)
print("FINAL DISEASE MODEL COMPARISON")
print("=" * 75)

print()

print(
    comparison_df.to_string(
        index=False
    )
)

print()

print("-" * 75)

print(
    "PlantDoc Accuracy Improvement : "
    f"+{plantdoc_accuracy_improvement:.2f} "
    "percentage points"
)

print(
    "PlantDoc Weighted F1 Improvement : "
    f"+{plantdoc_f1_improvement:.2f} "
    "percentage points"
)

print(
    "PlantDoc Macro F1 Improvement : "
    f"+{plantdoc_macro_f1_improvement:.2f} "
    "percentage points"
)

print(
    "PlantVillage Accuracy Change : "
    f"{plantvillage_accuracy_change:+.2f} "
    "percentage points"
)

print(
    "Relative PlantDoc Accuracy Improvement : "
    f"{relative_accuracy_improvement:.2f}%"
)

print("-" * 75)


# ============================================================
# 7. GRAPH 1
# PLANTDOC ACCURACY: ORIGINAL VS ADAPTED
# ============================================================

models = [
    "Original\nEfficientNetV2-B0",
    "Domain-Adapted\nEfficientNetV2-B0",
]

plantdoc_accuracy_values = [
    ORIGINAL_PLANTDOC_ACCURACY,
    ADAPTED_PLANTDOC_ACCURACY,
]


plt.figure(
    figsize=(9, 6)
)

bars = plt.bar(
    models,
    plantdoc_accuracy_values
)

plt.ylabel(
    "Accuracy (%)"
)

plt.xlabel(
    "Model"
)

plt.title(
    "PlantDoc External Accuracy:\n"
    "Original vs Domain-Adapted Model"
)

plt.ylim(
    0,
    100
)


for bar, value in zip(
    bars,
    plantdoc_accuracy_values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        value + 2,

        f"{value:.2f}%",

        ha="center",

        va="bottom",

        fontsize=11
    )


plt.tight_layout()


plantdoc_accuracy_graph = (
    FINAL_DIR
    / "plantdoc_accuracy_comparison.png"
)

plt.savefig(
    plantdoc_accuracy_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. GRAPH 2
# INTERNAL VS EXTERNAL PERFORMANCE
# ============================================================

x_positions = [0, 1]

width = 0.35


plantvillage_values = [
    ORIGINAL_PLANTVILLAGE_ACCURACY,
    ADAPTED_PLANTVILLAGE_ACCURACY,
]


plantdoc_values = [
    ORIGINAL_PLANTDOC_ACCURACY,
    ADAPTED_PLANTDOC_ACCURACY,
]


plt.figure(
    figsize=(10, 6)
)


bars1 = plt.bar(
    [
        x - width / 2
        for x in x_positions
    ],

    plantvillage_values,

    width,

    label="PlantVillage Test"
)


bars2 = plt.bar(
    [
        x + width / 2
        for x in x_positions
    ],

    plantdoc_values,

    width,

    label="PlantDoc External"
)


plt.xticks(
    x_positions,

    [
        "Original\nEfficientNetV2-B0",

        "Domain-Adapted\nEfficientNetV2-B0"
    ]
)


plt.ylabel(
    "Accuracy (%)"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Internal vs External Domain Performance"
)

plt.ylim(
    0,
    105
)

plt.legend()


# Add values above bars

for bars in [
    bars1,
    bars2
]:

    for bar in bars:

        value = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            value + 2,

            f"{value:.2f}%",

            ha="center",

            va="bottom",

            fontsize=10
        )


plt.tight_layout()


internal_external_graph = (
    FINAL_DIR
    / "internal_vs_external_performance.png"
)

plt.savefig(
    internal_external_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. GRAPH 3
# PLANTDOC F1 COMPARISON
# ============================================================

f1_values = [
    ORIGINAL_PLANTDOC_F1,
    ADAPTED_PLANTDOC_F1,
]


plt.figure(
    figsize=(9, 6)
)


bars = plt.bar(
    models,
    f1_values
)


plt.ylabel(
    "Weighted F1-Score (%)"
)

plt.xlabel(
    "Model"
)

plt.title(
    "PlantDoc External Weighted F1:\n"
    "Original vs Domain-Adapted Model"
)

plt.ylim(
    0,
    100
)


for bar, value in zip(
    bars,
    f1_values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        value + 2,

        f"{value:.2f}%",

        ha="center",

        va="bottom",

        fontsize=11
    )


plt.tight_layout()


f1_graph = (
    FINAL_DIR
    / "plantdoc_f1_comparison.png"
)

plt.savefig(
    f1_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. FINAL RESULTS SUMMARY JSON
# ============================================================

summary = {

    "experiment":
        "Plant disease domain adaptation",

    "model":
        "EfficientNetV2-B0",

    "external_dataset":
        "PlantDoc",

    "plantdoc_test_images":
        PLANTDOC_TEST_IMAGES,

    "plantdoc_mapped_classes":
        PLANTDOC_MAPPED_CLASSES,


    "original_model": {

        "plantvillage_accuracy_percent":
            ORIGINAL_PLANTVILLAGE_ACCURACY,

        "plantdoc_accuracy_percent":
            ORIGINAL_PLANTDOC_ACCURACY,

        "plantdoc_weighted_precision_percent":
            ORIGINAL_PLANTDOC_PRECISION,

        "plantdoc_weighted_recall_percent":
            ORIGINAL_PLANTDOC_RECALL,

        "plantdoc_weighted_f1_percent":
            ORIGINAL_PLANTDOC_F1,

        "plantdoc_macro_f1_percent":
            ORIGINAL_PLANTDOC_MACRO_F1,
    },


    "domain_adapted_model": {

        "plantvillage_accuracy_percent":
            ADAPTED_PLANTVILLAGE_ACCURACY,

        "plantdoc_accuracy_percent":
            ADAPTED_PLANTDOC_ACCURACY,

        "plantdoc_weighted_precision_percent":
            ADAPTED_PLANTDOC_PRECISION,

        "plantdoc_weighted_recall_percent":
            ADAPTED_PLANTDOC_RECALL,

        "plantdoc_weighted_f1_percent":
            ADAPTED_PLANTDOC_F1,

        "plantdoc_macro_f1_percent":
            ADAPTED_PLANTDOC_MACRO_F1,
    },


    "improvement": {

        "plantdoc_accuracy_percentage_points":
            plantdoc_accuracy_improvement,

        "plantdoc_weighted_f1_percentage_points":
            plantdoc_f1_improvement,

        "plantdoc_macro_f1_percentage_points":
            plantdoc_macro_f1_improvement,

        "plantvillage_accuracy_percentage_points":
            plantvillage_accuracy_change,

        "relative_plantdoc_accuracy_improvement_percent":
            relative_accuracy_improvement,
    }
}


summary_json = (
    FINAL_DIR
    / "final_results_summary.json"
)


with open(
    summary_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# 11. PAPER-READY TEXT SUMMARY
# ============================================================

paper_summary = f"""
DOMAIN ADAPTATION RESULTS
==========================

Model:
EfficientNetV2-B0

External Dataset:
PlantDoc

PlantDoc Test Set:
{PLANTDOC_TEST_IMAGES} images
{PLANTDOC_MAPPED_CLASSES} mapped classes


Original EfficientNetV2-B0:
--------------------------------
PlantVillage Test Accuracy : {ORIGINAL_PLANTVILLAGE_ACCURACY:.2f}%
PlantDoc Accuracy          : {ORIGINAL_PLANTDOC_ACCURACY:.2f}%
PlantDoc Weighted Precision: {ORIGINAL_PLANTDOC_PRECISION:.2f}%
PlantDoc Weighted Recall   : {ORIGINAL_PLANTDOC_RECALL:.2f}%
PlantDoc Weighted F1       : {ORIGINAL_PLANTDOC_F1:.2f}%
PlantDoc Macro F1          : {ORIGINAL_PLANTDOC_MACRO_F1:.2f}%


Domain-Adapted EfficientNetV2-B0:
--------------------------------
PlantVillage Test Accuracy : {ADAPTED_PLANTVILLAGE_ACCURACY:.2f}%
PlantDoc Accuracy          : {ADAPTED_PLANTDOC_ACCURACY:.2f}%
PlantDoc Weighted Precision: {ADAPTED_PLANTDOC_PRECISION:.2f}%
PlantDoc Weighted Recall   : {ADAPTED_PLANTDOC_RECALL:.2f}%
PlantDoc Weighted F1       : {ADAPTED_PLANTDOC_F1:.2f}%
PlantDoc Macro F1          : {ADAPTED_PLANTDOC_MACRO_F1:.2f}%


Domain Adaptation Improvement:
--------------------------------
PlantDoc Accuracy:
{ORIGINAL_PLANTDOC_ACCURACY:.2f}% -> {ADAPTED_PLANTDOC_ACCURACY:.2f}%

Improvement:
+{plantdoc_accuracy_improvement:.2f} percentage points

PlantDoc Weighted F1:
{ORIGINAL_PLANTDOC_F1:.2f}% -> {ADAPTED_PLANTDOC_F1:.2f}%

Improvement:
+{plantdoc_f1_improvement:.2f} percentage points

PlantVillage Accuracy:
{ORIGINAL_PLANTVILLAGE_ACCURACY:.2f}% -> {ADAPTED_PLANTVILLAGE_ACCURACY:.2f}%

Change:
{plantvillage_accuracy_change:+.2f} percentage points
"""


paper_summary_path = (
    FINAL_DIR
    / "paper_results_summary.txt"
)


with open(
    paper_summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        paper_summary.strip()
    )


# ============================================================
# 12. FINAL OUTPUT
# ============================================================

print()
print("=" * 75)
print("FINAL COMPARISON PACKAGE CREATED")
print("=" * 75)

print()

print(
    f"Comparison CSV:"
)

print(
    comparison_csv
)

print()

print(
    f"PlantDoc accuracy graph:"
)

print(
    plantdoc_accuracy_graph
)

print()

print(
    f"Internal vs external graph:"
)

print(
    internal_external_graph
)

print()

print(
    f"F1 comparison graph:"
)

print(
    f1_graph
)

print()

print(
    f"Summary JSON:"
)

print(
    summary_json
)

print()

print(
    f"Paper summary:"
)

print(
    paper_summary_path
)

print()

print(
    "PlantDoc accuracy improvement: "
    f"+{plantdoc_accuracy_improvement:.2f} "
    "percentage points"
)

print(
    "PlantDoc weighted F1 improvement: "
    f"+{plantdoc_f1_improvement:.2f} "
    "percentage points"
)

print(
    "PlantVillage accuracy change: "
    f"{plantvillage_accuracy_change:+.2f} "
    "percentage points"
)

print()

print("=" * 75)