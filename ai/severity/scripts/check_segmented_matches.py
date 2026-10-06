from pathlib import Path
import re


PROJECT_ROOT = Path(r"C:\smart_agriculture")

ANNOTATION_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "results"
    / "sam2_annotations"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "ai"
    / "severity"
    / "dataset"
    / "images"
    / "annotation_pool"
)

SEGMENTED_DIR = (
    PROJECT_ROOT
    / "PlantVillage-Dataset"
    / "raw"
    / "segmented"
    / "Tomato___Early_blight"
)


UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}"
)


def extract_uuid(filename):
    """
    Extract UUID from either:

    UUID___RS_Erly.B 8389_final_masked.jpg

    or

    leaf_2_0_UUID___RS_Erly.B 6336
    """

    # Everything before ___
    prefix = filename.split("___", 1)[0]

    # Case 1:
    # UUID directly at beginning
    match = UUID_PATTERN.match(prefix)

    if match:
        return match.group(0).lower()

    # Case 2:
    # leaf_2_0_UUID
    parts = prefix.split("_")

    if len(parts) >= 4:

        possible_uuid = parts[-1]

        if UUID_PATTERN.match(possible_uuid):

            return possible_uuid.lower()

    return None


def main():

    print("=" * 70)
    print("CHECKING SEGMENTED IMAGE MATCHES")
    print("=" * 70)

    # --------------------------------------------------------
    # Disease masks
    # --------------------------------------------------------

    disease_masks = sorted(
        ANNOTATION_DIR.glob("*_disease_mask.png")
    )

    print()
    print(
        f"Disease masks found: {len(disease_masks)}"
    )

    # --------------------------------------------------------
    # Original images
    # --------------------------------------------------------

    original_files = [
        p
        for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
        }
    ]

    original_lookup = {
        p.stem: p
        for p in original_files
    }

    print(
        f"Original images found: {len(original_files)}"
    )

    # --------------------------------------------------------
    # Segmented images
    # --------------------------------------------------------

    segmented_files = list(
        SEGMENTED_DIR.glob("*_final_masked.jpg")
    )

    print(
        f"Segmented images found: {len(segmented_files)}"
    )

    # Build UUID lookup
    segmented_lookup = {}

    for path in segmented_files:

        uuid = extract_uuid(path.name)

        if uuid is not None:

            segmented_lookup[uuid] = path

    print(
        f"Segmented UUIDs indexed: "
        f"{len(segmented_lookup)}"
    )

    # --------------------------------------------------------
    # Match
    # --------------------------------------------------------

    matched = []

    missing_original = []

    missing_segmented = []

    for mask_path in disease_masks:

        base_name = mask_path.name.replace(
            "_disease_mask.png",
            "",
        )

        # -----------------------------------------------
        # Original
        # -----------------------------------------------

        original_path = original_lookup.get(
            base_name
        )

        if original_path is None:

            missing_original.append(
                base_name
            )

            continue

        # -----------------------------------------------
        # UUID
        # -----------------------------------------------

        uuid = extract_uuid(base_name)

        if uuid is None:

            missing_segmented.append(
                {
                    "name": base_name,
                    "uuid": None,
                }
            )

            continue

        # -----------------------------------------------
        # Segmented
        # -----------------------------------------------

        segmented_path = segmented_lookup.get(
            uuid
        )

        if segmented_path is None:

            missing_segmented.append(
                {
                    "name": base_name,
                    "uuid": uuid,
                }
            )

            continue

        # -----------------------------------------------
        # Success
        # -----------------------------------------------

        matched.append(
            {
                "name": base_name,
                "uuid": uuid,
                "original": original_path,
                "disease_mask": mask_path,
                "segmented": segmented_path,
            }
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MATCHING RESULT")
    print("=" * 70)

    print(
        f"Total disease masks : {len(disease_masks)}"
    )

    print(
        f"Matched successfully : {len(matched)}"
    )

    print(
        f"Missing original     : {len(missing_original)}"
    )

    print(
        f"Missing segmented    : {len(missing_segmented)}"
    )

    # --------------------------------------------------------
    # Show first 10
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FIRST 10 SUCCESSFUL MATCHES")
    print("=" * 70)

    for item in matched[:10]:

        print()
        print("Original:")
        print(
            " ",
            item["original"].name
        )

        print("Disease mask:")
        print(
            " ",
            item["disease_mask"].name
        )

        print("Segmented:")
        print(
            " ",
            item["segmented"].name
        )

    # --------------------------------------------------------
    # Missing
    # --------------------------------------------------------

    if missing_segmented:

        print()
        print("=" * 70)
        print("MISSING SEGMENTED IMAGES")
        print("=" * 70)

        for item in missing_segmented:

            print(
                item["name"]
            )

            print(
                "UUID:",
                item["uuid"]
            )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)

    if (
        len(matched) == len(disease_masks)
        and len(missing_original) == 0
        and len(missing_segmented) == 0
    ):

        print(
            "SUCCESS: ALL 49 IMAGES MATCH."
        )

    else:

        print(
            "WARNING: SOME FILES DID NOT MATCH."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()