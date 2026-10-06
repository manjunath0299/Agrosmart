# backend/treatment_engine.py

from pathlib import Path
import json


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "treatment_database.json"


# ============================================================
# TREATMENT ENGINE
# ============================================================

class TreatmentEngine:
    """
    Treatment database lookup engine.

    IMPORTANT:
    This class only reads treatment rules from the JSON database.
    It does not calculate, invent, or modify pesticide dosages.
    """

    def __init__(self):
        self.database = self._load_database()

    # ========================================================
    # LOAD DATABASE
    # ========================================================

    def _load_database(self):
        """
        Load treatment_database.json.
        """

        if not DATABASE_PATH.exists():

            print(
                f"WARNING: Treatment database not found: "
                f"{DATABASE_PATH}"
            )

            return {
                "treatments": []
            }

        try:

            with open(
                DATABASE_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

        except json.JSONDecodeError as exc:

            print(
                "WARNING: Invalid treatment database JSON: "
                f"{exc}"
            )

            return {
                "treatments": []
            }

        except OSError as exc:

            print(
                "WARNING: Could not read treatment database: "
                f"{exc}"
            )

            return {
                "treatments": []
            }

        if not isinstance(data, dict):

            print(
                "WARNING: Treatment database must be "
                "a JSON object."
            )

            return {
                "treatments": []
            }

        if not isinstance(
            data.get("treatments"),
            list
        ):

            print(
                "WARNING: 'treatments' must be a list."
            )

            return {
                "treatments": []
            }

        return data

    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    @staticmethod
    def _normalize(value):
        """
        Normalize crop/disease/severity names.

        Examples:

            Potato - Early blight
            Potato Early Blight
            Potato___Early_blight
            Potato_Early_blight

        become equivalent for lookup.
        """

        if value is None:
            return ""

        value = str(value).strip().lower()

        # PlantVillage separator
        value = value.replace("___", " ")

        # Underscores
        value = value.replace("_", " ")

        # Hyphens
        value = value.replace("-", " ")

        # Remove repeated spaces
        value = " ".join(value.split())

        return value

    # ========================================================
    # FIND TREATMENT
    # ========================================================

    def find_treatment(
        self,
        crop: str,
        disease: str,
        severity_level: str
    ):
        """
        Find treatment using:

            crop
            disease
            severity_level

        The treatment must already exist in the database.
        """

        crop_key = self._normalize(crop)
        disease_key = self._normalize(disease)
        severity_key = self._normalize(severity_level)

        # ----------------------------------------------------
        # INPUT VALIDATION
        # ----------------------------------------------------

        if not crop_key:

            return {
                "available": False,
                "reason": "Crop information is missing."
            }

        if not disease_key:

            return {
                "available": False,
                "reason": "Disease information is missing."
            }

        if not severity_key:

            return {
                "available": False,
                "reason": "Severity level is missing."
            }

        # ----------------------------------------------------
        # DATABASE SEARCH
        # ----------------------------------------------------

        for treatment in self.database.get(
            "treatments",
            []
        ):

            if not isinstance(treatment, dict):
                continue

            database_crop = self._normalize(
                treatment.get("crop")
            )

            database_disease = self._normalize(
                treatment.get("disease")
            )

            database_severity = self._normalize(
                treatment.get("severity_level")
            )

            # ------------------------------------------------
            # MATCH
            # ------------------------------------------------

            if (
                database_crop == crop_key
                and
                database_disease == disease_key
                and
                database_severity == severity_key
            ):

                # --------------------------------------------
                # TREATMENT DISABLED
                # --------------------------------------------

                if treatment.get("available") is not True:

                    return {
                        "available": False,
                        "reason": (
                            "Validated treatment is not "
                            "configured."
                        )
                    }

                # --------------------------------------------
                # REQUIRED FIELDS
                # --------------------------------------------

                required_fields = [
                    "bottle",
                    "product",
                    "amount_ml"
                ]

                missing_fields = [
                    field
                    for field in required_fields
                    if field not in treatment
                ]

                if missing_fields:

                    return {
                        "available": False,
                        "reason": (
                            "Treatment rule is incomplete. "
                            f"Missing: "
                            f"{', '.join(missing_fields)}."
                        )
                    }

                # --------------------------------------------
                # BOTTLE VALIDATION
                # --------------------------------------------

                bottle = treatment.get("bottle")

                if bottle not in [1, 2, 3, 4]:

                    return {
                        "available": False,
                        "reason": (
                            "Treatment rule contains an "
                            "invalid bottle number."
                        )
                    }

                # --------------------------------------------
                # PRODUCT VALIDATION
                # --------------------------------------------

                product = treatment.get("product")

                if (
                    not isinstance(product, str)
                    or not product.strip()
                ):

                    return {
                        "available": False,
                        "reason": (
                            "Treatment rule does not contain "
                            "a valid product."
                        )
                    }

                # --------------------------------------------
                # AMOUNT VALIDATION
                # --------------------------------------------

                amount_ml = treatment.get("amount_ml")

                if (
                    not isinstance(
                        amount_ml,
                        (int, float)
                    )
                    or isinstance(
                        amount_ml,
                        bool
                    )
                    or amount_ml <= 0
                ):

                    return {
                        "available": False,
                        "reason": (
                            "Treatment rule does not contain "
                            "a valid amount."
                        )
                    }

                # --------------------------------------------
                # VALID TREATMENT
                # --------------------------------------------

                return {
                    "available": True,

                    "crop": treatment.get(
                        "crop"
                    ),

                    "disease": treatment.get(
                        "disease"
                    ),

                    "severity_level": treatment.get(
                        "severity_level"
                    ),

                    "bottle": bottle,

                    "product": product,

                    "amount_ml": amount_ml,

                    "mode": treatment.get(
                        "mode",
                        "DEMO_ONLY"
                    ),

                    "execution_enabled": (
                        treatment.get(
                            "execution_enabled",
                            False
                        ) is True
                    )
                }

        # ----------------------------------------------------
        # NO MATCH
        # ----------------------------------------------------

        return {
            "available": False,
            "reason": (
                "Treatment not available for this crop, "
                "disease, and severity level."
            )
        }


# ============================================================
# SINGLE ENGINE INSTANCE
# ============================================================

treatment_engine = TreatmentEngine()