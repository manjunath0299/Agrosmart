# ============================================================
# AgroSmart - FastAPI Backend
# ============================================================

from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from ai.disease.disease_service import DiseaseDetectionService

from .severity_engine import severity_engine
from .treatment_engine import treatment_engine
from .mqtt_service import mqtt_service


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="AgroSmart AI Backend",
    description=(
        "AI-powered crop disease detection, "
        "severity estimation, treatment recommendation "
        "and MQTT-based smart spraying system."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SERVICES
# ============================================================

disease_service = DiseaseDetectionService()


# ============================================================
# REQUEST MODELS
# ============================================================

class SprayRequest(BaseModel):
    crop: str
    disease: str
    severity_level: str
    device_id: str
    approved: bool


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(image_bytes: bytes) -> Image.Image:
    """
    Convert uploaded image bytes into a PIL RGB image.
    """

    try:
        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        return image

    except UnidentifiedImageError:
        raise ValueError(
            "Uploaded file is not a valid image."
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "success": True,
        "service": "AgroSmart AI Backend",
        "version": "1.0.0",
        "status": "running",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "success": True,
        "backend": "online",
        "disease_model": "loaded",
        "severity_model": (
            "loaded"
            if severity_engine.model is not None
            else "not_loaded"
        ),
        "treatment_engine": "loaded",
        "mqtt": mqtt_service.health(),
    }


# ============================================================
# MQTT HEALTH
# ============================================================

@app.get("/mqtt/health")
def mqtt_health():
    return mqtt_service.health()


# ============================================================
# DISEASE ANALYSIS
# ============================================================

@app.post("/analyze-disease")
async def analyze_disease(
    file: UploadFile = File(...)
):
    """
    Run only disease classification.
    """

    try:

        image_bytes = await file.read()

        image = load_image(
            image_bytes
        )

        result = disease_service.predict_pil(
            image
        )

        return {
            "success": True,
            "result": result,
        }

    except ValueError as exc:

        return {
            "success": False,
            "status": "INVALID_IMAGE",
            "message": str(exc),
        }

    except Exception as exc:

        return {
            "success": False,
            "status": "DISEASE_ANALYSIS_ERROR",
            "message": str(exc),
        }


# ============================================================
# SEVERITY ANALYSIS
# ============================================================

@app.post("/analyze-severity")
async def analyze_severity(
    file: UploadFile = File(...)
):
    """
    Run only severity estimation.
    """

    try:

        image_bytes = await file.read()

        image = load_image(
            image_bytes
        )

        result = severity_engine.predict(
            image
        )

        return {
            "success": True,
            "result": result,
        }

    except ValueError as exc:

        return {
            "success": False,
            "status": "INVALID_IMAGE",
            "message": str(exc),
        }

    except Exception as exc:

        return {
            "success": False,
            "status": "SEVERITY_ANALYSIS_ERROR",
            "message": str(exc),
        }


# ============================================================
# COMPLETE AI ANALYSIS
# ============================================================

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...)
):
    """
    Complete AgroSmart pipeline:

        Image
          ↓
        Disease
          ↓
        Confidence gate
          ↓
        Severity
          ↓
        Treatment lookup
          ↓
        Spray eligibility
    """

    try:

        # ------------------------------------------------------
        # READ IMAGE
        # ------------------------------------------------------

        image_bytes = await file.read()

        if not image_bytes:
            return {
                "success": False,
                "status": "EMPTY_FILE",
                "message": "No image was uploaded.",
            }

        image = load_image(
            image_bytes
        )

        # ------------------------------------------------------
        # DISEASE DETECTION
        # ------------------------------------------------------

        disease_result = disease_service.predict_pil(
            image
        )

        # ------------------------------------------------------
        # CHECK DISEASE RESULT
        # ------------------------------------------------------

        if not disease_result:

            return {
                "success": False,
                "analysis_status": "DISEASE_ANALYSIS_FAILED",
                "message": (
                    "Disease detection did not return a result."
                ),
            }

        disease_status = disease_result.get(
            "status"
        )

        # ------------------------------------------------------
        # LOW CONFIDENCE
        # ------------------------------------------------------

        if disease_status == "LOW_CONFIDENCE":

            return {
                "success": True,
                "analysis_status": "LOW_CONFIDENCE",
                "disease": disease_result,
                "severity": {
                    "available": False,
                },
                "treatment": {
                    "available": False,
                    "reason": (
                        "Disease confidence is below "
                        "the 70% threshold."
                    ),
                },
                "spray": {
                    "allowed": False,
                    "requires_approval": False,
                    "reason": (
                        "Spraying is disabled because "
                        "disease confidence is too low."
                    ),
                },
                "message": (
                    "Please upload a clearer leaf image."
                ),
            }

        # ------------------------------------------------------
        # EXTRACT DISEASE INFORMATION
        # ------------------------------------------------------

        prediction = disease_result.get(
            "prediction"
        )

        if not prediction:

            return {
                "success": False,
                "analysis_status": "NO_PREDICTION",
                "disease": disease_result,
                "message": (
                    "The disease model did not "
                    "return a prediction."
                ),
            }

        disease_name = prediction.get(
            "class_name",
            prediction.get(
                "name",
                ""
            ),
        )

        confidence = prediction.get(
            "confidence",
            0,
        )

        crop = prediction.get(
            "crop",
            "",
        )

        disease_type = prediction.get(
            "type",
            "Unknown",
        )

        # ------------------------------------------------------
        # SEVERITY
        # ------------------------------------------------------

        severity_result = severity_engine.predict(
            image
        )

        severity_available = severity_result.get(
            "severity_available",
            False,
        )

        if not severity_available:

            return {
                "success": True,
                "analysis_status": "SEVERITY_UNAVAILABLE",
                "disease": {
                    "crop": crop,
                    "name": disease_name,
                    "confidence": confidence,
                    "status": disease_status,
                    "type": disease_type,
                },
                "severity": severity_result,
                "treatment": {
                    "available": False,
                    "reason": (
                        "Severity estimation is unavailable."
                    ),
                },
                "spray": {
                    "allowed": False,
                    "requires_approval": False,
                    "reason": (
                        "Spraying requires a valid "
                        "severity estimate."
                    ),
                },
                "message": (
                    "Disease detected, but severity "
                    "could not be estimated."
                ),
            }

        severity_level = severity_result.get(
            "severity_level"
        )

        # ------------------------------------------------------
        # TREATMENT LOOKUP
        # ------------------------------------------------------

        treatment = treatment_engine.find_treatment(
            crop=crop,
            disease=disease_name,
            severity_level=severity_level,
        )

        treatment_available = treatment.get(
            "available",
            False,
        )

        # ------------------------------------------------------
        # SPRAY ELIGIBILITY
        # ------------------------------------------------------

        if treatment_available:

            spray_allowed = True

            spray_reason = (
                "Validated treatment is available. "
                "User approval is required before spraying."
            )

        else:

            spray_allowed = False

            spray_reason = (
                treatment.get(
                    "reason",
                    "Treatment is unavailable.",
                )
            )

        # ------------------------------------------------------
        # COMPLETE RESULT
        # ------------------------------------------------------

        return {
            "success": True,
            "analysis_status": "ANALYSIS_COMPLETE",

            "disease": {
                "crop": crop,
                "name": disease_name,
                "confidence": confidence,
                "status": disease_status,
                "type": disease_type,
            },

            "severity": {
                "available": True,
                "percentage": severity_result.get(
                    "severity"
                ),
                "level": severity_level,
                "leaf_pixels": severity_result.get(
                    "leaf_pixels"
                ),
                "disease_pixels": severity_result.get(
                    "disease_pixels"
                ),
            },

            "treatment": treatment,

            "spray": {
                "allowed": spray_allowed,
                "requires_approval": (
                    spray_allowed
                ),
                "reason": spray_reason,
            },

            "message": (
                "Disease, severity and treatment "
                "validation analysis completed successfully."
            ),
        }

    except ValueError as exc:

        return {
            "success": False,
            "analysis_status": "INVALID_IMAGE",
            "message": str(exc),
        }

    except Exception as exc:

        return {
            "success": False,
            "analysis_status": "ANALYSIS_ERROR",
            "message": str(exc),
        }


# ============================================================
# SPRAY REQUEST
# ============================================================

@app.post("/spray")
def spray(
    request: SprayRequest
):
    """
    Authorize and send a spray command.

    IMPORTANT:
    The frontend does NOT provide bottle/product/amount.
    These values are retrieved from the server-side
    treatment database.
    """

    # ----------------------------------------------------------
    # USER APPROVAL
    # ----------------------------------------------------------

    if request.approved is not True:

        return {
            "success": False,
            "status": "SPRAY_NOT_APPROVED",
            "executed": False,
            "message": (
                "User approval is required "
                "before spraying."
            ),
        }

    # ----------------------------------------------------------
    # DEVICE VALIDATION
    # ----------------------------------------------------------

    device_id = request.device_id.strip()

    if not device_id:

        return {
            "success": False,
            "status": "INVALID_DEVICE",
            "executed": False,
            "message": (
                "A valid device_id is required."
            ),
        }

    # ----------------------------------------------------------
    # SERVER-SIDE TREATMENT LOOKUP
    # ----------------------------------------------------------

    treatment = treatment_engine.find_treatment(
        crop=request.crop,
        disease=request.disease,
        severity_level=request.severity_level,
    )

    # ----------------------------------------------------------
    # TREATMENT UNAVAILABLE
    # ----------------------------------------------------------

    if not treatment.get("available"):

        return {
            "success": False,
            "status": "TREATMENT_UNAVAILABLE",
            "executed": False,
            "treatment": treatment,
            "message": (
                "Spray blocked because a validated "
                "treatment is not available."
            ),
        }

    # ----------------------------------------------------------
    # PHYSICAL EXECUTION SAFETY GATE
    # ----------------------------------------------------------
    #
    # Your current treatment database is DEMO_ONLY.
    #
    # Therefore physical execution remains disabled.
    #
    # We will later create a separate simulation execution
    # path so the simulated ESP32 can be tested safely.
    # ----------------------------------------------------------

    if treatment.get(
        "execution_enabled"
    ) is not True:

        return {
            "success": True,
            "status": "SIMULATION_ONLY",
            "executed": False,
            "device_id": device_id,
            "treatment": treatment,
            "message": (
                "Treatment is configured for "
                "demonstration only. Physical spray "
                "execution is disabled."
            ),
        }

    # ----------------------------------------------------------
    # MQTT COMMAND
    # ----------------------------------------------------------

    try:

        mqtt_result = (
            mqtt_service.publish_spray_command(
                device_id=device_id,
                bottle=int(
                    treatment["bottle"]
                ),
                target_volume_ml=float(
                    treatment["amount_ml"]
                ),
            )
        )

    except Exception as exc:

        return {
            "success": False,
            "status": "MQTT_ERROR",
            "executed": False,
            "device_id": device_id,
            "message": str(exc),
        }

    # ----------------------------------------------------------
    # COMMAND SENT
    # ----------------------------------------------------------

    return {
        "success": True,
        "status": "SPRAY_COMMAND_SENT",
        "executed": False,
        "device_id": device_id,
        "command": mqtt_result,
        "treatment": treatment,
        "message": (
            "Spray command successfully sent "
            "to the device."
        ),
    }