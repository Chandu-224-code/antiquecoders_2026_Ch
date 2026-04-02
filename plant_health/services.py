"""Services for turning uploaded images into farmer-friendly diagnosis responses."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from PIL import Image, UnidentifiedImageError

from model.inference import ModelNotReadyError, PlantDiseasePredictor
from plant_health.knowledge import get_disease_guidance

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_predictor() -> PlantDiseasePredictor:
    return PlantDiseasePredictor()


def _confidence_percent(score: float | None) -> str:
    if score is None:
        return "Unknown"
    return f"{round(float(score) * 100)}%"


def enrich_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    disease_code = prediction.get("disease_code")
    guidance = get_disease_guidance(disease_code)

    if prediction["status"] in {"ok", "uncertain"}:
        explanation = guidance.get(
            "explanation",
            "The model completed the analysis, but there is no guidance text for this label yet.",
        )
        if disease_code == "healthy":
            explanation = "The uploaded plant image appears healthy with no visible disease symptoms."

        prediction = {
            **prediction,
            "confidence_percent": _confidence_percent(prediction.get("confidence")),
            "explanation": explanation,
            "treatment": guidance.get("treatment", []),
            "organic_treatment": guidance.get("organic_treatment", []),
            "chemical_treatment": guidance.get("chemical_treatment", []),
            "prevention": guidance.get("prevention", []),
            "monitoring": guidance.get("monitoring", ""),
            "safety_note": guidance.get("safety_note", ""),
        }

    return prediction


def diagnose_uploaded_image(uploaded_file) -> dict[str, Any]:
    try:
        # Ensure file pointer is at the beginning
        uploaded_file.seek(0)
        logger.info(f"Opening image file: {uploaded_file.name}")
        with Image.open(uploaded_file) as image:
            logger.info(f"Image opened successfully: {image.size}")
            prediction = get_predictor().predict_image(image.convert("RGB"))
            logger.info(f"Prediction completed: {prediction.get('status')}")
            
            # Add note about validation status
            validation_info = prediction.get("validation", {})
            if validation_info.get("status") == "skipped":
                logger.warning("Plant validation was skipped - validation model not available")
    except UnidentifiedImageError:
        logger.error("File is not a readable image")
        return {
            "status": "error",
            "message": "The uploaded file is not a readable image.",
        }
    except ModelNotReadyError as exc:
        logger.error(f"Disease model not ready: {exc}")
        return {
            "status": "model_not_ready",
            "message": "The disease detection model is not trained yet.",
            "reason": str(exc),
        }

    return enrich_prediction(prediction)


def format_prediction_for_chat(prediction: dict[str, Any]) -> str:
    status = prediction.get("status")

    if status == "model_not_ready":
        return (
            "The upload pipeline is ready, but the model weights are missing. "
            "Train the `validation` and `disease` models first, then upload the image again."
        )

    if status == "error":
        return prediction.get("message", "The image could not be processed.")

    if status == "invalid_subject":
        return "Please upload only plant leaf images. Non-plant images are not supported."

    if status == "reupload":
        reason = prediction.get("reason", "Image quality is too low.")
        suggestions = prediction.get("suggestions", [])
        suggestion_text = " ".join(suggestions[:2])
        return f"Please upload a clearer image. {reason} {suggestion_text}".strip()

    top_predictions = prediction.get("top_predictions", [])
    alternatives = ", ".join(
        f"{item['label']} ({_confidence_percent(item['score'])})"
        for item in top_predictions[1:3]
    )

    message = (
        f"Diagnosis: {prediction.get('disease', 'Unknown')} "
        f"with confidence {prediction.get('confidence_percent', 'Unknown')}. "
        f"{prediction.get('explanation', '')}"
    )

    if status == "uncertain":
        message += " The model is not fully confident, so please retake the image in daylight if possible."

    if alternatives:
        message += f" Other likely options: {alternatives}."

    organic = prediction.get("organic_treatment", [])
    chemical = prediction.get("chemical_treatment", [])
    if organic:
        message += f" Organic option: {organic[0]}."
    if chemical:
        message += f" Chemical option: {chemical[0]}."

    return message.strip()
