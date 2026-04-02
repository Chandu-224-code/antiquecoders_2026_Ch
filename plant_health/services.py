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


def _disease_stage_from_confidence(confidence: float | None) -> str:
    if confidence is None:
        return "unknown"
    if confidence >= 0.80:
        return "advanced"
    if confidence >= 0.60:
        return "mid"
    return "early"


def _risk_color_from_stage(stage: str) -> str:
    normalized = (stage or "").lower()
    return {
        "advanced": "red",
        "mid": "orange",
        "early": "yellow",
        "healthy": "green",
        "unknown": "gray",
    }.get(normalized, "gray")


def enrich_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    disease_code = prediction.get("disease_code")
    guidance = get_disease_guidance(disease_code)

    disease_confidence = prediction.get("confidence")
    disease_stage = _disease_stage_from_confidence(disease_confidence)

    if disease_code and disease_code.lower() == "healthy":
        disease_stage = "healthy"

    risk_color = _risk_color_from_stage(disease_stage)

    # copy original values to avoid mutation problems
    prediction = {**prediction}
    prediction.update(
        {
            "confidence_score": float(disease_confidence) if disease_confidence is not None else None,
            "disease_type": prediction.get("disease", "unknown"),
            "risk_stage": disease_stage,
            "risk_color": risk_color,
            "pesticides": {
                "organic": guidance.get("organic_treatment", []),
                "fertilizers": guidance.get("chemical_treatment", []),
            },
        }
    )

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

    disease = prediction.get("disease", "Unknown")
    confidence_score = prediction.get("confidence_score")
    confidence_text = (
        f"{round(float(confidence_score) * 100)}%" if confidence_score is not None else "Unknown"
    )
    risk_stage = prediction.get("risk_stage", "unknown")
    risk_color = prediction.get("risk_color", "gray")

    pesticides = prediction.get("pesticides", {})
    organic = pesticides.get("organic", [])
    fertilizers = pesticides.get("fertilizers", [])

    text_lines = [
        "=== Disease ===",
        f"Disease: {disease}",
        "",
        "=== Confidence ===",
        f"Confidence score: {confidence_text}",
        "",
        "=== Risk ===",
        f"Risk stage: {risk_stage}",
        f"Risk color: {risk_color}",
        "",
        "=== Pesticides (treatments) ===",
        f"Organic: {', '.join(organic) if organic else 'No organic recommendations available.'}",
        f"Fertilizers/Chemical: {', '.join(fertilizers) if fertilizers else 'No fertilizer/chemical recommendations available.'}",
    ]

    if status == "uncertain":
        text_lines.append(
            "Note: The model is not fully confident, please retake the image in daylight if possible."
        )

    alternatives = ", ".join(
        f"{item['label']} ({_confidence_percent(item['score'])})"
        for item in prediction.get("top_predictions", [])[1:3]
    )
    if alternatives:
        text_lines.append(f"Other likely options: {alternatives}.")

    explanation = prediction.get("explanation")
    if explanation:
        text_lines.append(f"Explanation: {explanation}")

    return "\n".join(text_lines)

