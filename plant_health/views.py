from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import logging

from plant_health.services import diagnose_uploaded_image

logger = logging.getLogger(__name__)


@csrf_exempt
def diagnose_plant(request):
    logger.info(f"diagnose_plant called: method={request.method}")
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    uploaded_image = request.FILES.get("image")
    logger.info(f"Image in files: {bool(uploaded_image)}")
    if not uploaded_image:
        logger.warning("No image provided")
        return JsonResponse(
            {"error": "Upload an image file using the 'image' field."},
            status=400,
        )

    try:
        logger.info(f"Starting diagnosis for {uploaded_image.name}")
        diagnosis = diagnose_uploaded_image(uploaded_image)
        logger.info(f"Diagnosis status: {diagnosis.get('status')}")
        status_code = 200 if diagnosis.get("status") in {
            "ok",
            "uncertain",
            "reupload",
            "model_not_ready",
            "unsupported_crop",
            "invalid_subject",
        } else 400
        return JsonResponse(diagnosis, status=status_code)
    except Exception as exc:
        logger.error(f"Error in diagnose_plant: {exc}", exc_info=True)
        return JsonResponse(
            {"status": "error", "message": f"Error processing image: {str(exc)}"},
            status=500,
        )
