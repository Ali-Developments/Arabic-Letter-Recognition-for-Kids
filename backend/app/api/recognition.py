"""
Recognition API Endpoints
Implements /health, /api/v1/model-info, and /api/v1/recognize-letter.
"""

import time
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from app.ml.labels import ARABIC_LETTERS, NUM_CLASSES
from app.ml.predictor import ArabicPredictor
from app.schemas.recognition import (
    HealthResponse,
    LetterPrediction,
    ModelInfoResponse,
    RecognitionResponse,
)

router = APIRouter()

# Instantiate predictor instance
predictor = ArabicPredictor()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check endpoint."""
    return HealthResponse(status="ok")


@router.get("/api/v1/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Returns metadata about the active Arabic character recognition model."""
    return ModelInfoResponse(
        model="arabic-character-classifier",
        version=predictor.model_version,
        classes=NUM_CLASSES,
        target_letters=ARABIC_LETTERS,
        confidence_threshold=predictor.confidence_manager.acceptance_threshold,
    )


@router.post("/api/v1/recognize-letter", response_model=RecognitionResponse, tags=["Inference"])
async def recognize_letter(image: UploadFile = File(...)):
    """
    Recognize one isolated Arabic handwritten character from an uploaded image.
    Image is processed in-memory without persistent disk storage (Child Privacy Compliance).
    """
    start_time = time.perf_counter()

    # Validate content type
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file type '{image.content_type}' is not a supported image format.",
        )

    # Read binary bytes into RAM
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image payload received.",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read image stream: {str(e)}",
        )

    # Generate unique request trace ID
    import uuid
    request_id = f"req_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    # Check debug camera images environment variable (defaults to false for child privacy compliance)
    import os
    save_debug = os.getenv("DEBUG_CAMERA_IMAGES", "false").lower() in ("true", "1", "yes")

    # Perform prediction
    try:
        pred_dict = predictor.predict_from_bytes(
            image_bytes, top_k=3, request_id=request_id, save_debug=save_debug
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error during character preprocessing and inference: {str(e)}",
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    prediction = LetterPrediction(
        letter=pred_dict["letter"],
        class_id=pred_dict["class_id"],
        confidence=pred_dict["confidence"],
        accepted=pred_dict["accepted"],
        name_en=pred_dict.get("name_en"),
        name_ar=pred_dict.get("name_ar"),
    )

    top_candidates = [
        LetterPrediction(
            letter=c["letter"],
            class_id=c["class_id"],
            confidence=c["confidence"],
            accepted=c["accepted"],
            name_en=c.get("name_en"),
            name_ar=c.get("name_ar"),
        )
        for c in pred_dict.get("top_candidates", [])
    ]

    return RecognitionResponse(
        success=True,
        request_id=request_id,
        prediction=prediction,
        processing_time_ms=round(latency_ms, 2),
        message=pred_dict.get("message"),
        top_candidates=top_candidates,
    )
