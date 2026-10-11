from typing import List, Optional
from pydantic import BaseModel, Field


class LetterPrediction(BaseModel):
    letter: Optional[str] = Field(None, description="Recognized Arabic letter (or None if rejected)")
    class_id: Optional[int] = Field(None, ge=0, le=27, description="Class index (0 to 27, or None if rejected)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax confidence score")
    accepted: bool = Field(..., description="Whether prediction meets acceptance threshold")
    name_en: Optional[str] = Field(None, description="English name of the letter")
    name_ar: Optional[str] = Field(None, description="Arabic name of the letter")


class RecognitionResponse(BaseModel):
    success: bool = Field(..., description="True if inference succeeded")
    request_id: Optional[str] = Field(None, description="Unique trace ID for this recognition request")
    prediction: Optional[LetterPrediction] = Field(None, description="Prediction details")
    processing_time_ms: float = Field(..., description="Processing and inference latency in milliseconds")
    message: Optional[str] = Field(None, description="Status or guidance message (e.g. if rejected)")
    top_candidates: Optional[List[LetterPrediction]] = Field(None, description="Top N candidate predictions")


class ModelInfoResponse(BaseModel):
    model: str = Field(..., description="Model architecture/identifier")
    version: str = Field(..., description="Model version")
    classes: int = Field(default=28, description="Number of target classes")
    target_letters: List[str] = Field(..., description="List of supported Arabic letters")
    confidence_threshold: float = Field(..., description="Configured acceptance threshold")


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Service health status")
