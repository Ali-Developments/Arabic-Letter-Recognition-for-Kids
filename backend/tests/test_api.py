"""
Integration tests for FastAPI endpoints with Trained ArabicCNN Model.
Validates /health, /api/v1/model-info, and /api/v1/recognize-letter using known test samples.
"""

import io
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from app.main import app
from app.api.recognition import predictor

client = TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed" / "hijja" / "test"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_endpoint():
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "arabic-character-classifier"
    assert "0.1.0-baseline" in data["version"]
    assert data["classes"] == 28
    assert len(data["target_letters"]) == 28
    assert data["confidence_threshold"] == predictor.confidence_manager.acceptance_threshold


def test_recognize_letter_known_test_samples():
    """Verify inference endpoint on 3 known test samples from different classes."""
    test_cases = [
        ("00", "ا", "alef"),
        ("01", "ب", "baa"),
        ("27", "ي", "yaa"),
    ]

    for cid_str, expected_char, name_en in test_cases:
        class_folder = TEST_DATA_DIR / cid_str
        files = list(class_folder.glob("*.png"))
        assert len(files) > 0, f"No test files found for class {cid_str}"

        with open(files[0], "rb") as f:
            img_bytes = f.read()

        response = client.post(
            "/api/v1/recognize-letter",
            files={"image": (f"test_{name_en}.png", img_bytes, "image/png")},
        )
        assert response.status_code == 200, f"Failed for {name_en}: {response.text}"
        data = response.json()

        assert data["success"] is True
        pred = data["prediction"]
        assert pred["letter"] == expected_char, f"Expected {expected_char}, got {pred['letter']}"
        assert pred["class_id"] == int(cid_str)
        assert pred["confidence"] > 0.80, f"Confidence too low: {pred['confidence']}"
        assert pred["accepted"] is True
        assert data["processing_time_ms"] > 0


def test_recognize_letter_endpoint_invalid_file_type():
    buf = io.BytesIO(b"not an image file")
    response = client.post(
        "/api/v1/recognize-letter",
        files={"image": ("test.txt", buf, "text/plain")},
    )
    assert response.status_code == 400
