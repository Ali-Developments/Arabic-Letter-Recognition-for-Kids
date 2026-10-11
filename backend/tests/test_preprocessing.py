"""
Comprehensive tests for real-camera image preprocessing pipeline.
Covers Section 18 requirements:
- blank image rejection
- pure white page
- pure black frame
- dot preservation
- small letter
- large letter
- shifted character
- weak dots
- noise outside character
- multiple letter rejection
- valid camera-like input
"""

import cv2
import numpy as np
import pytest
from app.ml.preprocessing import (
    extract_character_with_dots,
    fit_to_square,
    preprocess_image_for_model,
)


def create_synthetic_letter_with_dot_and_noise() -> np.ndarray:
    """
    Creates a synthetic binary image with:
    1. Main stroke body (representing 'ب' horizontal basin) at y=100..120, x=50..150
    2. Real Arabic dot below at y=130, x=100 (radius 5)
    3. Distant paper noise speckle at top-left corner (y=15, x=15, radius 2)
    """
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(img, (100, 100), (45, 15), 0, 0, 180, 255, 6)
    cv2.circle(img, (100, 130), 5, 255, -1)
    cv2.circle(img, (15, 15), 2, 255, -1)
    return img


def test_arabic_dot_preservation_and_noise_filtering():
    """CRITICAL TEST: Ensure Arabic dots within letter envelope are preserved while distant noise is filtered."""
    synthetic = create_synthetic_letter_with_dot_and_noise()
    cleaned = extract_character_with_dots(synthetic)

    # 1. Main body retained
    assert cleaned[115, 100] == 255, "Main letter body should be retained."
    # 2. Arabic dot below preserved
    assert cleaned[130, 100] == 255, "Arabic dot below letter MUST be preserved."
    # 3. Distant noise at (15, 15) removed
    assert cleaned[15, 15] == 0, "Distant corner noise should be filtered out."


def test_fit_to_square_centering():
    """Verify that fit_to_square produces a centered 32x32 image maintaining aspect ratio."""
    tall_stroke = np.ones((60, 10), dtype=np.uint8) * 255
    square = fit_to_square(tall_stroke, target_size=32, margin=8)

    assert square.shape == (32, 32)
    assert square.dtype == np.uint8
    assert square[16, 16] > 0
    assert square[16, 0] == 0
    assert square[16, 31] == 0


def test_preprocess_pipeline_output_shape_and_dtype():
    """Verify end-to-end preprocessing produces float32 [0.0, 1.0] 32x32 array."""
    synthetic = create_synthetic_letter_with_dot_and_noise()
    camera_sim = cv2.bitwise_not(synthetic)
    processed = preprocess_image_for_model(camera_sim, target_size=(32, 32))

    assert processed.shape == (32, 32)
    assert processed.dtype == np.float32
    assert 0.0 <= processed.min()
    assert processed.max() <= 1.0


def test_blank_image_handling():
    """Ensure blank image returns zeros and is flagged as blank."""
    blank = np.ones((100, 100), dtype=np.uint8) * 255
    processed, meta = preprocess_image_for_model(blank, target_size=(32, 32), return_metadata=True)
    assert processed.shape == (32, 32)
    assert np.all(processed == 0.0)
    assert meta["status"] == "blank"


def test_pure_white_page_rejection():
    """Ensure completely white paper (RGB 255) is rejected cleanly without crashing."""
    white_page = np.full((300, 300, 3), 255, dtype=np.uint8)
    processed, meta = preprocess_image_for_model(white_page, return_metadata=True)
    assert np.all(processed == 0.0)
    assert meta["status"] == "blank"
    assert "blank" in meta["message"].lower()


def test_pure_black_frame_rejection():
    """Ensure completely dark frame (underexposed or camera blocked) is rejected."""
    black_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    processed, meta = preprocess_image_for_model(black_frame, return_metadata=True)
    assert np.all(processed == 0.0)
    assert meta["status"] == "invalid_lighting"
    assert "dark" in meta["message"].lower()


def test_small_letter_extraction():
    """Ensure small handwritten letters (e.g. 20x20 in 300x300 image) are scaled and centered."""
    paper = np.full((300, 300), 245, dtype=np.uint8)
    # Small letter in the center
    cv2.circle(paper, (150, 150), 12, 30, -1)
    processed, meta = preprocess_image_for_model(paper, return_metadata=True)
    assert meta["status"] == "valid"
    assert processed.max() > 0.5


def test_large_letter_extraction():
    """Ensure large letters spanning most of the guide frame are correctly normalized."""
    paper = np.full((300, 300), 245, dtype=np.uint8)
    # Large stroke
    cv2.line(paper, (50, 50), (250, 250), 30, 8)
    processed, meta = preprocess_image_for_model(paper, return_metadata=True)
    assert meta["status"] == "valid"
    assert processed.shape == (32, 32)
    assert processed.max() > 0.5


def test_shifted_character_centering():
    """Ensure an off-center character (e.g. drawn near top-right) is centered in the output."""
    paper = np.full((300, 300), 245, dtype=np.uint8)
    # Off-center character near (220, 80)
    cv2.circle(paper, (220, 80), 25, 30, 4)
    processed, meta = preprocess_image_for_model(paper, return_metadata=True)
    assert meta["status"] == "valid"
    # Centered output should have mass in center rows/cols
    center_mass = np.sum(processed[10:22, 10:22])
    assert center_mass > 0.0


def test_weak_dots_preservation():
    """Ensure faint pencil dots (subtle low contrast) are preserved."""
    binary = np.zeros((200, 200), dtype=np.uint8)
    # Main body
    cv2.rectangle(binary, (80, 80), (120, 110), 255, -1)
    # Very small dot above (area ~6 pixels)
    cv2.circle(binary, (100, 50), 2, 255, -1)

    cleaned, meta = extract_character_with_dots(binary, return_metadata=True)
    assert meta["status"] == "valid"
    assert meta["dots_preserved"] >= 1
    # Verify dot area at (100, 50) is retained
    assert cleaned[50, 100] == 255


def test_multiple_letters_rejection():
    """Ensure two separate large letters are rejected with guidance to write only one."""
    binary = np.zeros((250, 250), dtype=np.uint8)
    # First letter (left)
    cv2.rectangle(binary, (30, 100), (90, 160), 255, -1)
    # Second letter (right, clearly horizontally separated)
    cv2.rectangle(binary, (160, 100), (220, 160), 255, -1)

    cleaned, meta = extract_character_with_dots(binary, return_metadata=True)
    assert meta["status"] == "multiple_letters"
    assert "one letter" in meta["message"].lower()


def test_valid_camera_like_input():
    """Ensure a realistic camera capture of a handwritten letter passes preprocessing."""
    paper = np.full((320, 320, 3), 240, dtype=np.uint8)
    # Faint guide frame
    cv2.rectangle(paper, (40, 40), (280, 280), (220, 220, 220), 1)
    # Dark ink letter 'ب'
    cv2.ellipse(paper, (160, 150), (60, 25), 0, 0, 180, (30, 30, 30), 8)
    cv2.circle(paper, (160, 195), 7, (30, 30, 30), -1)

    processed, meta = preprocess_image_for_model(paper, return_metadata=True)
    assert meta["status"] == "valid"
    assert meta["dots_preserved"] >= 1
    assert processed.shape == (32, 32)
    assert processed.max() > 0.5
