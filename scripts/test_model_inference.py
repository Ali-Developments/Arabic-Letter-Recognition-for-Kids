"""
Real Model Inference Verification Script.
Ensures backend/app/ml/predictor.py loads the trained checkpoint ml/models/baseline_arabic_cnn_best.pt,
and verifies real inferences across known test samples.
"""

import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.ml.predictor import ArabicPredictor
from app.ml.labels import ARABIC_LETTERS, ID_TO_CHAR, ID_TO_METADATA

TEST_SPLIT_DIR = PROJECT_ROOT / "ml" / "data" / "processed" / "hijja" / "test"
CHECKPOINT_PATH = PROJECT_ROOT / "ml" / "models" / "baseline_arabic_cnn_best.pt"


def run_inference_verification():
    print("=" * 60)
    print("STEP 2: REAL MODEL INFERENCE CHECK WITH TRAINED CHECKPOINT")
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    print("=" * 60)

    predictor = ArabicPredictor(model_path=str(CHECKPOINT_PATH))

    assert predictor.model_loaded is True, "CRITICAL ERROR: Predictor failed to load checkpoint!"
    print(f"Predictor Model Version: {predictor.model_version}")
    print(f"Predictor Loaded Status: {predictor.model_loaded}")

    # Test cases: 5 distinct classes
    test_cases = [
        ("00", "ا", "alef"),
        ("01", "ب", "baa"),
        ("04", "ج", "jeem"),
        ("11", "س", "seen"),
        ("27", "ي", "yaa"),
    ]

    print("\n--- Test Predictions ---")
    correct_count = 0

    for i, (cid_str, expected_char, name_en) in enumerate(test_cases, 1):
        class_folder = TEST_SPLIT_DIR / cid_str
        files = list(class_folder.glob("*.png"))
        assert len(files) > 0, f"No test files found for class {cid_str}"

        test_img_path = files[0]
        with open(test_img_path, "rb") as f:
            img_bytes = f.read()

        pred = predictor.predict_from_bytes(img_bytes)

        true_char = expected_char
        predicted_char = pred["letter"]
        confidence = pred["confidence"]
        accepted = pred["accepted"]

        print(f"Prediction {i}:")
        print(f"True letter:      {true_char} ({name_en})")
        print(f"Predicted letter: {predicted_char} ({pred['name_en']})")
        print(f"Confidence:       {confidence * 100:.2f}%")
        print(f"Accepted:         {accepted}")
        print("-" * 40)

        if predicted_char == true_char:
            correct_count += 1

    print(f"\nVerification Result: {correct_count}/{len(test_cases)} predictions matched ground truth.")
    assert correct_count >= 4, f"Expected at least 4/5 correct, got {correct_count}"
    print("Inference verification PASSED.")


if __name__ == "__main__":
    run_inference_verification()
