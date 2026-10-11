"""
Benchmark and Compare Preprocessing Variants (Step 4, Section 6).
Evaluates:
- Variant A: Current Pipeline (CLAHE + Otsu + CC)
- Variant B: Adaptive Threshold (Illumination Norm + Adaptive Gaussian)
- Variant C: Shadow Robustness (Background Division / Illumination Field + CLAHE + Adaptive Threshold)
Compares dot preservation, shadow resilience, false acceptance rate, and accuracy on camera stress dataset.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path("backend").resolve()))
from app.ml.labels import NUM_CLASSES, ID_TO_CHAR, ID_TO_METADATA
from app.ml.architectures import ArabicCNN

PROJECT_ROOT = Path(".")
MANIFEST_PATH = PROJECT_ROOT / "tests" / "camera_samples" / "manifest.json"
CHECKPOINT_PATH = PROJECT_ROOT / "ml" / "models" / "final_candidate_model.pt"


def preprocess_variant_a(img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Variant A: Grayscale -> CLAHE -> Otsu -> CC Dot Preservation -> Fit to 32x32"""
    meta = {"stages": {}, "status": "valid", "message": None}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    meta["stages"]["gray"] = gray

    # Quick blank / dark check
    if np.mean(gray) > 250 and np.std(gray) < 4:
        return np.zeros((32, 32), dtype=np.float32), {"status": "blank", "message": "No clear handwritten letter detected."}
    if np.mean(gray) < 15:
        return np.zeros((32, 32), dtype=np.float32), {"status": "invalid_lighting", "message": "Image is too dark."}

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm_gray = clahe.apply(gray)
    filtered = cv2.bilateralFilter(norm_gray, 5, 50, 50)

    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Check polarity
    border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    if np.mean(border) > 127:
        binary = cv2.bitwise_not(binary)
    meta["stages"]["binary"] = binary

    cleaned, cc_info = extract_character_and_dots(binary)
    meta["stages"]["cleaned"] = cleaned
    meta.update(cc_info)

    if meta["status"] != "valid":
        return np.zeros((32, 32), dtype=np.float32), meta

    coords = cv2.findNonZero(cleaned)
    if coords is None:
        return np.zeros((32, 32), dtype=np.float32), {"status": "blank", "message": "No clear handwritten letter detected."}

    x, y, bw, bh = cv2.boundingRect(coords)
    cropped = cleaned[y:y + bh, x:x + bw]
    fitted = fit_to_square(cropped, 32, margin=8)
    meta["stages"]["final"] = fitted
    return fitted.astype(np.float32) / 255.0, meta


def preprocess_variant_b(img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Variant B: Grayscale -> Gaussian Blur -> Adaptive Gaussian Threshold -> CC Dot Preservation"""
    meta = {"stages": {}, "status": "valid", "message": None}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    meta["stages"]["gray"] = gray

    if np.mean(gray) > 250 and np.std(gray) < 4:
        return np.zeros((32, 32), dtype=np.float32), {"status": "blank", "message": "No clear handwritten letter detected."}
    if np.mean(gray) < 15:
        return np.zeros((32, 32), dtype=np.float32), {"status": "invalid_lighting", "message": "Image is too dark."}

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold: localized thresholding window 21x21
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )
    meta["stages"]["binary"] = binary

    cleaned, cc_info = extract_character_and_dots(binary)
    meta["stages"]["cleaned"] = cleaned
    meta.update(cc_info)

    if meta["status"] != "valid":
        return np.zeros((32, 32), dtype=np.float32), meta

    coords = cv2.findNonZero(cleaned)
    if coords is None:
        return np.zeros((32, 32), dtype=np.float32), {"status": "blank", "message": "No clear handwritten letter detected."}

    x, y, bw, bh = cv2.boundingRect(coords)
    cropped = cleaned[y:y + bh, x:x + bw]
    fitted = fit_to_square(cropped, 32, margin=8)
    meta["stages"]["final"] = fitted
    return fitted.astype(np.float32) / 255.0, meta


def preprocess_variant_c(img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Variant C: Background Field Estimation & Division (Shadow Invariance)
    1. Estimate background illumination via morphological closing (large kernel)
    2. Divide image by background -> perfectly flat illumination across shadows
    3. CLAHE enhancement
    4. Adaptive thresholding
    5. Connected Component Analysis with strict Arabic dot preservation & multiple-letter detection
    6. Aspect-ratio preserving square fit (32x32)
    """
    meta = {"stages": {}, "status": "valid", "message": None}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    meta["stages"]["gray"] = gray

    # Check for solid black / white
    if np.mean(gray) > 250 and np.std(gray) < 4:
        return np.zeros((32, 32), dtype=np.float32), {"status": "blank", "message": "No clear handwritten letter detected."}
    if np.mean(gray) < 15:
        return np.zeros((32, 32), dtype=np.float32), {"status": "invalid_lighting", "message": "Image is too dark."}

    # 1. Background field estimation (morphological closing with large 31x31 ellipse)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    meta["stages"]["background"] = background

    # 2. Illumination Division (Shadow Flattening)
    # Norm = (gray / background) * 255
    bg_float = np.maximum(background.astype(np.float32), 1.0)
    norm = np.clip((gray.astype(np.float32) / bg_float) * 255.0, 0, 255).astype(np.uint8)
    meta["stages"]["shadow_normalized"] = norm

    # 3. CLAHE Enhancement on flattened image
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)

    # 4. Adaptive Thresholding with bilateral smoothing
    smoothed = cv2.bilateralFilter(enhanced, 5, 40, 40)
    binary = cv2.adaptiveThreshold(
        smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )
    meta["stages"]["binary"] = binary

    # 5. Connected Component Analysis with strict dot preservation & multiple letter detection
    cleaned, cc_info = extract_character_and_dots(binary)
    meta["stages"]["cleaned"] = cleaned
    meta.update(cc_info)

    if meta["status"] != "valid":
        return np.zeros((32, 32), dtype=np.float32), meta

    coords = cv2.findNonZero(cleaned)
    if coords is None:
        return np.zeros((32, 32), dtype=np.float32), {"status": "blank", "message": "No clear handwritten letter detected."}

    x, y, bw, bh = cv2.boundingRect(coords)
    cropped = cleaned[y:y + bh, x:x + bw]
    fitted = fit_to_square(cropped, 32, margin=8)
    meta["stages"]["final"] = fitted
    return fitted.astype(np.float32) / 255.0, meta


def extract_character_and_dots(binary_img: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Analyzes connected components:
    - Detects blank paper (total stroke < threshold)
    - Detects multiple large disconnected characters
    - Preserves all valid dots inside Arabic letter envelope
    - Removes distant margin noise
    """
    total_fg_pixels = cv2.countNonZero(binary_img)
    if total_fg_pixels < 25:
        return np.zeros_like(binary_img), {"status": "blank", "message": "No clear handwritten letter detected."}

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary_img), {"status": "blank", "message": "No clear handwritten letter detected."}

    areas = stats[1:, cv2.CC_STAT_AREA]
    sorted_cc_indices = np.argsort(areas)[::-1] + 1  # 1-indexed

    # Filter out outer border / guide frame components (spans > 45% in both dimensions)
    valid_cc_indices = []
    for idx in sorted_cc_indices:
        cw = stats[idx, cv2.CC_STAT_WIDTH]
        ch = stats[idx, cv2.CC_STAT_HEIGHT]
        if cw > 0.45 * binary_img.shape[1] and ch > 0.45 * binary_img.shape[0]:
            continue
        valid_cc_indices.append(idx)

    if not valid_cc_indices:
        return np.zeros_like(binary_img), {"status": "blank", "message": "No clear handwritten letter detected."}

    max_label = valid_cc_indices[0]
    max_area = stats[max_label, cv2.CC_STAT_AREA]

    # Multiple Letter Check:
    # If second largest component is also large (> 35% of max) and significantly horizontally separated
    if len(valid_cc_indices) > 1:
        second_label = valid_cc_indices[1]
        second_area = stats[second_label, cv2.CC_STAT_AREA]
        if second_area > 0.35 * max_area and second_area > 150:
            c1_x, c1_w = stats[max_label, cv2.CC_STAT_LEFT], stats[max_label, cv2.CC_STAT_WIDTH]
            c2_x, c2_w = stats[second_label, cv2.CC_STAT_LEFT], stats[second_label, cv2.CC_STAT_WIDTH]
            h_distance = abs((c1_x + c1_w / 2) - (c2_x + c2_w / 2))
            min_width = min(c1_w, c2_w)
            if h_distance > min_width * 1.2:
                return np.zeros_like(binary_img), {
                    "status": "multiple_letters",
                    "message": "Please write only one letter inside the frame.",
                }

    body_x = stats[max_label, cv2.CC_STAT_LEFT]
    body_y = stats[max_label, cv2.CC_STAT_TOP]
    body_w = stats[max_label, cv2.CC_STAT_WIDTH]
    body_h = stats[max_label, cv2.CC_STAT_HEIGHT]

    # Arabic letter envelope for dots:
    # Generous envelope: children place dots slightly offset
    x_expand = int(body_w * 0.7)
    y_expand = int(body_h * 1.3)

    env_x_min = max(0, body_x - x_expand)
    env_x_max = min(binary_img.shape[1], body_x + body_w + x_expand)
    env_y_min = max(0, body_y - y_expand)
    env_y_max = min(binary_img.shape[0], body_y + body_h + y_expand)

    kept_mask = np.zeros_like(binary_img, dtype=np.uint8)
    kept_mask[labels == max_label] = 255
    dots_preserved = 0

    min_dot_area = max(2, int(max_area * 0.003))

    for label in valid_cc_indices:
        if label == max_label:
            continue
        comp_area = stats[label, cv2.CC_STAT_AREA]
        cx, cy = centroids[label]

        if env_x_min <= cx <= env_x_max and env_y_min <= cy <= env_y_max:
            if min_dot_area <= comp_area <= (max_area * 0.5):
                kept_mask[labels == label] = 255
                dots_preserved += 1

    return kept_mask, {
        "status": "valid",
        "message": None,
        "dots_preserved": dots_preserved,
        "body_box": (int(body_x), int(body_y), int(body_w), int(body_h)),
    }


def fit_to_square(img: np.ndarray, target_size: int = 32, margin: int = 8) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    usable_size = max(4, target_size - (2 * margin))
    scale = usable_size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    start_x = (target_size - new_w) // 2
    start_y = (target_size - new_h) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
    return canvas


def run_comparison():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Load Model
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = ArabicCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    variants = {
        "Variant A (Current CLAHE+Otsu)": preprocess_variant_a,
        "Variant B (Adaptive Gaussian)": preprocess_variant_b,
        "Variant C (Shadow Field Division + Adaptive)": preprocess_variant_c,
    }

    results = {}

    for var_name, var_fn in variants.items():
        correct = 0
        accepted_correct = 0
        accepted_wrong = 0
        rejected_valid = 0
        correctly_rejected_invalid = 0
        total_valid = 0
        total_invalid = 0
        shadow_correct = 0
        faint_dots_correct = 0

        times = []

        for item in manifest:
            img_path = PROJECT_ROOT / item["file"]
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            cond = item["condition"]
            expected_behavior = item["expected_behavior"]
            true_id = item["true_class_id"]

            t0 = time.perf_counter()
            processed_32, meta = var_fn(img_bgr)
            t_prep = (time.perf_counter() - t0) * 1000
            times.append(t_prep)

            if expected_behavior == "reject":
                total_invalid += 1
                if meta["status"] != "valid":
                    correctly_rejected_invalid += 1
                continue

            total_valid += 1
            if meta["status"] != "valid":
                rejected_valid += 1
                continue

            # Inference
            tensor = torch.from_numpy(processed_32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).numpy()[0]

            top_indices = np.argsort(probs)[::-1]
            pred_id = int(top_indices[0])
            top1_prob = float(probs[pred_id])
            top2_prob = float(probs[top_indices[1]])
            margin = top1_prob - top2_prob

            accepted = (top1_prob >= 0.65 and margin >= 0.15)
            is_correct = (pred_id == true_id)

            if is_correct:
                correct += 1
                if cond == "shadow":
                    shadow_correct += 1
                if cond == "faint_dots":
                    faint_dots_correct += 1

            if accepted:
                if is_correct:
                    accepted_correct += 1
                else:
                    accepted_wrong += 1
            else:
                rejected_valid += 1

        accuracy_on_valid = (correct / total_valid) * 100.0 if total_valid else 0.0
        acc_accepted = (accepted_correct / (accepted_correct + accepted_wrong)) * 100.0 if (accepted_correct + accepted_wrong) else 0.0
        invalid_rejection_rate = (correctly_rejected_invalid / total_invalid) * 100.0 if total_invalid else 0.0

        results[var_name] = {
            "valid_accuracy": round(accuracy_on_valid, 2),
            "accepted_correct": accepted_correct,
            "accepted_wrong": accepted_wrong,
            "accepted_accuracy": round(acc_accepted, 2),
            "rejected_valid": rejected_valid,
            "invalid_rejected_rate": round(invalid_rejection_rate, 2),
            "shadow_correct_count": f"{shadow_correct}/6",
            "faint_dots_correct_count": f"{faint_dots_correct}/8",
            "mean_preprocessing_ms": round(float(np.mean(times)), 2),
        }

    print("\n" + "=" * 75)
    print("PREPROCESSING PIPELINE COMPARISON RESULTS")
    print("=" * 75)
    for v, res in results.items():
        print(f"\n{v}:")
        for k, val in res.items():
            print(f"  {k}: {val}")

    # Save to JSON
    with open("ml/evaluation/preprocessing_variants_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_comparison()
