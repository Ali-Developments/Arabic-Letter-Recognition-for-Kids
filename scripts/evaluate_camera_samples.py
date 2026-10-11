"""
Evaluate Camera Samples and Benchmark Camera Robustness Pipeline (Step 4).
Loads final candidate model checkpoint, processes all structured camera test samples,
computes metrics per condition, performs confidence calibration evaluation,
generates camera_results.csv and camera_benchmark.json, and produces
multi-stage visual debug outputs in ml/evaluation/preprocessing_debug/.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
import torch

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.ml.labels import ARABIC_LETTERS, ID_TO_CHAR, ID_TO_METADATA, NUM_CLASSES
from app.ml.predictor import ArabicPredictor
from app.ml.preprocessing import decode_image_bytes, fit_to_square, preprocess_image_for_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "tests" / "camera_samples" / "manifest.json"
CHECKPOINT_PATH = PROJECT_ROOT / "ml" / "models" / "final_candidate_model.pt"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "evaluation"
DEBUG_IMG_DIR = OUTPUT_DIR / "preprocessing_debug"


def create_debug_visualization(
    item: Dict,
    img_bgr: np.ndarray,
    meta: Dict,
    pred_res: Dict,
    out_path: Path,
):
    """
    Renders an 8-panel debug comparison image for an evaluated sample:
    1. Original Camera Image
    2. Guide Crop / Margin
    3. Grayscale
    4. Threshold (Binary)
    5. Detected Components
    6. Combined Character ROI
    7. Final 32x32 Model Input (Upscaled)
    8. Diagnostic Summary Card (True vs Pred, Conf, Latency, Status)
    """
    canvas_w = 1200
    canvas_h = 600
    viz = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

    # Header bar
    cv2.rectangle(viz, (0, 0), (canvas_w, 40), (40, 40, 40), -1)
    cond = item.get("condition", "unknown")
    title_text = f"Condition: {cond.upper()} | Sample: {Path(item['file']).name}"
    cv2.putText(viz, title_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    panel_size = 220
    y_top = 60
    y_bot = 320

    # Panel 1: Original
    orig_thumb = cv2.resize(img_bgr, (panel_size, panel_size))
    viz[y_top:y_top + panel_size, 20:20 + panel_size] = orig_thumb
    cv2.putText(viz, "1. Original Camera", (20, y_top + panel_size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)

    # Panel 2: Guide Box ROI
    guide_viz = img_bgr.copy()
    h, w = guide_viz.shape[:2]
    cv2.rectangle(guide_viz, (int(w * 0.12), int(h * 0.12)), (int(w * 0.88), int(h * 0.88)), (0, 255, 0), 2)
    guide_thumb = cv2.resize(guide_viz, (panel_size, panel_size))
    viz[y_top:y_top + panel_size, 260:260 + panel_size] = guide_thumb
    cv2.putText(viz, "2. Guide ROI", (260, y_top + panel_size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)

    # Panel 3: Grayscale
    stages = meta.get("stages", {})
    gray = stages.get("gray", cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    gray_bgr = cv2.cvtColor(cv2.resize(gray, (panel_size, panel_size)), cv2.COLOR_GRAY2BGR)
    viz[y_top:y_top + panel_size, 500:500 + panel_size] = gray_bgr
    cv2.putText(viz, "3. Grayscale", (500, y_top + panel_size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)

    # Panel 4: Binary Threshold
    bin_img = stages.get("binary", np.zeros((h, w), dtype=np.uint8))
    bin_bgr = cv2.cvtColor(cv2.resize(bin_img, (panel_size, panel_size)), cv2.COLOR_GRAY2BGR)
    viz[y_top:y_top + panel_size, 740:740 + panel_size] = bin_bgr
    cv2.putText(viz, "4. Adaptive Threshold", (740, y_top + panel_size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)

    # Panel 5: Detected Components & Cleaned
    cleaned = stages.get("cleaned", np.zeros((h, w), dtype=np.uint8))
    cleaned_bgr = cv2.cvtColor(cv2.resize(cleaned, (panel_size, panel_size)), cv2.COLOR_GRAY2BGR)
    viz[y_bot:y_bot + panel_size, 20:20 + panel_size] = cleaned_bgr
    cv2.putText(viz, f"5. CC + Dots ({meta.get('dots_preserved', 0)} dots)", (20, y_bot + panel_size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)

    # Panel 6: Character ROI Box
    roi_viz = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    coords = cv2.findNonZero(cleaned)
    if coords is not None:
        rx, ry, rw, rh = cv2.boundingRect(coords)
        cv2.rectangle(roi_viz, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
    roi_thumb = cv2.resize(roi_viz, (panel_size, panel_size))
    viz[y_bot:y_bot + panel_size, 260:260 + panel_size] = roi_thumb
    cv2.putText(viz, "6. Character ROI", (260, y_bot + panel_size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)

    # Panel 7: Final 32x32 Model Input (Upscaled to panel_size)
    final_32 = stages.get("final", np.zeros((32, 32), dtype=np.uint8))
    final_bgr = cv2.cvtColor(cv2.resize(final_32, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    # Add subtle grid lines showing 32x32 pixel structure
    viz[y_bot:y_bot + panel_size, 500:500 + panel_size] = final_bgr
    cv2.putText(viz, "7. Model Input (32x32)", (500, y_bot + panel_size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)

    # Panel 8: Diagnostic Summary Card
    card_x = 740
    card_y = y_bot
    card_w = 440
    card_h = panel_size

    status = meta.get("status", "unknown")
    accepted = pred_res.get("accepted", False)
    true_char = item.get("true_letter", "N/A")
    pred_char = pred_res.get("letter") or "REJECTED"
    conf = pred_res.get("confidence", 0.0)

    # Card background
    if status != "valid":
        card_bg = (230, 230, 255) if item.get("expected_behavior") == "reject" else (220, 220, 255)
    elif accepted and (pred_res.get("class_id") == item.get("true_class_id")):
        card_bg = (220, 255, 220)  # Green tint
    elif accepted:
        card_bg = (220, 220, 255)  # Red tint
    else:
        card_bg = (255, 240, 220)  # Orange/amber tint (rejected)

    cv2.rectangle(viz, (card_x, card_y), (card_x + card_w, card_y + card_h), card_bg, -1)
    cv2.rectangle(viz, (card_x, card_y), (card_x + card_w, card_y + card_h), (120, 120, 120), 1)

    cv2.putText(viz, "DIAGNOSTIC VERDICT", (card_x + 15, card_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    cv2.putText(viz, f"Status: {status.upper()}", (card_x + 15, card_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)

    true_name = ID_TO_METADATA.get(item.get("true_class_id", -1), {}).get("name_en", "N/A")
    pred_name = pred_res.get("name_en") or "None"
    cv2.putText(viz, f"True: {true_name} ({true_char})", (card_x + 15, card_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
    cv2.putText(viz, f"Pred: {pred_name} ({pred_char})", (card_x + 15, card_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2)
    cv2.putText(viz, f"Confidence: {conf*100:.1f}%", (card_x + 15, card_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
    cv2.putText(viz, f"Decision: {'ACCEPTED' if accepted else 'REJECTED'}", (card_x + 15, card_y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 130, 0) if accepted else (0, 0, 180), 2)

    cv2.putText(viz, "8. Inference Summary", (card_x, y_bot + panel_size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), viz)


def run_benchmark():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_IMG_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest {MANIFEST_PATH} not found. Run generate_camera_stress_dataset.py first.")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    predictor = ArabicPredictor(model_path=str(CHECKPOINT_PATH), acceptance_threshold=0.75)

    print("=" * 70)
    print("RUNNING CAMERA BENCHMARK EVALUATION (STEP 4)")
    print(f"Model Checkpoint: {CHECKPOINT_PATH.name}")
    print(f"Active Acceptance Threshold: {predictor.confidence_manager.acceptance_threshold}")
    print(f"Active Min Top-2 Margin: {predictor.confidence_manager.min_margin_threshold}")
    print(f"Total Test Samples: {len(manifest)}")
    print("=" * 70)

    rows: List[Dict] = []
    condition_stats: Dict[str, Dict] = {}

    prep_times = []
    infer_times = []
    e2e_times = []

    # Process all samples
    for idx, item in enumerate(manifest):
        file_path = PROJECT_ROOT / item["file"]
        if not file_path.exists():
            print(f"Warning: File {file_path} does not exist. Skipping.")
            continue

        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)

        # 1. Preprocessing measurement
        t0 = time.perf_counter()
        processed_norm, prep_meta = preprocess_image_for_model(img_bgr, return_metadata=True)
        t_prep = (time.perf_counter() - t0) * 1000.0
        prep_times.append(t_prep)

        # 2. End-to-end predictor invocation
        t1 = time.perf_counter()
        pred_res = predictor.predict_from_bytes(raw_bytes, top_k=3)
        t_e2e = (time.perf_counter() - t1) * 1000.0
        e2e_times.append(t_e2e)
        t_infer = max(0.1, t_e2e - t_prep)
        infer_times.append(t_infer)

        cond = item.get("condition", "general")
        expected_behavior = item.get("expected_behavior", "accept")
        true_cid = item.get("true_class_id")
        true_char = item.get("true_letter")

        pred_cid = pred_res.get("class_id")
        pred_char = pred_res.get("letter")
        confidence = pred_res.get("confidence", 0.0)
        accepted = pred_res.get("accepted", False)

        # Top 2 margin calculation
        top_cands = pred_res.get("top_candidates", [])
        if len(top_cands) >= 2:
            top2_margin = round(top_cands[0]["confidence"] - top_cands[1]["confidence"], 4)
        else:
            top2_margin = 0.0

        # Outcome evaluation
        if expected_behavior == "reject":
            correct = (not accepted)
            failure_reason = None if correct else "False acceptance on invalid input"
        else:
            is_match = (pred_cid == true_cid)
            correct = is_match
            if not correct and accepted:
                failure_reason = f"Misclassified as {pred_char} (conf {confidence:.2f})"
            elif not accepted:
                failure_reason = pred_res.get("message") or "Low confidence / rejected"
            else:
                failure_reason = None

        row = {
            "file": item["file"],
            "condition": cond,
            "true_letter": true_char,
            "predicted_letter": pred_char,
            "confidence": confidence,
            "top2_margin": top2_margin,
            "accepted": accepted,
            "correct": correct,
            "processing_time_ms": round(t_e2e, 2),
            "failure_reason": failure_reason,
        }
        rows.append(row)

        # Accumulate condition stats
        if cond not in condition_stats:
            condition_stats[cond] = {
                "total": 0,
                "correct": 0,
                "accepted_correct": 0,
                "accepted_wrong": 0,
                "rejected": 0,
            }

        cs = condition_stats[cond]
        cs["total"] += 1
        if expected_behavior == "reject":
            if not accepted:
                cs["correct"] += 1
                cs["rejected"] += 1
            else:
                cs["accepted_wrong"] += 1
        else:
            if correct:
                cs["correct"] += 1
                if accepted:
                    cs["accepted_correct"] += 1
                else:
                    cs["rejected"] += 1
            else:
                if accepted:
                    cs["accepted_wrong"] += 1
                else:
                    cs["rejected"] += 1

        # Generate representative debug images (first 2 per category or upon error)
        if (cs["total"] <= 2) or (not correct and cs["total"] <= 4):
            debug_filename = f"debug_{cond}_{Path(item['file']).stem}.jpg"
            debug_path = DEBUG_IMG_DIR / debug_filename
            create_debug_visualization(item, img_bgr, prep_meta, pred_res, debug_path)

    # Write CSV
    csv_path = OUTPUT_DIR / "camera_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "condition", "true_letter", "predicted_letter",
            "confidence", "top2_margin", "accepted", "correct",
            "processing_time_ms", "failure_reason",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate global metrics
    valid_rows = [r for r in rows if r["true_letter"] is not None]
    invalid_rows = [r for r in rows if r["true_letter"] is None]

    total_valid = len(valid_rows)
    overall_correct = sum(1 for r in valid_rows if r["correct"])
    accepted_correct = sum(1 for r in valid_rows if r["accepted"] and r["correct"])
    accepted_wrong = sum(1 for r in valid_rows if r["accepted"] and not r["correct"])
    rejected_count = sum(1 for r in valid_rows if not r["accepted"])

    total_accepted = accepted_correct + accepted_wrong
    accepted_acc = (accepted_correct / total_accepted * 100.0) if total_accepted > 0 else 0.0
    overall_acc = (overall_correct / total_valid * 100.0) if total_valid > 0 else 0.0
    false_acc_rate = (accepted_wrong / total_valid * 100.0) if total_valid > 0 else 0.0
    rejection_rate = (rejected_count / total_valid * 100.0) if total_valid > 0 else 0.0

    # Invalid rejection rate
    invalid_rejected = sum(1 for r in invalid_rows if not r["accepted"])
    invalid_rejection_rate = (invalid_rejected / len(invalid_rows) * 100.0) if invalid_rows else 100.0

    benchmark_summary = {
        "model_architecture": "ArabicCNN",
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "acceptance_threshold": predictor.confidence_manager.acceptance_threshold,
        "min_top2_margin": predictor.confidence_manager.min_margin_threshold,
        "total_camera_samples": len(rows),
        "valid_samples": total_valid,
        "invalid_samples": len(invalid_rows),
        "overall_camera_accuracy": round(overall_acc, 2),
        "accepted_accuracy": round(accepted_acc, 2),
        "accepted_correct_count": accepted_correct,
        "accepted_wrong_count": accepted_wrong,
        "rejection_rate_percent": round(rejection_rate, 2),
        "false_acceptance_rate_percent": round(false_acc_rate, 2),
        "invalid_rejection_rate_percent": round(invalid_rejection_rate, 2),
        "latency_ms": {
            "preprocessing_mean": round(float(np.mean(prep_times)), 2),
            "model_inference_mean": round(float(np.mean(infer_times)), 2),
            "end_to_end_mean": round(float(np.mean(e2e_times)), 2),
        },
        "condition_breakdown": {},
    }

    for cond, cs in condition_stats.items():
        cond_tot = cs["total"]
        cond_acc = (cs["correct"] / cond_tot * 100.0) if cond_tot > 0 else 0.0
        cond_acc_corr = cs["accepted_correct"]
        cond_acc_wrg = cs["accepted_wrong"]
        tot_acc = cond_acc_corr + cond_acc_wrg
        cond_acc_prec = (cond_acc_corr / tot_acc * 100.0) if tot_acc > 0 else 0.0
        benchmark_summary["condition_breakdown"][cond] = {
            "total": cond_tot,
            "accuracy_percent": round(cond_acc, 2),
            "accepted_correct": cond_acc_corr,
            "accepted_wrong": cond_acc_wrg,
            "accepted_precision_percent": round(cond_acc_prec, 2),
            "rejected": cs["rejected"],
        }

    json_path = OUTPUT_DIR / "camera_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("CAMERA BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Overall Accuracy:        {overall_acc:.2f}% ({overall_correct}/{total_valid})")
    print(f"Accepted Accuracy:       {accepted_acc:.2f}% ({accepted_correct}/{total_accepted})")
    print(f"Accepted Correct:        {accepted_correct}")
    print(f"Accepted Wrong:          {accepted_wrong}")
    print(f"False Acceptance Rate:   {false_acc_rate:.2f}%")
    print(f"Rejection Rate:          {rejection_rate:.2f}% ({rejected_count}/{total_valid})")
    print(f"Invalid Rejection Rate:  {invalid_rejection_rate:.2f}% ({invalid_rejected}/{len(invalid_rows)})")
    print("-" * 70)
    print(f"Mean Preprocessing Time: {np.mean(prep_times):.2f} ms")
    print(f"Mean Inference Time:     {np.mean(infer_times):.2f} ms")
    print(f"Mean End-to-End Latency: {np.mean(e2e_times):.2f} ms")
    print("-" * 70)
    print("Results by condition:")
    for cond, data in benchmark_summary["condition_breakdown"].items():
        print(f"  {cond:<15}: Acc: {data['accuracy_percent']:5.1f}% | AccCorr: {data['accepted_correct']} | AccWrong: {data['accepted_wrong']} | Rej: {data['rejected']}")
    print("=" * 70)
    print(f"Saved results to: {csv_path}")
    print(f"Saved benchmark to: {json_path}")
    print(f"Generated debug images in: {DEBUG_IMG_DIR}")


if __name__ == "__main__":
    run_benchmark()
