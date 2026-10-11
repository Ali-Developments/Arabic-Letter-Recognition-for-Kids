import json
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import cv2
from app.ml.predictor import ArabicPredictor

predictor = ArabicPredictor()
target_letters = ["ب", "ت", "ث", "ج", "خ", "ش", "ض", "ظ", "غ", "ف", "ق", "ن", "ي"]

manifest_path = Path(__file__).resolve().parent.parent / "tests" / "camera_samples" / "manifest.json"
with open(manifest_path, "r", encoding="utf-8") as f:
    samples = json.load(f)

results = []
for s in samples:
    char = s.get("true_letter")
    if char in target_letters:
        img_path = Path(__file__).resolve().parent.parent / s["file"]
        if not img_path.exists():
            continue
        with open(img_path, "rb") as f_img:
            img_bytes = f_img.read()

        req_id = f"eval_{s['file'].replace('/', '_').replace('.', '_')}"
        pred = predictor.predict_from_bytes(img_bytes, top_k=3, request_id=req_id, save_debug=True)
        is_correct = (pred["letter"] == char)
        results.append({
            "char": char,
            "condition": s["condition"],
            "file": s["file"],
            "pred": pred["letter"],
            "conf": pred["confidence"],
            "accepted": pred["accepted"],
            "correct": is_correct,
            "message": pred["message"]
        })

print(f"=== EVALUATED {len(results)} TARGET DOTTED SAMPLES ===")
correct_count = sum(1 for r in results if r["correct"])
accepted_count = sum(1 for r in results if r["accepted"])
print(f"Total: {len(results)}, Correct: {correct_count}, Accepted: {accepted_count}")

# Group by letter
by_letter = {}
for r in results:
    char = r["char"]
    if char not in by_letter:
        by_letter[char] = {"total": 0, "correct": 0, "accepted": 0, "conditions": []}
    by_letter[char]["total"] += 1
    if r["correct"]:
        by_letter[char]["correct"] += 1
    if r["accepted"]:
        by_letter[char]["accepted"] += 1
    by_letter[char]["conditions"].append((r["condition"], r["pred"], r["conf"], r["correct"]))

for char, stats in by_letter.items():
    char_esc = char.encode("unicode_escape").decode()
    print(f"\nLetter {char_esc} (Total: {stats['total']}, Correct: {stats['correct']}, Accepted: {stats['accepted']}):")
    for cond, pred, conf, corr in stats["conditions"]:
        pred_esc = pred.encode("unicode_escape").decode() if pred else "None"
        mark = "[PASS]" if corr else "[FAIL]"
        print(f"  {mark} {cond}: pred={pred_esc} (conf={conf:.2f})")
