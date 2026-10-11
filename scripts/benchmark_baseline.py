import time
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path("backend").resolve()))
sys.path.insert(0, ".")

from app.ml.labels import NUM_CLASSES, ID_TO_CHAR, ID_TO_METADATA
from app.ml.predictor import ArabicCNN
from ml.training.dataset import ArabicHandwritingDataset

PROJECT_ROOT = Path(".")
HIJJA_TEST = PROJECT_ROOT / "ml" / "data" / "processed" / "hijja" / "test"
CHECKPOINT_PATH = PROJECT_ROOT / "ml" / "models" / "baseline_arabic_cnn_best.pt"

DOTTED_CLASS_IDS = [cid for cid, meta in ID_TO_METADATA.items() if meta.get("dots", 0) > 0]
print(f"Dotted Class IDs ({len(DOTTED_CLASS_IDS)}):", DOTTED_CLASS_IDS)
print("Dotted Letters:", [ID_TO_CHAR[cid] for cid in DOTTED_CLASS_IDS])

# 1. Load Model and measure load time
t0 = time.perf_counter()
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
model = ArabicCNN(num_classes=NUM_CLASSES)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
t1 = time.perf_counter()
load_time_ms = (t1 - t0) * 1000

# 2. Evaluate on Hijja test set
test_ds = ArabicHandwritingDataset(str(HIJJA_TEST), is_training=False)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)

all_preds = []
all_targets = []
all_probs = []
latencies_ms = []

with torch.no_grad():
    for images, labels in test_loader:
        t_start = time.perf_counter()
        outputs = model(images)
        t_end = time.perf_counter()
        latencies_ms.append((t_end - t_start) * 1000)

        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred = int(np.argmax(probs))
        all_preds.append(pred)
        all_targets.append(int(labels[0]))
        all_probs.append(probs)

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

# Metrics
acc = accuracy_score(all_targets, all_preds) * 100.0
macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0

# Dotted-letter subset accuracy
dotted_mask = np.isin(all_targets, DOTTED_CLASS_IDS)
dotted_targets = all_targets[dotted_mask]
dotted_preds = all_preds[dotted_mask]
dotted_acc = accuracy_score(dotted_targets, dotted_preds) * 100.0

# Accepted Wrong Rate (threshold = 0.65)
wrong_mask = (all_preds != all_targets)
high_conf_mask = np.array([max(p) >= 0.65 for p in all_probs])
accepted_wrong_count = int(np.sum(wrong_mask & high_conf_mask))
accepted_wrong_rate = (accepted_wrong_count / len(all_targets)) * 100.0

# Latency metrics (ignore first 10 warmup runs)
bench_latencies = latencies_ms[10:]
mean_lat = float(np.mean(bench_latencies))
med_lat = float(np.median(bench_latencies))
p95_lat = float(np.percentile(bench_latencies, 95))

params = sum(p.numel() for p in model.parameters())
size_mb = CHECKPOINT_PATH.stat().st_size / (1024 * 1024)

results = {
    "model": "ArabicCNN (Baseline)",
    "training_data": "Hijja Child Train",
    "test_accuracy": round(acc, 2),
    "macro_f1": round(macro_f1, 2),
    "dotted_accuracy": round(dotted_acc, 2),
    "dotted_samples_count": int(np.sum(dotted_mask)),
    "params": params,
    "checkpoint_size_mb": round(size_mb, 2),
    "load_time_ms": round(load_time_ms, 2),
    "latency_mean_ms": round(mean_lat, 2),
    "latency_median_ms": round(med_lat, 2),
    "latency_p95_ms": round(p95_lat, 2),
    "accepted_wrong_rate": round(accepted_wrong_rate, 2),
    "accepted_wrong_count": accepted_wrong_count,
}

print("\n--- BASELINE METRICS ---")
print(json.dumps(results, indent=2))
with open("ml/evaluation/baseline_benchmarks.json", "w") as f:
    json.dump(results, f, indent=2)
