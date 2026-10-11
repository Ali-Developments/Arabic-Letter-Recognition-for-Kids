"""
Training and Evaluation Pipeline for MobileNetV3-Small (Step 3).
Executes:
1. Experiment A: MobileNetV3-Small on Hijja child-only data.
2. Experiment B: MobileNetV3-Small on Mixed data (Hijja child 70% + AHCD adult 30% balanced sampling).
Evaluates both on untouched Hijja child test split.
Generates metrics, training curves, confusion matrices, and detailed comparison reports.
"""

import os
import sys
import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.ml.labels import (
    ARABIC_LETTERS,
    LETTER_METADATA,
    NUM_CLASSES,
    ID_TO_CHAR,
    ID_TO_METADATA,
)
from app.ml.architectures import ArabicMobileNetV3
from ml.training.dataset import ArabicHandwritingDataset

HIJJA_TRAIN = PROJECT_ROOT / "ml" / "data" / "processed" / "hijja" / "train"
HIJJA_VAL = PROJECT_ROOT / "ml" / "data" / "processed" / "hijja" / "val"
HIJJA_TEST = PROJECT_ROOT / "ml" / "data" / "processed" / "hijja" / "test"
AHCD_TRAIN = PROJECT_ROOT / "ml" / "data" / "processed" / "ahcd" / "train"

MODELS_DIR = PROJECT_ROOT / "ml" / "models"
EVALUATION_DIR = PROJECT_ROOT / "ml" / "evaluation"

DOTTED_CLASS_IDS = [cid for cid, meta in ID_TO_METADATA.items() if meta.get("dots", 0) > 0]


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_single_experiment(
    exp_name: str,
    checkpoint_basename: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_ds: Dataset,
    num_train_samples: int,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_str: str = "cpu",
    is_mixed: bool = False,
) -> Dict:
    set_seed(42)
    device = torch.device(device_str)

    exp_dir = PROJECT_ROOT / "ml" / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = MODELS_DIR / f"{checkpoint_basename}_best.pt"

    config = {
        "model_architecture": "ArabicMobileNetV3",
        "experiment_name": exp_name,
        "is_mixed": is_mixed,
        "num_classes": NUM_CLASSES,
        "input_size": [1, 32, 32],
        "training_dataset": "Hijja + AHCD Balanced (70/30)" if is_mixed else "Hijja Children (Isolated)",
        "train_samples_per_epoch": num_train_samples,
        "batch_size": train_loader.batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau(mode='max', factor=0.5, patience=2)",
        "early_stopping_patience": 5,
        "max_epochs": epochs,
        "random_seed": 42,
        "device": device_str,
        "pretrained_imagenet": True,
        "first_layer_adaptation": "RGB 3-channel average to 1-channel grayscale",
    }

    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print(f"RUNNING EXPERIMENT: {exp_name.upper()}")
    print(f"Dataset: {config['training_dataset']} | Epochs: {epochs} | Batch: {train_loader.batch_size} | LR: {lr}")
    print("=" * 70)

    model = ArabicMobileNetV3(num_classes=NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100.0

        # Validation on Hijja Child Val Set
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss = val_running_loss / total_val
        val_acc = (correct_val / total_val) * 100.0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 2))
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | LR: {current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            checkpoint_payload = {
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "class_mapping": ID_TO_CHAR,
                "input_size": [1, 32, 32],
                "best_epoch": best_epoch,
                "validation_accuracy": round(best_val_acc, 2),
                "validation_loss": round(best_val_loss, 4),
                "training_config": config,
            }
            torch.save(checkpoint_payload, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint to {best_checkpoint_path.name} (Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # Save training_history.csv
    history_csv = exp_dir / "training_history.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i],
                history["lr"][i],
            ])

    # Plot training loss & accuracy
    plot_curves(history, exp_name)

    # Evaluate on untouched Hijja Child Test Set
    test_results = evaluate_candidate(best_checkpoint_path, test_ds, exp_name, device_str)

    metrics_summary = {
        "experiment": exp_name,
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_acc, 2),
        "best_val_loss": round(best_val_loss, 4),
        "test_accuracy": test_results["test_accuracy"],
        "macro_precision": test_results["macro_precision"],
        "macro_recall": test_results["macro_recall"],
        "macro_f1": test_results["macro_f1"],
        "dotted_accuracy": test_results["dotted_accuracy"],
        "accepted_wrong_rate": test_results["accepted_wrong_rate"],
        "top_confusions": test_results["top_confusions"],
    }

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    return test_results


def plot_curves(history: Dict, exp_name: str):
    epochs = history["epoch"]

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#4361ee", linewidth=2)
    plt.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ef4444", linewidth=2)
    plt.title(f"{exp_name} — Loss Curve", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(EVALUATION_DIR / f"{exp_name}_training_loss.png", dpi=150)
    plt.close()

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], "o-", label="Train Acc", color="#10b981", linewidth=2)
    plt.plot(epochs, history["val_acc"], "s--", label="Val Acc", color="#7209b7", linewidth=2)
    plt.title(f"{exp_name} — Accuracy Curve", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(EVALUATION_DIR / f"{exp_name}_training_accuracy.png", dpi=150)
    plt.close()


def evaluate_candidate(checkpoint_path: Path, test_ds: Dataset, exp_name: str, device_str: str) -> Dict:
    print(f"\nEvaluating {checkpoint_path.name} on untouched Hijja child test split...")
    device = torch.device(device_str)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = ArabicMobileNetV3(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    all_preds = []
    all_targets = []
    all_probs = []
    latencies_ms = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            t0 = time.perf_counter()
            outputs = model(images)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            all_preds.append(pred)
            all_targets.append(int(labels[0]))
            all_probs.append(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    macro_rec = recall_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0

    # Dotted letter subset accuracy
    dotted_mask = np.isin(all_targets, DOTTED_CLASS_IDS)
    dotted_acc = accuracy_score(all_targets[dotted_mask], all_preds[dotted_mask]) * 100.0

    # Accepted wrong rate
    wrong_mask = (all_preds != all_targets)
    high_conf_mask = np.array([max(p) >= 0.65 for p in all_probs])
    accepted_wrong_rate = (np.sum(wrong_mask & high_conf_mask) / len(all_targets)) * 100.0

    # Latencies
    bench_latencies = latencies_ms[10:]
    mean_lat = float(np.mean(bench_latencies))
    med_lat = float(np.median(bench_latencies))
    p95_lat = float(np.percentile(bench_latencies, 95))

    params = sum(p.numel() for p in model.parameters())
    size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    # Per-class report
    class_report_dict = classification_report(
        all_targets, all_preds, target_names=[f"{i:02d}" for i in range(28)], output_dict=True, zero_division=0
    )

    per_class_csv = EVALUATION_DIR / f"{exp_name}_per_class_metrics.csv"
    with open(per_class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "letter", "name_en", "name_ar", "precision", "recall", "f1_score", "support"])
        for cid in range(28):
            key = f"{cid:02d}"
            meta = ID_TO_METADATA[cid]
            c_m = class_report_dict[key]
            writer.writerow([
                cid,
                meta["letter"],
                meta["name_en"],
                meta["name_ar"],
                round(c_m["precision"] * 100, 2),
                round(c_m["recall"] * 100, 2),
                round(c_m["f1-score"] * 100, 2),
                int(c_m["support"]),
            ])

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(28)))
    top_confusions = plot_cm(cm, exp_name)

    # Full report
    report_data = {
        "experiment": exp_name,
        "checkpoint": checkpoint_path.name,
        "test_accuracy": round(acc, 2),
        "macro_precision": round(macro_prec, 2),
        "macro_recall": round(macro_rec, 2),
        "macro_f1": round(macro_f1, 2),
        "dotted_accuracy": round(dotted_acc, 2),
        "params": params,
        "checkpoint_size_mb": round(size_mb, 2),
        "latency_mean_ms": round(mean_lat, 2),
        "latency_median_ms": round(med_lat, 2),
        "latency_p95_ms": round(p95_lat, 2),
        "accepted_wrong_rate": round(accepted_wrong_rate, 2),
        "top_confusions": top_confusions,
    }

    with open(EVALUATION_DIR / f"{exp_name}_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"Results for {exp_name}:")
    print(f"  Hijja Test Acc: {acc:.2f}% | Macro F1: {macro_f1:.2f}% | Dotted Acc: {dotted_acc:.2f}%")
    print(f"  Latency (mean): {mean_lat:.2f} ms | Params: {params:,} | Size: {size_mb:.2f} MB")

    return report_data


def plot_cm(cm: np.ndarray, exp_name: str) -> List[Dict]:
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix — {exp_name}", fontsize=14, fontweight="bold")
    plt.colorbar(fraction=0.046, pad=0.04)

    labels = [f"{i:02d}" for i in range(28)]
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, fontsize=8)
    plt.yticks(tick_marks, labels, fontsize=8)

    thresh = cm.max() / 2.0
    for i in range(28):
        for j in range(28):
            val = cm[i, j]
            if val > 0:
                plt.text(
                    j,
                    i,
                    format(val, "d"),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if val > thresh else "black",
                    fontsize=7,
                )

    plt.ylabel("True Class ID", fontsize=11)
    plt.xlabel("Predicted Class ID", fontsize=11)
    plt.tight_layout()
    plt.savefig(EVALUATION_DIR / f"{exp_name}_confusion_matrix.png", dpi=150)
    plt.close()

    # Identify top confusion pairs
    confusions = []
    for i in range(28):
        for j in range(28):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "true_id": i,
                    "pred_id": j,
                    "true_letter": ID_TO_CHAR[i],
                    "pred_letter": ID_TO_CHAR[j],
                    "true_name": ID_TO_METADATA[i]["name_en"],
                    "pred_name": ID_TO_METADATA[j]["name_en"],
                    "count": int(cm[i, j]),
                })

    confusions.sort(key=lambda x: x["count"], reverse=True)
    return confusions[:10]


def run_all_experiments():
    set_seed(42)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device for Step 3 Experiments: {device_str}")

    # Shared Validation and Test Datasets (Untouched Hijja Child Data)
    val_ds = ArabicHandwritingDataset(str(HIJJA_VAL), is_training=False)
    test_ds = ArabicHandwritingDataset(str(HIJJA_TEST), is_training=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # ----------------------------------------------------
    # EXPERIMENT A: MobileNetV3-Small Hijja Only
    # ----------------------------------------------------
    hijja_train_ds = ArabicHandwritingDataset(str(HIJJA_TRAIN), is_training=True)
    hijja_train_loader = DataLoader(hijja_train_ds, batch_size=64, shuffle=True, num_workers=0)

    exp_a_results = train_single_experiment(
        exp_name="mobilenet_hijja",
        checkpoint_basename="mobilenetv3_hijja",
        train_loader=hijja_train_loader,
        val_loader=val_loader,
        test_ds=test_ds,
        num_train_samples=len(hijja_train_ds),
        epochs=20,
        lr=1e-3,
        weight_decay=1e-4,
        device_str=device_str,
        is_mixed=False,
    )

    # ----------------------------------------------------
    # EXPERIMENT B: MobileNetV3-Small Mixed (Hijja + AHCD Balanced)
    # ----------------------------------------------------
    ahcd_train_ds = ArabicHandwritingDataset(str(AHCD_TRAIN), is_training=True)
    mixed_ds = ConcatDataset([hijja_train_ds, ahcd_train_ds])

    # Controlled Weighted Sampling: 70% Hijja (Child), 30% AHCD (Adult)
    n_hijja = len(hijja_train_ds)
    n_ahcd = len(ahcd_train_ds)
    weights = [0.70 / n_hijja] * n_hijja + [0.30 / n_ahcd] * n_ahcd
    sampler = WeightedRandomSampler(weights, num_samples=n_hijja, replacement=True)
    mixed_train_loader = DataLoader(mixed_ds, batch_size=64, sampler=sampler, num_workers=0)

    exp_b_results = train_single_experiment(
        exp_name="mobilenet_mixed",
        checkpoint_basename="mobilenetv3_mixed",
        train_loader=mixed_train_loader,
        val_loader=val_loader,
        test_ds=test_ds,
        num_train_samples=n_hijja,
        epochs=20,
        lr=1e-3,
        weight_decay=1e-4,
        device_str=device_str,
        is_mixed=True,
    )

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments()
