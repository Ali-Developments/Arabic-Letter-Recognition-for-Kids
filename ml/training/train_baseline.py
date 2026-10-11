"""
Train and Evaluate Baseline ArabicCNN Classifier.
Adheres strictly to Step 2 of ARABIC_HANDWRITING_AI_MASTER_EXECUTION_PLAN.md.
Trains exclusively on Hijja isolated children handwriting.
Evaluates on the untouched Hijja test split.
Generates metrics, training plots, confusion matrix, and misclassification grids.
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
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure backend imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.ml.labels import (
    ARABIC_LETTERS,
    LETTER_METADATA,
    NUM_CLASSES,
    ID_TO_CHAR,
    ID_TO_METADATA,
)
from app.ml.predictor import ArabicCNN
from ml.training.dataset import ArabicHandwritingDataset
HIJJA_TRAIN = PROJECT_ROOT / "ml" / "data" / "processed" / "hijja" / "train"
HIJJA_VAL = PROJECT_ROOT / "ml" / "data" / "processed" / "hijja" / "val"
HIJJA_TEST = PROJECT_ROOT / "ml" / "data" / "processed" / "hijja" / "test"

MODELS_DIR = PROJECT_ROOT / "ml" / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "ml" / "experiments" / "baseline_cnn"
EVALUATION_DIR = PROJECT_ROOT / "ml" / "evaluation"

for d in [MODELS_DIR, EXPERIMENTS_DIR, EVALUATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_baseline_model(
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 6,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    set_seed(42)
    device = torch.device(device_str)

    config = {
        "model_architecture": "ArabicCNN",
        "num_classes": NUM_CLASSES,
        "input_size": [1, 32, 32],
        "training_dataset": "Hijja Children Handwriting (Isolated)",
        "train_samples_count": 8632,
        "val_samples_count": 1840,
        "test_samples_count": 1879,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau(mode='max', factor=0.5, patience=2)",
        "early_stopping_patience": early_stopping_patience,
        "max_epochs": epochs,
        "random_seed": 42,
        "device": device_str,
        "augmentation": {
            "rotation_degrees": 10.0,
            "translation_pixels": 3,
            "scale_range": [0.9, 1.1],
            "noise_std": 0.03,
            "horizontal_flip": False,
            "vertical_flip": False,
        },
    }

    # Save config.json
    config_file = EXPERIMENTS_DIR / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("=" * 65)
    print("STEP 2: TRAINING BASELINE ARABICCNN (HIJJA CHILDREN DATASET)")
    print(f"Device: {device} | Batch Size: {batch_size} | Max Epochs: {epochs} | LR: {lr}")
    print("=" * 65)

    train_ds = ArabicHandwritingDataset(str(HIJJA_TRAIN), is_training=True)
    val_ds = ArabicHandwritingDataset(str(HIJJA_VAL), is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ArabicCNN(num_classes=NUM_CLASSES).to(device)
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
    best_state_dict = None
    patience_counter = 0

    best_checkpoint_path = MODELS_DIR / "baseline_arabic_cnn_best.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        current_lr = optimizer.param_groups[0]["lr"]

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = (val_correct / val_total) * 100.0

        scheduler.step(val_acc)
        elapsed = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 2))
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | LR: {current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0

            # Save checkpoint
            checkpoint_payload = {
                "model_state_dict": best_state_dict,
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
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (patience: {early_stopping_patience}).")
                break

    # Save training_history.csv
    history_csv = EXPERIMENTS_DIR / "training_history.csv"
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
    plot_training_curves(history)

    print("\nTraining stage finished.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% at Epoch {best_epoch}")

    # Evaluate on the untouched Test set
    test_results = evaluate_on_test_set(best_checkpoint_path, device_str)

    # Save metrics.json
    metrics_summary = {
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_acc, 2),
        "best_val_loss": round(best_val_loss, 4),
        "test_accuracy": test_results["test_accuracy"],
        "macro_precision": test_results["macro_precision"],
        "macro_recall": test_results["macro_recall"],
        "macro_f1": test_results["macro_f1"],
        "top_confusions": test_results["top_confusions"],
    }
    with open(EXPERIMENTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    return metrics_summary


def plot_training_curves(history: Dict):
    epochs = history["epoch"]

    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#4361ee", linewidth=2)
    plt.plot(epochs, history["val_loss"], "s--", label="Validation Loss", color="#ef4444", linewidth=2)
    plt.title("Baseline ArabicCNN — Training & Validation Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    loss_path = EVALUATION_DIR / "baseline_training_loss.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()

    # 2. Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], "o-", label="Train Accuracy", color="#10b981", linewidth=2)
    plt.plot(epochs, history["val_acc"], "s--", label="Validation Accuracy", color="#7209b7", linewidth=2)
    plt.title("Baseline ArabicCNN — Training & Validation Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    acc_path = EVALUATION_DIR / "baseline_training_accuracy.png"
    plt.savefig(acc_path, dpi=150)
    plt.close()

    print(f"Saved training loss curve to: {loss_path}")
    print(f"Saved training accuracy curve to: {acc_path}")


def evaluate_on_test_set(checkpoint_path: Path, device_str: str) -> Dict:
    print("\n" + "=" * 65)
    print("STEP 2: EVALUATING BEST BASELINE MODEL ON HIJJA TEST SPLIT")
    print("=" * 65)

    device = torch.device(device_str)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = ArabicCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_ds = ArabicHandwritingDataset(str(HIJJA_TEST), is_training=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    all_preds = []
    all_targets = []
    all_probs = []
    misclassified_samples = []

    with torch.no_grad():
        sample_global_idx = 0
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            labels_np = labels.numpy()
            all_preds.extend(preds)
            all_targets.extend(labels_np)
            all_probs.extend(probs)

            # Record misclassified samples
            for i in range(len(labels_np)):
                if preds[i] != labels_np[i]:
                    img_path, true_cls = test_ds.samples[sample_global_idx + i]
                    misclassified_samples.append({
                        "image_path": str(img_path),
                        "true_id": int(true_cls),
                        "pred_id": int(preds[i]),
                        "true_letter": ID_TO_CHAR[true_cls],
                        "pred_letter": ID_TO_CHAR[preds[i]],
                        "confidence": float(probs[i][preds[i]]),
                    })

            sample_global_idx += len(labels_np)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute overall metrics
    acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    macro_rec = recall_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0

    print(f"Overall Test Accuracy: {acc:.2f}%")
    print(f"Macro Precision:       {macro_prec:.2f}%")
    print(f"Macro Recall:          {macro_rec:.2f}%")
    print(f"Macro F1-Score:        {macro_f1:.2f}%")

    # Per-class metrics
    class_report_dict = classification_report(
        all_targets, all_preds, target_names=[f"{i:02d}" for i in range(28)], output_dict=True, zero_division=0
    )

    per_class_csv_path = EVALUATION_DIR / "baseline_per_class_metrics.csv"
    with open(per_class_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "letter", "name_en", "name_ar", "precision", "recall", "f1_score", "support"])
        for cid in range(28):
            key = f"{cid:02d}"
            meta = ID_TO_METADATA[cid]
            c_metrics = class_report_dict[key]
            writer.writerow([
                cid,
                meta["letter"],
                meta["name_en"],
                meta["name_ar"],
                round(c_metrics["precision"] * 100, 2),
                round(c_metrics["recall"] * 100, 2),
                round(c_metrics["f1-score"] * 100, 2),
                int(c_metrics["support"]),
            ])

    print(f"Saved per-class metrics to: {per_class_csv_path}")

    # Generate Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(28)))
    top_confusions = plot_confusion_matrix(cm)

    # Plot misclassified samples
    plot_misclassified_grid(misclassified_samples)

    # Save detailed JSON report
    report_json_path = EVALUATION_DIR / "baseline_report.json"
    report_data = {
        "model_architecture": "ArabicCNN (Lightweight Custom CNN)",
        "checkpoint": str(checkpoint_path.name),
        "test_dataset": "Hijja Isolated Child Handwriting (Test Split)",
        "test_samples_count": len(all_targets),
        "test_accuracy": round(acc, 2),
        "macro_precision": round(macro_prec, 2),
        "macro_recall": round(macro_rec, 2),
        "macro_f1": round(macro_f1, 2),
        "per_class_metrics": {
            f"{cid:02d}_{ID_TO_CHAR[cid]}": {
                "letter": ID_TO_CHAR[cid],
                "name_en": ID_TO_METADATA[cid]["name_en"],
                "precision": round(class_report_dict[f"{cid:02d}"]["precision"] * 100, 2),
                "recall": round(class_report_dict[f"{cid:02d}"]["recall"] * 100, 2),
                "f1_score": round(class_report_dict[f"{cid:02d}"]["f1-score"] * 100, 2),
                "support": int(class_report_dict[f"{cid:02d}"]["support"]),
            }
            for cid in range(28)
        },
        "top_confusions": top_confusions,
        "total_misclassified": len(misclassified_samples),
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f"Saved full test report to: {report_json_path}")
    return report_data


def plot_confusion_matrix(cm: np.ndarray) -> List[Dict]:
    fig, ax = plt.subplots(figsize=(14, 12))
    class_labels = [f"{i:02d} {ID_TO_METADATA[i]['name_en']}" for i in range(28)]

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_labels,
        yticklabels=class_labels,
        title="Baseline ArabicCNN — Confusion Matrix (Hijja 28 Child Classes)",
        ylabel="True Class",
        xlabel="Predicted Class",
    )

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                ax.text(
                    j, i, format(val, "d"),
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=7,
                )

    plt.tight_layout()
    cm_path = EVALUATION_DIR / "baseline_confusion_matrix.png"
    plt.savefig(cm_path, dpi=180)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # Identify top confused pairs (off-diagonal)
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


def plot_misclassified_grid(misclassified_samples: List[Dict], max_samples: int = 16):
    if not misclassified_samples:
        return

    n_show = min(max_samples, len(misclassified_samples))
    rows = int(np.ceil(n_show / 4))
    cols = min(4, n_show)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    fig.patch.set_facecolor("#f8f9fa")
    axes = np.array(axes).reshape(-1)

    for idx in range(n_show):
        sample = misclassified_samples[idx]
        ax = axes[idx]
        img = Image.open(sample["image_path"]).convert("L")
        ax.imshow(img, cmap="gray")
        title = f"True: {sample['true_letter']} ({ID_TO_METADATA[sample['true_id']]['name_en']})\nPred: {sample['pred_letter']} ({ID_TO_METADATA[sample['pred_id']]['name_en']})\nConf: {sample['confidence']*100:.1f}%"
        ax.set_title(title, fontsize=9, color="#dc2626", fontweight="bold")
        ax.axis("off")

    for idx in range(n_show, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Baseline ArabicCNN — Sample Misclassifications (Child Test Set)", fontsize=13, y=0.98)
    plt.tight_layout()
    grid_path = EVALUATION_DIR / "baseline_misclassified_samples.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved misclassified sample grid to: {grid_path}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    train_baseline_model()
