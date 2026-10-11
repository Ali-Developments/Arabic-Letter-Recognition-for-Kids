"""
Arabic Letter Classifier Inference Engine
Supports lightweight custom CNN and MobileNetV3 architectures in PyTorch.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.ml.labels import (
    ARABIC_LETTERS,
    LETTER_METADATA,
    NUM_CLASSES,
    ID_TO_METADATA,
    ID_TO_CHAR,
)
from app.ml.preprocessing import preprocess_image_for_model, decode_image_bytes
from app.ml.confidence import ConfidenceManager


from app.ml.architectures import ArabicCNN, ArabicMobileNetV3
from app.ml.preprocessing import preprocess_image_for_model, decode_image_bytes
from app.ml.confidence import ConfidenceManager


class ArabicPredictor:
    """
    High-level predictor that coordinates image decoding, preprocessing,
    PyTorch inference, confidence checking, and response formatting.
    Supports ArabicCNN and ArabicMobileNetV3 seamlessly.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        acceptance_threshold: float = 0.75,
    ):
        self.device = torch.device(device)
        self.model: nn.Module = ArabicCNN(num_classes=NUM_CLASSES).to(self.device)
        self.model_loaded = False
        self.model_version = "0.1.0-baseline"
        self.confidence_manager = ConfidenceManager(acceptance_threshold=acceptance_threshold)

        if model_path and os.path.exists(model_path):
            self.load_weights(model_path)
        else:
            # Check default model locations in order of priority
            models_dir = Path(__file__).resolve().parent.parent.parent.parent / "ml" / "models"
            candidates = [
                models_dir / "final_candidate_model.pt",
                models_dir / "baseline_arabic_cnn_best.pt",
                models_dir / "best_arabic_cnn.pt",
            ]
            for candidate in candidates:
                if candidate.exists():
                    self.load_weights(str(candidate))
                    break

        self.model.eval()

    def load_weights(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            training_cfg = checkpoint.get("training_config", {})
            arch_name = training_cfg.get("model_architecture", "")
            val_acc = checkpoint.get("validation_accuracy")
            epoch = checkpoint.get("best_epoch")
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
            arch_name = ""
            val_acc = None
            epoch = None
        else:
            raise ValueError(f"Unrecognized checkpoint format in {path}")

        # Detect architecture if not explicitly stated
        if not arch_name:
            if any(k.startswith("features.") for k in state_dict.keys()):
                arch_name = "ArabicMobileNetV3"
            else:
                arch_name = "ArabicCNN"

        if "MobileNet" in arch_name:
            self.model = ArabicMobileNetV3(num_classes=NUM_CLASSES, pretrained=False).to(self.device)
        else:
            self.model = ArabicCNN(num_classes=NUM_CLASSES).to(self.device)

        self.model.load_state_dict(state_dict)
        if val_acc and epoch:
            self.model_version = f"0.1.0-baseline-{arch_name}-epoch{epoch}-valacc{val_acc:.1f}"
        else:
            self.model_version = f"0.1.0-baseline-{arch_name}"

        self.model_loaded = True
        self.model.eval()

    def predict_from_bytes(
        self,
        image_bytes: bytes,
        top_k: int = 3,
        request_id: Optional[str] = None,
        save_debug: bool = True,
    ) -> Dict:
        """
        End-to-end prediction from raw image bytes.
        Returns dictionary formatted for API responses.
        Handles early rejection for blank paper, glare, extreme dark, or multiple letters.
        Saves step-by-step diagnostic images to debug/real_camera/<request_id>/ for auditing.
        """
        # 1. Decode image bytes
        cv_img = decode_image_bytes(image_bytes)

        # 2. Preprocess with Dot-Preserving Camera Pipeline
        preprocessed, prep_meta = preprocess_image_for_model(
            cv_img, target_size=(32, 32), return_metadata=True
        )

        # 2b. Debug saving for real-camera auditing
        if save_debug and request_id:
            try:
                import cv2
                debug_dir = Path(__file__).resolve().parent.parent.parent.parent / "debug" / "real_camera" / request_id
                debug_dir.mkdir(parents=True, exist_ok=True)

                # Save original received
                cv2.imwrite(str(debug_dir / "original_received.jpg"), cv_img)

                stages = prep_meta.get("stages", {})
                if "gray" in stages:
                    cv2.imwrite(str(debug_dir / "grayscale.png"), stages["gray"])
                if "binary" in stages:
                    cv2.imwrite(str(debug_dir / "threshold.png"), stages["binary"])
                if "cleaned" in stages:
                    cv2.imwrite(str(debug_dir / "cleaned.png"), stages["cleaned"])
                    # Create colored components visualization
                    num_cc, _, stats_cc, _ = cv2.connectedComponentsWithStats(stages["cleaned"])
                    comp_vis = cv2.cvtColor(stages["cleaned"], cv2.COLOR_GRAY2BGR)
                    for c_i in range(1, num_cc):
                        x, y, w, h = stats_cc[c_i, :4]
                        cv2.rectangle(comp_vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.imwrite(str(debug_dir / "components.png"), comp_vis)

                if "final" in stages:
                    # Save final model input (also resized to 28x28 if requested)
                    cv2.imwrite(str(debug_dir / "final_32x32.png"), stages["final"])
                    final_28 = cv2.resize(stages["final"], (28, 28), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(debug_dir / "final_28x28.png"), final_28)
            except Exception as e:
                # Debug saving failure must not crash inference
                pass

        # Early rejection handling (blank, multiple letters, extreme lighting)
        if prep_meta["status"] != "valid":
            return {
                "letter": None,
                "class_id": None,
                "confidence": 0.0,
                "accepted": False,
                "name_en": None,
                "name_ar": None,
                "message": prep_meta["message"] or "No clear handwritten letter detected.",
                "top_candidates": [],
                "request_id": request_id,
            }

        # 3. Convert to PyTorch Tensor [1, 1, 32, 32]
        tensor = torch.from_numpy(preprocessed).unsqueeze(0).unsqueeze(0).to(self.device)

        # 4. Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        top_indices = np.argsort(probs)[::-1]
        top1_idx = int(top_indices[0])
        top1_prob = float(probs[top1_idx])

        # 5. Evaluate Acceptance & Confidence Calibration
        accepted, message = self.confidence_manager.evaluate_prediction(probs, top_indices)

        top_candidates = []
        for rank in range(min(top_k, len(top_indices))):
            idx = int(top_indices[rank])
            meta = ID_TO_METADATA[idx]
            top_candidates.append({
                "letter": meta["letter"] if accepted else None,
                "class_id": idx if accepted else None,
                "confidence": round(float(probs[idx]), 4),
                "accepted": (rank == 0 and accepted),
                "name_en": meta["name_en"] if accepted else None,
                "name_ar": meta["name_ar"] if accepted else None,
            })

        meta_top1 = ID_TO_METADATA[top1_idx]
        return {
            "letter": meta_top1["letter"] if accepted else None,
            "class_id": top1_idx if accepted else None,
            "confidence": round(top1_prob, 4),
            "accepted": accepted,
            "name_en": meta_top1["name_en"] if accepted else None,
            "name_ar": meta_top1["name_ar"] if accepted else None,
            "message": message,
            "top_candidates": top_candidates,
            "request_id": request_id,
        }
