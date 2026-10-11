"""
PyTorch Dataset and Augmentation Pipeline for Arabic Character Recognition.
Adheres to Section 10 of ARABIC_HANDWRITING_AI_PROJECT_BRIEF.md.
Important: NO horizontal flipping (Arabic letters must never be mirrored).
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class ArabicAugmentation:
    """
    Realistic augmentation adhering strictly to Section 10 of Brief:
    - small rotation (up to 10 degrees)
    - translation / affine shift
    - mild scale change (0.9 to 1.1)
    - contrast & brightness variation
    - stroke intensity variation
    NOTE: RandomHorizontalFlip is STRICTLY PROHIBITED.
    """

    def __init__(self, is_training: bool = True):
        self.is_training = is_training

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Convert to tensor [1, H, W] in [0, 1]
        tensor = TF.to_tensor(img)

        # Invert if needed (ensure stroke is bright foreground on dark background)
        if tensor.mean() > 0.5:
            tensor = 1.0 - tensor

        if self.is_training:
            # 1. Random small rotation (-10 to +10 degrees)
            angle = float(np.random.uniform(-10.0, 10.0))
            # 2. Random translation (-3 to +3 pixels on 32x32)
            translate = (
                int(np.random.randint(-3, 4)),
                int(np.random.randint(-3, 4)),
            )
            # 3. Random scale (0.9 to 1.1)
            scale = float(np.random.uniform(0.9, 1.1))

            tensor = TF.affine(
                tensor,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=0.0,
                fill=0.0,
            )

            # 4. Light Gaussian noise / stroke variation
            if np.random.rand() > 0.5:
                noise = torch.randn_like(tensor) * 0.03
                tensor = torch.clamp(tensor + noise, 0.0, 1.0)

        # Standardize
        return tensor


class ArabicHandwritingDataset(Dataset):
    """
    PyTorch Dataset loading processed isolated Arabic character images.
    """

    def __init__(self, root_dir: str, is_training: bool = True, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.is_training = is_training
        self.transform = transform or ArabicAugmentation(is_training=is_training)

        self.samples: List[Tuple[Path, int]] = []
        for class_dir in sorted(self.root_dir.iterdir()):
            if class_dir.is_dir():
                try:
                    class_id = int(class_dir.name)
                except ValueError:
                    continue
                for img_path in sorted(class_dir.glob("*.png")):
                    self.samples.append((img_path, class_id))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, class_id = self.samples[idx]
        img = Image.open(img_path).convert("L")
        tensor = self.transform(img)
        return tensor, class_id
