"""
Image Preprocessing Pipeline for Arabic Handwritten Letter Recognition
Adheres strictly to ARABIC_HANDWRITING_AI_PROJECT_BRIEF.md (Sections 8 & 9)
and STEP 4 Real Camera Robustness specifications.

Core principles:
1. Robust under non-uniform phone/tablet camera lighting and cast shadows.
2. Filter out camera guide frames, notebook borders, and distant dust noise.
3. Strictly preserve Arabic dots (1, 2, or 3 dots above, below, or inside).
4. Reject blank sheets, extreme glare/underexposure, or multiple letters.
5. Proportional scaling with margin=8 to match the Hijja training distribution.
"""

from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np


def preprocess_image_for_model(
    image: np.ndarray,
    target_size: Tuple[int, int] = (32, 32),
    invert_if_needed: bool = True,
    padding: int = 8,
    return_metadata: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Full real-world camera image preprocessing pipeline:
    1. Grayscale conversion & Fast-path for 32x32 clean inputs
    2. Invalid illumination check (solid dark or pure blank)
    3. Bilateral smoothing for noise suppression while preserving crisp stroke edges
    4. Adaptive Gaussian Thresholding (illumination & shadow invariant)
    5. Connected Component Analysis:
       - Guide frame / border component filtering
       - Blank sheet rejection (foreground pixels < 25)
       - Multiple disconnected letter rejection
       - Arabic dot preservation with spatial envelope
    6. Aspect-ratio preserving square resize & centering (target_size x target_size)
    7. Float32 normalization into [0.0, 1.0]

    Args:
        image: Input image as BGR or Grayscale numpy array
        target_size: Target dimension (default 32x32)
        invert_if_needed: Ensure white foreground (character) on black background
        padding: Margin pixels around character (default 8 to match Hijja letter scale)
        return_metadata: If True, returns (normalized_array, metadata_dict)

    Returns:
        Normalized float32 numpy array, or (array, metadata) if return_metadata=True
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty image provided to preprocessor.")

    meta: Dict = {
        "status": "valid",
        "message": None,
        "dots_preserved": 0,
        "body_box": None,
        "stages": {},
    }

    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    meta["stages"]["gray"] = gray
    h, w = gray.shape

    # Fast path: if image is already 32x32 (e.g. clean dataset test image)
    if h == target_size[0] and w == target_size[1]:
        border = np.concatenate([gray[0, :], gray[h - 1, :], gray[:, 0], gray[:, w - 1]])
        if np.mean(border) > 127:
            normalized = (255.0 - gray.astype(np.float32)) / 255.0
        else:
            normalized = gray.astype(np.float32) / 255.0
        meta["stages"]["final"] = (normalized * 255).astype(np.uint8)
        if return_metadata:
            return normalized, meta
        return normalized

    # 2. Lighting & Blank Validation Checks
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    if mean_val > 250 and std_val < 4.0:
        meta["status"] = "blank"
        meta["message"] = "No clear handwritten letter detected. The paper appears blank."
        zeros = np.zeros(target_size, dtype=np.float32)
        return (zeros, meta) if return_metadata else zeros

    if mean_val < 20:
        meta["status"] = "invalid_lighting"
        meta["message"] = "The image is too dark. Please use better lighting."
        zeros = np.zeros(target_size, dtype=np.float32)
        return (zeros, meta) if return_metadata else zeros

    # 3. Bilateral smoothing for edge-preserving noise reduction
    smoothed = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)
    meta["stages"]["smoothed"] = smoothed

    # 4. Adaptive Gaussian Thresholding (Winning Variant B)
    # Robust against non-uniform shadows and lighting gradients across the paper
    binary = cv2.adaptiveThreshold(
        smoothed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=10,
    )

    # Polarity check: border pixels should be dark (0) in binary
    border_pixels = np.concatenate([
        binary[0, :], binary[h - 1, :], binary[:, 0], binary[:, w - 1]
    ])
    if np.mean(border_pixels) > 127:
        binary = cv2.bitwise_not(binary)

    meta["stages"]["binary"] = binary

    # 5. Connected Component Analysis with Dot Preservation & Frame Filtering
    cleaned_binary, cc_info = extract_character_with_dots(binary, return_metadata=True)
    meta.update(cc_info)
    meta["stages"]["cleaned"] = cleaned_binary

    if meta["status"] != "valid":
        zeros = np.zeros(target_size, dtype=np.float32)
        return (zeros, meta) if return_metadata else zeros

    # 6. Extract Bounding Box of all preserved components
    coords = cv2.findNonZero(cleaned_binary)
    if coords is None:
        meta["status"] = "blank"
        meta["message"] = "No clear handwritten letter detected."
        zeros = np.zeros(target_size, dtype=np.float32)
        return (zeros, meta) if return_metadata else zeros

    x, y, bw, bh = cv2.boundingRect(coords)
    cropped = cleaned_binary[y:y + bh, x:x + bw]

    # 7. Aspect-Ratio Preserving Resize with Center Padding (margin=8 matches Hijja stroke scale)
    padded = fit_to_square(cropped, target_size=target_size[0], margin=padding)
    meta["stages"]["final"] = padded

    # 8. Normalize to float [0.0, 1.0]
    normalized = padded.astype(np.float32) / 255.0

    if return_metadata:
        return normalized, meta
    return normalized


def extract_character_with_dots(
    binary_img: np.ndarray,
    return_metadata: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Extracts the main character body and all associated Arabic dots.
    Filters out camera guide frames, blank paper, and multiple letters.

    Returns:
        cleaned_binary_mask (or (cleaned_binary_mask, info_dict) if return_metadata=True)
    """
    total_fg_pixels = cv2.countNonZero(binary_img)
    if total_fg_pixels < 25:
        info = {
            "status": "blank",
            "message": "No clear handwritten letter detected.",
            "dots_preserved": 0,
            "body_box": None,
        }
        return (np.zeros_like(binary_img), info) if return_metadata else np.zeros_like(binary_img)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )

    if num_labels <= 1:
        info = {
            "status": "blank",
            "message": "No clear handwritten letter detected.",
            "dots_preserved": 0,
            "body_box": None,
        }
        return (np.zeros_like(binary_img), info) if return_metadata else np.zeros_like(binary_img)

    img_h, img_w = binary_img.shape[:2]
    areas = stats[1:, cv2.CC_STAT_AREA]
    sorted_cc_indices = np.argsort(areas)[::-1] + 1  # 1-indexed

    # Filter out outer border / guide frame components (spans > 60% in both dimensions, fill ratio < 0.15, and has pixels at all 4 corners)
    valid_cc_indices = []
    for idx in sorted_cc_indices:
        cx = stats[idx, cv2.CC_STAT_LEFT]
        cy = stats[idx, cv2.CC_STAT_TOP]
        cw = stats[idx, cv2.CC_STAT_WIDTH]
        ch = stats[idx, cv2.CC_STAT_HEIGHT]
        c_area = stats[idx, cv2.CC_STAT_AREA]

        is_large = (cw > 0.60 * img_w and ch > 0.60 * img_h)
        is_hollow = ((c_area / float(cw * ch)) < 0.15)
        if is_large and is_hollow:
            comp_mask = (labels == idx)
            pad = min(8, min(cw, ch) // 4)
            c1 = np.any(comp_mask[cy:cy + pad, cx:cx + pad])
            c2 = np.any(comp_mask[cy:cy + pad, cx + cw - pad:cx + cw])
            c3 = np.any(comp_mask[cy + ch - pad:cy + ch, cx:cx + pad])
            c4 = np.any(comp_mask[cy + ch - pad:cy + ch, cx + cw - pad:cx + cw])
            if c1 and c2 and c3 and c4:
                continue  # Guide frame / border box
        valid_cc_indices.append(idx)

    if not valid_cc_indices:
        info = {
            "status": "blank",
            "message": "No clear handwritten letter detected.",
            "dots_preserved": 0,
            "body_box": None,
        }
        return (np.zeros_like(binary_img), info) if return_metadata else np.zeros_like(binary_img)

    max_label = valid_cc_indices[0]
    max_area = stats[max_label, cv2.CC_STAT_AREA]

    # Multiple Letter Check:
    # If second largest valid component is also substantial (> 35% of max) and horizontally separated
    if len(valid_cc_indices) > 1:
        second_label = valid_cc_indices[1]
        second_area = stats[second_label, cv2.CC_STAT_AREA]
        if second_area > 0.35 * max_area and second_area > 150:
            c1_x, c1_w = stats[max_label, cv2.CC_STAT_LEFT], stats[max_label, cv2.CC_STAT_WIDTH]
            c2_x, c2_w = stats[second_label, cv2.CC_STAT_LEFT], stats[second_label, cv2.CC_STAT_WIDTH]
            h_distance = abs((c1_x + c1_w / 2.0) - (c2_x + c2_w / 2.0))
            min_width = min(c1_w, c2_w)
            if h_distance > min_width * 1.2:
                info = {
                    "status": "multiple_letters",
                    "message": "Please write only one letter inside the frame.",
                    "dots_preserved": 0,
                    "body_box": None,
                }
                return (np.zeros_like(binary_img), info) if return_metadata else np.zeros_like(binary_img)

    body_x = stats[max_label, cv2.CC_STAT_LEFT]
    body_y = stats[max_label, cv2.CC_STAT_TOP]
    body_w = stats[max_label, cv2.CC_STAT_WIDTH]
    body_h = stats[max_label, cv2.CC_STAT_HEIGHT]

    # Generous envelope for Arabic dots (dots can be placed above, below, or slightly offset)
    x_expand = int(body_w * 0.7)
    y_expand = int(body_h * 1.3)

    env_x_min = max(0, body_x - x_expand)
    env_x_max = min(img_w, body_x + body_w + x_expand)
    env_y_min = max(0, body_y - y_expand)
    env_y_max = min(img_h, body_y + body_h + y_expand)

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

    info = {
        "status": "valid",
        "message": None,
        "dots_preserved": dots_preserved,
        "body_box": (int(body_x), int(body_y), int(body_w), int(body_h)),
    }
    return (kept_mask, info) if return_metadata else kept_mask


def fit_to_square(img: np.ndarray, target_size: int = 32, margin: int = 8) -> np.ndarray:
    """
    Fits an arbitrary bounding box image into a target_size x target_size square canvas
    while strictly preserving aspect ratio and centering the stroke.
    Default margin=8 yields usable size 16x16 matching the Hijja dataset stroke distribution.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    usable_size = max(4, target_size - (2 * margin))

    # Scale while keeping aspect ratio
    scale = usable_size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Canvas
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)

    # Centering offsets
    start_x = (target_size - new_w) // 2
    start_y = (target_size - new_h) // 2

    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
    return canvas


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decodes raw image bytes safely into a numpy BGR/Grayscale array."""
    np_buf = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes. Unsupported or corrupted file format.")
    return img
