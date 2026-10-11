"""
Generate Structured Camera Stress Test Dataset (Step 4).
Generates realistic phone/tablet camera simulation captures based on actual child handwriting strokes.
Simulates white notebook paper, camera guide boxes, shadows, lighting gradients, perspective tilts,
pencil/pen textures, faint dots, clutter, blank sheets, and multi-letter invalid inputs.
All samples are explicitly labeled as synthetic stress tests in manifest.json.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path("backend").resolve()))
from app.ml.labels import ARABIC_LETTERS, ID_TO_CHAR, ID_TO_METADATA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIJJA_TEST_DIR = PROJECT_ROOT / "ml" / "data" / "processed" / "hijja" / "test"
OUTPUT_ROOT = PROJECT_ROOT / "tests" / "camera_samples"

CATEGORIES = [
    "clean",
    "shadow",
    "low_light",
    "bright_light",
    "rotated",
    "perspective",
    "pencil",
    "pen",
    "small_letter",
    "large_letter",
    "faint_dots",
    "clutter",
    "blank",
    "invalid",
]

# Representative letters for testing diverse topologies (vertical alef, basin baa/taa/thaa, cavity jeem/khaa, loop saad/daad, multi-dot sheen/noon/yaa)
TEST_CLASS_IDS = [0, 1, 2, 3, 4, 6, 8, 10, 11, 12, 13, 15, 16, 24, 27]  # ا, ب, ت, ث, ج, خ, ذ, ز, س, ش, ص, ط, ظ, ن, ي


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def create_paper_canvas(
    width: int = 480,
    height: int = 480,
    base_color: int = 245,
    noise_level: float = 3.0,
) -> np.ndarray:
    """Creates a base white/off-white paper texture with subtle paper grain."""
    canvas = np.full((height, width), base_color, dtype=np.float32)
    # Subtle paper grain
    grain = np.random.normal(0, noise_level, (height, width)).astype(np.float32)
    canvas = np.clip(canvas + grain, 0, 255).astype(np.uint8)

    # Convert to 3-channel BGR
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def get_stroke_mask_from_hijja(class_id: int, sample_idx: int = 0) -> np.ndarray:
    """Loads a real child handwriting stroke from the Hijja test split."""
    class_folder = HIJJA_TEST_DIR / f"{class_id:02d}"
    if not class_folder.exists():
        # Fallback to any class directory
        class_folder = next(HIJJA_TEST_DIR.glob(f"{class_id:02d}*"))

    samples = sorted(list(class_folder.glob("*.png")))
    chosen_path = samples[sample_idx % len(samples)]
    img = cv2.imread(str(chosen_path), cv2.IMREAD_GRAYSCALE)

    # Hijja images are white stroke on dark background
    if np.mean(img) > 127:
        img = 255 - img
    return img


def draw_letter_on_paper(
    paper: np.ndarray,
    stroke_32: np.ndarray,
    center_xy: Tuple[int, int] = (240, 240),
    scale_factor: float = 4.5,
    ink_color: Tuple[int, int, int] = (30, 30, 30),
    rotation_deg: float = 0.0,
    faint_dots: bool = False,
) -> np.ndarray:
    """Renders the 32x32 child stroke onto high-res paper canvas with realistic ink dynamics."""
    h_canvas, w_canvas = paper.shape[:2]

    # Resize stroke to high-resolution
    target_stroke_size = int(32 * scale_factor)
    highres_stroke = cv2.resize(
        stroke_32,
        (target_stroke_size, target_stroke_size),
        interpolation=cv2.INTER_CUBIC,
    )

    if rotation_deg != 0.0:
        rot_mat = cv2.getRotationMatrix2D(
            (target_stroke_size / 2, target_stroke_size / 2),
            rotation_deg,
            1.0,
        )
        highres_stroke = cv2.warpAffine(
            highres_stroke,
            rot_mat,
            (target_stroke_size, target_stroke_size),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )

    # Smooth anti-aliased stroke mask in [0.0, 1.0]
    stroke_mask = np.clip(highres_stroke.astype(np.float32) / 255.0, 0.0, 1.0)

    # Handle faint dots if requested
    if faint_dots:
        # Separate main body from top/bottom dots by connected components
        bin_for_cc = (highres_stroke > 30).astype(np.uint8) * 255
        n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(bin_for_cc)
        if n_cc > 2:
            max_cc = 1 + np.argmax(cc_stats[1:, cv2.CC_STAT_AREA])
            for cc_id in range(1, n_cc):
                if cc_id != max_cc:
                    # Fade the dot component
                    mask_dot = (cc_labels == cc_id)
                    highres_stroke[mask_dot] = (highres_stroke[mask_dot] * 0.4).astype(np.uint8)
        stroke_mask = np.clip(highres_stroke.astype(np.float32) / 255.0, 0.0, 1.0)

    # Place on canvas
    x_offset = center_xy[0] - target_stroke_size // 2
    y_offset = center_xy[1] - target_stroke_size // 2

    x1 = max(0, x_offset)
    y1 = max(0, y_offset)
    x2 = min(w_canvas, x_offset + target_stroke_size)
    y2 = min(h_canvas, y_offset + target_stroke_size)

    sx1 = max(0, -x_offset)
    sy1 = max(0, -y_offset)
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    sub_canvas = paper[y1:y2, x1:x2].astype(np.float32)
    sub_mask = stroke_mask[sy1:sy2, sx1:sx2, np.newaxis]

    # Alpha blend ink onto paper
    ink_bgr = np.array(ink_color, dtype=np.float32).reshape(1, 1, 3)
    blended = sub_canvas * (1.0 - sub_mask) + ink_bgr * sub_mask
    paper[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return paper


def apply_guide_box(paper: np.ndarray, color: Tuple[int, int, int] = (225, 225, 225)) -> np.ndarray:
    """Draws a subtle rounded camera guide box simulating paper guideline/frame."""
    h, w = paper.shape[:2]
    # Draw faint guideline frame at 12% margins
    m_x, m_y = int(w * 0.12), int(h * 0.12)
    cv2.rectangle(paper, (m_x, m_y), (w - m_x, h - m_y), color, 1, cv2.LINE_AA)
    return paper


def apply_shadow(paper: np.ndarray) -> np.ndarray:
    """Applies a realistic soft diagonal arm/phone shadow gradient."""
    h, w = paper.shape[:2]
    # Create diagonal gradient
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    diag = (x_coords + y_coords) / (h + w)  # 0.0 at top-left, 1.0 at bottom-right

    # Soft shadow band across the middle-bottom
    shadow_mask = 1.0 - 0.55 * np.exp(-((diag - 0.6) ** 2) / 0.08)
    shadow_mask = np.clip(shadow_mask, 0.45, 1.0)[:, :, np.newaxis]

    shadowed = (paper.astype(np.float32) * shadow_mask).astype(np.uint8)
    return shadowed


def apply_perspective_distortion(paper: np.ndarray) -> np.ndarray:
    """Applies realistic perspective keystoning (camera tilted relative to paper)."""
    h, w = paper.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    # Tilt: top is narrower and shifted
    pts2 = np.float32([[w * 0.08, h * 0.04], [w * 0.92, h * 0.06], [w * 0.02, h * 0.96], [w * 0.98, h * 0.95]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(paper, matrix, (w, h), borderValue=(220, 220, 220))
    return warped


def generate_all_samples():
    set_seed(42)
    for cat in CATEGORIES:
        (OUTPUT_ROOT / cat).mkdir(parents=True, exist_ok=True)

    manifest = []
    file_counter = 0

    print("Generating comprehensive camera stress test dataset...")

    # 1. CLEAN
    for cid in TEST_CLASS_IDS[:8]:
        paper = create_paper_canvas(base_color=245)
        paper = apply_guide_box(paper)
        stroke = get_stroke_mask_from_hijja(cid, sample_idx=0)
        paper = draw_letter_on_paper(paper, stroke, ink_color=(35, 35, 35), scale_factor=4.5)

        filename = f"clean_{cid:02d}_{ID_TO_METADATA[cid]['name_en']}.jpg"
        out_path = OUTPUT_ROOT / "clean" / filename
        cv2.imwrite(str(out_path), paper, [cv2.IMWRITE_JPEG_QUALITY, 90])

        manifest.append({
            "file": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "condition": "clean",
            "true_letter": ID_TO_CHAR[cid],
            "true_class_id": cid,
            "is_synthetic": True,
            "expected_behavior": "accept",
            "description": f"Crisp centered {ID_TO_METADATA[cid]['name_en']} under even room lighting.",
        })

    # 2. SHADOW
    for cid in TEST_CLASS_IDS[:6]:
        paper = create_paper_canvas(base_color=240)
        paper = apply_guide_box(paper)
        stroke = get_stroke_mask_from_hijja(cid, sample_idx=1)
        paper = draw_letter_on_paper(paper, stroke, ink_color=(35, 35, 35), scale_factor=4.5)
        paper = apply_shadow(paper)

        filename = f"shadow_{cid:02d}_{ID_TO_METADATA[cid]['name_en']}.jpg"
        out_path = OUTPUT_ROOT / "shadow" / filename
        cv2.imwrite(str(out_path), paper, [cv2.IMWRITE_JPEG_QUALITY, 88])

        manifest.append({
            "file": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "condition": "shadow",
            "true_letter": ID_TO_CHAR[cid],
            "true_class_id": cid,
            "is_synthetic": True,
            "expected_behavior": "accept",
            "description": f"Diagonal hand/desk cast shadow over {ID_TO_METADATA[cid]['name_en']}.",
        })

    # 3. LOW LIGHT
    for cid in TEST_CLASS_IDS[2:7]:
        paper = create_paper_canvas(base_color=130, noise_level=6.0)
        paper = apply_guide_box(paper, color=(100, 100, 100))
        stroke = get_stroke_mask_from_hijja(cid, sample_idx=2)
        paper = draw_letter_on_paper(paper, stroke, ink_color=(20, 20, 20), scale_factor=4.2)

        filename = f"lowlight_{cid:02d}_{ID_TO_METADATA[cid]['name_en']}.jpg"
        out_path = OUTPUT_ROOT / "low_light" / filename
        cv2.imwrite(str(out_path), paper, [cv2.IMWRITE_JPEG_QUALITY, 85])

        manifest.append({
            "file": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "condition": "low_light",
            "true_letter": ID_TO_CHAR[cid],
            "true_class_id": cid,
            "is_synthetic": True,
            "expected_behavior": "accept",
            "description": f"Dim lighting underexposed capture for {ID_TO_METADATA[cid]['name_en']}.",
        })

    # 4. BRIGHT LIGHT / GLARE
    for cid in TEST_CLASS_IDS[4:8]:
        paper = create_paper_canvas(base_color=252)
        paper = apply_guide_box(paper, color=(220, 220, 220))
        stroke = get_stroke_mask_from_hijja(cid, sample_idx=3)
        # Bright light hotspot
        h, w = paper.shape[:2]
        cv2.circle(paper, (180, 180), 120, (255, 255, 255), -1)
        paper = cv2.GaussianBlur(paper, (35, 35), 0)
        paper = draw_letter_on_paper(paper, stroke, ink_color=(45, 45, 45), scale_factor=4.5)

        filename = f"bright_{cid:02d}_{ID_TO_METADATA[cid]['name_en']}.jpg"
        out_path = OUTPUT_ROOT / "bright_light" / filename
        cv2.imwrite(str(out_path), paper, [cv2.IMWRITE_JPEG_QUALITY, 90])

        manifest.append({
            "file": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "condition": "bright_light",
            "true_letter": ID_TO_CHAR[cid],
            "true_class_id": cid,
            "is_synthetic": True,
            "expected_behavior": "accept",
            "description": f"Bright desk lamp hotspot glare for {ID_TO_METADATA[cid]['name_en']}.",
        })

    # 5. ROTATED
    for cid in [1, 4, 11, 13, 24]:
        paper = create_paper_canvas(base_color=245)
        paper = apply_guide_box(paper)
        stroke = get_stroke_mask_from_hijja(cid, sample_idx=0)
        angle = random.choice([-15.0, 14.0, -12.0, 16.0])
        paper = draw_letter_on_paper(paper, stroke, rotation_deg=angle, scale_factor=4.5)

        filename = f"rotated_{cid:02d}_{ID_TO_METADATA[cid]['name_en']}.jpg"
        out_path = OUTPUT_ROOT / "rotated" / filename
        cv2.imwrite(str(out_path), paper, [cv2.IMWRITE_JPEG_QUALITY, 90])

        manifest.append({
            "file": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "condition": "rotated",
            "true_letter": ID_TO_CHAR[cid],
            "true_class_id": cid,
            "is_synthetic": True,
            "expected_behavior": "accept",
            "description": f"{ID_TO_METADATA[cid]['name_en']} rotated by {angle:.1f} degrees.",
        })

    # 6. PERSPECTIVE
    for cid in [0, 2, 8, 12, 27]:
        paper = create_paper_canvas(base_color=242)
        paper = apply_guide_box(paper)
        stroke = get_stroke_mask_from_hijja(cid, sample_idx=1)
        paper = draw_letter_on_paper(paper, stroke, scale_factor=4.5)
        paper = apply_perspective_distortion(paper)

        filename = f"perspective_{cid:02d}_{ID_TO_METADATA[cid]['name_en']}.jpg"
        out_path = OUTPUT_ROOT / "perspective" / filename
        cv2.imwrite(str(out_path), paper, [cv2.IMWRITE_JPEG_QUALITY, 88])

        manifest.append({
            "file": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "condition": "perspective",
            "true_letter": ID_TO_CHAR[cid],
            "true_class_id": cid,
            "is_synthetic": True,
            "expected_behavior": "accept",
            "description": f"Perspective trapezoidal tilt for {ID_TO_METADATA[cid]['name_en']}.",
        })

    # 7. PENCIL TEXTURE
    for cid in [1, 2, 4, 13, 16]:
        paper = create_paper_canvas(base_color=240, noise_level=4.0)
        paper = apply_guide_box(paper)
        stroke = get_stroke_mask_from_hijja(cid, sample_idx=2)
        paper = draw_letter_on_paper(paper, stroke, ink_color=(80, 80, 80), scale_factor=4.5)

        filename = f"pencil_{cid:02d}_{ID_TO_METADATA[cid]['name_en']}.jpg"
        out_path = OUTPUT_ROOT / "pencil" / filename
        cv2.imwrite(str(out_path), paper, [cv2.IMWRITE_JPEG_QUALITY, 85])

        manifest.append({
            "file": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "condition": "pencil",
            "true_letter": ID_TO_CHAR[cid],
            "true_class_id": cid,
            "is_synthetic": True,
            "expected_behavior": "accept",
            "description": f"Grainy gray graphite pencil stroke for {ID_TO_METADATA[cid]['name_en']}.",
        })

    # 8. PEN (BLUE/BLACK INK)
    for cid in [0, 3, 6, 15, 24]:
        paper = create_paper_canvas(base_color=248)
        paper = apply_guide_box(paper)
        stroke = get_stroke_mask_from_hijja(cid, sample_idx=0)
        paper = draw_letter_on_paper(paper, stroke, ink_color=(120, 30, 20), scale_factor=4.8)  # Blue ballpoint ink

        filename = f"pen_{cid:02d}_{ID_TO_METADATA[cid]['name_en']}.jpg"
        out_path = OUTPUT_ROOT / "pen" / filename
        cv2.imwrite(str(out_path), paper, [cv2.IMWRITE_JPEG_QUALITY, 90])

        manifest.append({
            "file": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "condition": "pen",
            "true_letter": ID_TO_CHAR[cid],
            "true_class_id": cid,
            "is_synthetic": True,
            "expected_behavior": "accept",
            "description": f"Deep blue ballpoint pen stroke for {ID_TO_METADATA[cid]['name_en']}.",
        })

    # 9. SMALL LETTER
    for cid in [1, 7, 10, 11]:
        paper = create_paper_canvas(base_color=245)
        paper = apply_guide_box(paper)
        stroke = get_stroke_mask_from_hijja(cid, sample_idx=1)
        paper = draw_letter_on_paper(paper, stroke, scale_factor=2.0)  # Very small

        filename = f"small_{cid:02d}_{ID_TO_METADATA[cid]['name_en']}.jpg"
        out_path = OUTPUT_ROOT / "small_letter" / filename
        cv2.imwrite(str(out_path), paper, [cv2.IMWRITE_JPEG_QUALITY, 90])

        manifest.append({
            "file": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "condition": "small_letter",
            "true_letter": ID_TO_CHAR[cid],
            "true_class_id": cid,
            "is_synthetic": True,
            "expected_behavior": "accept",
            "description": f"Small handwriting {ID_TO_METADATA[cid]['name_en']} occupying ~18% frame.",
        })

    # 10. LARGE LETTER
    for cid in [0, 4, 12, 13]:
        paper = create_paper_canvas(base_color=245)
        paper = apply_guide_box(paper)
        stroke = get_stroke_mask_from_hijja(cid, sample_idx=0)
        paper = draw_letter_on_paper(paper, stroke, scale_factor=6.8)  # Very large

        filename = f"large_{cid:02d}_{ID_TO_METADATA[cid]['name_en']}.jpg"
        out_path = OUTPUT_ROOT / "large_letter" / filename
        cv2.imwrite(str(out_path), paper, [cv2.IMWRITE_JPEG_QUALITY, 90])

        manifest.append({
            "file": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "condition": "large_letter",
            "true_letter": ID_TO_CHAR[cid],
            "true_class_id": cid,
            "is_synthetic": True,
            "expected_behavior": "accept",
            "description": f"Large handwriting {ID_TO_METADATA[cid]['name_en']} filling guide frame.",
        })

    # 11. FAINT DOTS (CRITICAL ARABIC TEST)
    for cid in [1, 2, 3, 4, 6, 8, 16, 27]:  # baa, taa, thaa, jeem, khaa, dhaal, dhah, yaa
        paper = create_paper_canvas(base_color=244)
        paper = apply_guide_box(paper)
        stroke = get_stroke_mask_from_hijja(cid, sample_idx=1)
        paper = draw_letter_on_paper(paper, stroke, ink_color=(45, 45, 45), scale_factor=4.5, faint_dots=True)

        filename = f"faint_dots_{cid:02d}_{ID_TO_METADATA[cid]['name_en']}.jpg"
        out_path = OUTPUT_ROOT / "faint_dots" / filename
        cv2.imwrite(str(out_path), paper, [cv2.IMWRITE_JPEG_QUALITY, 90])

        manifest.append({
            "file": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "condition": "faint_dots",
            "true_letter": ID_TO_CHAR[cid],
            "true_class_id": cid,
            "is_synthetic": True,
            "expected_behavior": "accept",
            "description": f"Lightly pressed faint dots on {ID_TO_METADATA[cid]['name_en']}.",
        })

    # 12. CLUTTER / MARGIN NOISE
    for cid in [1, 8, 11, 24]:
        paper = create_paper_canvas(base_color=240)
        paper = apply_guide_box(paper)
        stroke = get_stroke_mask_from_hijja(cid, sample_idx=0)
        paper = draw_letter_on_paper(paper, stroke, scale_factor=4.5)
        # Add distant ruler/line and speckles outside frame
        cv2.line(paper, (10, 460), (470, 460), (140, 140, 140), 1)
        cv2.circle(paper, (30, 40), 4, (60, 60, 60), -1)
        cv2.circle(paper, (450, 430), 5, (60, 60, 60), -1)

        filename = f"clutter_{cid:02d}_{ID_TO_METADATA[cid]['name_en']}.jpg"
        out_path = OUTPUT_ROOT / "clutter" / filename
        cv2.imwrite(str(out_path), paper, [cv2.IMWRITE_JPEG_QUALITY, 88])

        manifest.append({
            "file": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "condition": "clutter",
            "true_letter": ID_TO_CHAR[cid],
            "true_class_id": cid,
            "is_synthetic": True,
            "expected_behavior": "accept",
            "description": f"Distant ruler margin and desk speckles around {ID_TO_METADATA[cid]['name_en']}.",
        })

    # 13. BLANK PAPER
    for b_idx in range(4):
        paper = create_paper_canvas(base_color=245)
        paper = apply_guide_box(paper)
        filename = f"blank_{b_idx + 1}.jpg"
        out_path = OUTPUT_ROOT / "blank" / filename
        cv2.imwrite(str(out_path), paper, [cv2.IMWRITE_JPEG_QUALITY, 90])

        manifest.append({
            "file": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "condition": "blank",
            "true_letter": None,
            "true_class_id": None,
            "is_synthetic": True,
            "expected_behavior": "reject",
            "description": "Completely blank notebook paper inside guide frame.",
        })

    # 14. INVALID INPUTS (All-black, Overexposed, Multiple Letters)
    # A: Completely black image
    black_img = np.zeros((480, 480, 3), dtype=np.uint8)
    out_path = OUTPUT_ROOT / "invalid" / "invalid_black_frame.jpg"
    cv2.imwrite(str(out_path), black_img)
    manifest.append({
        "file": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "condition": "invalid",
        "true_letter": None,
        "true_class_id": None,
        "is_synthetic": True,
        "expected_behavior": "reject",
        "description": "Pitch black camera occlusion (e.g. finger over camera lens).",
    })

    # B: Blown out white canvas
    white_img = np.full((480, 480, 3), 255, dtype=np.uint8)
    out_path = OUTPUT_ROOT / "invalid" / "invalid_white_washout.jpg"
    cv2.imwrite(str(out_path), white_img)
    manifest.append({
        "file": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "condition": "invalid",
        "true_letter": None,
        "true_class_id": None,
        "is_synthetic": True,
        "expected_behavior": "reject",
        "description": "Completely washed out blank white image without features.",
    })

    # C: Multiple letters (baa and taa side-by-side)
    paper = create_paper_canvas(base_color=245)
    paper = apply_guide_box(paper)
    stroke_baa = get_stroke_mask_from_hijja(1, sample_idx=0)
    stroke_taa = get_stroke_mask_from_hijja(2, sample_idx=0)
    paper = draw_letter_on_paper(paper, stroke_baa, center_xy=(160, 240), scale_factor=3.5)
    paper = draw_letter_on_paper(paper, stroke_taa, center_xy=(320, 240), scale_factor=3.5)
    out_path = OUTPUT_ROOT / "invalid" / "invalid_multiple_letters.jpg"
    cv2.imwrite(str(out_path), paper)
    manifest.append({
        "file": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "condition": "invalid",
        "true_letter": None,
        "true_class_id": None,
        "is_synthetic": True,
        "expected_behavior": "reject",
        "description": "Multiple separate letters (baa + taa) written inside frame.",
    })

    # Save manifest.json
    manifest_path = OUTPUT_ROOT / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(manifest)} camera test samples across 14 categories.")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    generate_all_samples()
