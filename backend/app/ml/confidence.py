"""
Confidence Evaluation and Calibration for Arabic Letter Classifier
Adheres to Section 11 of ARABIC_HANDWRITING_AI_PROJECT_BRIEF.md.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

DEFAULT_ACCEPTANCE_THRESHOLD = 0.75
DEFAULT_MIN_MARGIN_THRESHOLD = 0.20


class ConfidenceManager:
    """
    Manages acceptance thresholds, margin between top-1 and top-2 predictions,
    and returns informative guidance messages to children when re-capture is necessary.
    """

    def __init__(
        self,
        acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
        min_margin_threshold: float = DEFAULT_MIN_MARGIN_THRESHOLD,
    ):
        self.acceptance_threshold = acceptance_threshold
        self.min_margin_threshold = min_margin_threshold

    def evaluate_prediction(
        self,
        probabilities: np.ndarray,
        top_indices: np.ndarray,
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates whether a prediction should be accepted.

        Args:
            probabilities: Softmax probability distribution (length 28)
            top_indices: Sorted indices in descending order of probability

        Returns:
            (accepted: bool, feedback_message: Optional[str])
        """
        top1_prob = float(probabilities[top_indices[0]])
        top2_prob = float(probabilities[top_indices[1]]) if len(top_indices) > 1 else 0.0
        margin = top1_prob - top2_prob

        # Case 1: Overall confidence too low
        if top1_prob < self.acceptance_threshold:
            return False, (
                f"The handwriting was not clear enough ({top1_prob*100:.0f}% confidence). "
                "Please hold the paper steady under good light and try again!"
            )

        # Case 2: Ambiguous prediction (close tie between top 2 classes)
        if margin < self.min_margin_threshold:
            return False, (
                "The letter looks ambiguous. Please ensure the dots and strokes are clearly separated."
            )

        return True, "Letter recognized successfully."
