import warnings
import numpy as np
from typing import Optional
from scipy.special import xlogy

from premortemml.count import get_confident_thresholds

def _subtract_confident_thresholds(
    labels: Optional[np.ndarray],
    pred_probs: np.ndarray,
    multi_label: bool = False,
    confident_thresholds: Optional[np.ndarray] = None,
) -> np.ndarray:
    
    # Get expected (average) self-confidence for each class
    if confident_thresholds is None:
        if labels is None:
            raise ValueError(
                "Cannot calculate confident_thresholds without labels. Pass in either labels or already calculated "
                "confident_thresholds parameter. "
            )
        confident_thresholds = get_confident_thresholds(labels, pred_probs, multi_label=multi_label)

    # Subtract the class confident thresholds
    pred_probs_adj = pred_probs - confident_thresholds

    # Re-normalize by shifting data to take care of negative values from the subtraction
    pred_probs_adj += confident_thresholds.max()
    pred_probs_adj /= pred_probs_adj.sum(axis=1, keepdims=True)

    return pred_probs_adj

def get_normalized_entropy(
    pred_probs: np.ndarray, min_allowed_prob: Optional[float] = None
) -> np.ndarray:
    
    if np.any(pred_probs < 0) or np.any(pred_probs > 1):
        raise ValueError("All probabilities are required to be in the interval [0, 1].")
    num_classes = pred_probs.shape[1]

    if min_allowed_prob is not None:
        warnings.warn(
            "Using `min_allowed_prob` is not necessary anymore and will be removed.",
            DeprecationWarning,
        )
        pred_probs = np.clip(pred_probs, a_min=min_allowed_prob, a_max=None)

    # Note that dividing by log(num_classes) changes the base of the log which rescales entropy to 0-1 range
    return -np.sum(xlogy(pred_probs, pred_probs), axis=1) / np.log(num_classes)
