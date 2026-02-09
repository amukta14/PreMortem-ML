from typing import Optional
import numpy as np

from premortemml.internal.constants import EPSILON

def transform_distances_to_scores(
    avg_distances: np.ndarray, t: int, scaling_factor: float
) -> np.ndarray:
    
    # Map ood_features_scores to range 0-1 with 0 = most concerning
    return np.exp(-t * avg_distances / max(scaling_factor, EPSILON))

def correct_precision_errors(
    scores: np.ndarray,
    avg_distances: np.ndarray,
    metric: str,
    C: int = 100,
    p: Optional[int] = None,
):
    
    if metric == "cosine":
        tolerance = C * np.finfo(np.float64).epsneg
    elif metric == "euclidean":
        tolerance = np.sqrt(C * np.finfo(np.float64).eps)
    elif metric == "minkowski":
        if p is None:
            raise ValueError("When metric is 'minkowski' you must specify the 'p' parameter")
        tolerance = (C * np.finfo(np.float64).eps) ** (1 / p)
    else:
        return scores

    candidates_mask = avg_distances < tolerance
    scores[candidates_mask] = 1
    return scores
