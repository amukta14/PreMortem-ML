from typing import Dict, Callable, Optional, Union
import numpy as np
from numpy.typing import ArrayLike

from premortemml.internal.neighbor.metric import decide_euclidean_metric
from premortemml.internal.neighbor.knn_graph import features_to_knn
from premortemml.outlier import OutOfDistribution
from premortemml.internal.regression_utils import assert_valid_prediction_inputs

from premortemml.internal.constants import TINY_VALUE

def get_label_quality_scores(
    labels: ArrayLike,
    predictions: ArrayLike,
    *,
    method: str = "outre",
) -> np.ndarray:
    

    # Check if inputs are valid
    labels, predictions = assert_valid_prediction_inputs(
        labels=labels, predictions=predictions, method=method
    )

    scoring_funcs: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        "residual": _get_residual_score_for_each_label,
        "outre": _get_outre_score_for_each_label,
    }

    scoring_func = scoring_funcs.get(method, None)
    if not scoring_func:
        raise ValueError(
            f
        )

    # Calculate scores
    label_quality_scores = scoring_func(labels, predictions)
    return label_quality_scores

def _get_residual_score_for_each_label(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    
    residual = predictions - labels
    label_quality_scores = np.exp(-abs(residual))
    return label_quality_scores

def _get_outre_score_for_each_label(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    residual_scale: float = 5,
    frac_neighbors: float = 0.5,
    neighbor_metric: Optional[Union[str, Callable]] = None,
) -> np.ndarray:
    
    residual = predictions - labels
    labels = (labels - labels.mean()) / (labels.std() + TINY_VALUE)
    residual = residual_scale * ((residual - residual.mean()) / (residual.std() + TINY_VALUE))

    # 2D features by combining labels and residual
    features = np.array([labels, residual]).T

    neighbors = int(np.ceil(frac_neighbors * labels.shape[0]))
    # Use provided metric or select a decent implementation of the euclidean metric for knn search
    neighbor_metric = neighbor_metric or decide_euclidean_metric(features)
    knn = features_to_knn(features, n_neighbors=neighbors, metric=neighbor_metric)
    ood = OutOfDistribution(params={"knn": knn})

    label_quality_scores = ood.score(features=features)
    return label_quality_scores
