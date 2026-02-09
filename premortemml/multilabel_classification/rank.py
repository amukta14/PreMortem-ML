from __future__ import annotations

import numpy as np  # noqa: F401: Imported for type annotations
from typing import List, TypeVar, Dict, Any, Optional, Tuple, TYPE_CHECKING

from premortemml.internal.validation import assert_valid_inputs
from premortemml.internal.util import get_num_classes
from premortemml.internal.multilabel_utils import int2onehot
from premortemml.internal.multilabel_scorer import MultilabelScorer, ClassLabelScorer, Aggregator

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    T = TypeVar("T", bound=npt.NBitBase)

def _labels_to_binary(
    labels: List[List[int]],
    pred_probs: npt.NDArray["np.floating[T]"],
) -> np.ndarray:
    
    assert_valid_inputs(
        X=None, y=labels, pred_probs=pred_probs, multi_label=True, allow_one_class=True
    )
    num_classes = get_num_classes(labels=labels, pred_probs=pred_probs, multi_label=True)
    binary_labels = int2onehot(labels, K=num_classes)
    return binary_labels

def _create_multilabel_scorer(
    method: str,
    adjust_pred_probs: bool,
    aggregator_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[MultilabelScorer, Dict]:
    
    base_scorer = ClassLabelScorer.from_str(method)
    base_scorer_kwargs = {"adjust_pred_probs": adjust_pred_probs}
    if aggregator_kwargs:
        aggregator = Aggregator(**aggregator_kwargs)
        scorer = MultilabelScorer(base_scorer, aggregator)
    else:
        scorer = MultilabelScorer(base_scorer)
    return scorer, base_scorer_kwargs

def get_label_quality_scores(
    labels: List[List[int]],
    pred_probs: npt.NDArray["np.floating[T]"],
    *,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
    aggregator_kwargs: Dict[str, Any] = {"method": "exponential_moving_average", "alpha": 0.8},
) -> npt.NDArray["np.floating[T]"]:
    
    binary_labels = _labels_to_binary(labels, pred_probs)
    scorer, base_scorer_kwargs = _create_multilabel_scorer(
        method=method,
        adjust_pred_probs=adjust_pred_probs,
        aggregator_kwargs=aggregator_kwargs,
    )
    return scorer(binary_labels, pred_probs, base_scorer_kwargs=base_scorer_kwargs)

def get_label_quality_scores_per_class(
    labels: List[List[int]],
    pred_probs: npt.NDArray["np.floating[T]"],
    *,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
) -> np.ndarray:
    
    binary_labels = _labels_to_binary(labels, pred_probs)
    scorer, base_scorer_kwargs = _create_multilabel_scorer(
        method=method,
        adjust_pred_probs=adjust_pred_probs,
    )
    return scorer.get_class_label_quality_scores(
        labels=binary_labels, pred_probs=pred_probs, base_scorer_kwargs=base_scorer_kwargs
    )
