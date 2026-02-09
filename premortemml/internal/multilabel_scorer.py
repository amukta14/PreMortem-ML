from enum import Enum
from typing import Callable, Dict, Optional, Union

import numpy as np
from sklearn.model_selection import cross_val_predict

from premortemml.internal.label_quality_utils import _subtract_confident_thresholds
from premortemml.internal.multilabel_utils import _is_multilabel, stack_complement
from premortemml.internal.numerics import softmax
from premortemml.rank import (
    get_confidence_weighted_entropy_for_each_label,
    get_normalized_margin_for_each_label,
    get_self_confidence_for_each_label,
)

class _Wrapper:

    def __init__(self, f: Callable) -> None:
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __repr__(self):
        return self.f.__name__

class ClassLabelScorer(Enum):

    SELF_CONFIDENCE = _Wrapper(get_self_confidence_for_each_label)
    
    NORMALIZED_MARGIN = _Wrapper(get_normalized_margin_for_each_label)
    
    CONFIDENCE_WEIGHTED_ENTROPY = _Wrapper(get_confidence_weighted_entropy_for_each_label)
    

    def __call__(self, labels: np.ndarray, pred_probs: np.ndarray, **kwargs) -> np.ndarray:
        pred_probs = self._adjust_pred_probs(labels, pred_probs, **kwargs)
        return self.value(labels, pred_probs)

    def _adjust_pred_probs(
        self, labels: np.ndarray, pred_probs: np.ndarray, **kwargs
    ) -> np.ndarray:
        
        if kwargs.get("adjust_pred_probs", False) is True:
            if self == ClassLabelScorer.CONFIDENCE_WEIGHTED_ENTROPY:
                raise ValueError(f"adjust_pred_probs is not currently supported for {self}.")
            pred_probs = _subtract_confident_thresholds(labels, pred_probs)
        return pred_probs

    @classmethod
    def from_str(cls, method: str) -> "ClassLabelScorer":
        try:
            return cls[method.upper()]
        except KeyError:
            raise ValueError(f"Invalid method name: {method}")

def exponential_moving_average(
    s: np.ndarray,
    *,
    alpha: Optional[float] = None,
    axis: int = 1,
    **_,
) -> np.ndarray:
    r
    K = s.shape[1]
    s_sorted = np.fliplr(np.sort(s, axis=axis))
    if alpha is None:
        # One conventional choice for alpha is 2/(K + 1), where K is the number of periods in the moving average.
        alpha = float(2 / (K + 1))
    if not (0 <= alpha <= 1):
        raise ValueError(f"alpha must be in the interval [0, 1], got {alpha}")
    s_T = s_sorted.T
    s_ema, s_next = s_T[0], s_T[1:]
    for s_i in s_next:
        s_ema = alpha * s_i + (1 - alpha) * s_ema
    return s_ema

def softmin(
    s: np.ndarray,
    *,
    temperature: float = 0.1,
    axis: int = 1,
    **_,
) -> np.ndarray:
    

    return np.einsum(
        "ij,ij->i", s, softmax(x=1 - s, temperature=temperature, axis=axis, shift=True)
    )

class Aggregator:

    possible_methods: Dict[str, Callable[..., np.ndarray]] = {
        "exponential_moving_average": exponential_moving_average,
        "softmin": softmin,
    }

    def __init__(self, method: Union[str, Callable], **kwargs):
        if isinstance(method, str):  # convert to callable
            if method in self.possible_methods:
                method = self.possible_methods[method]
            else:
                raise ValueError(
                    f"Invalid aggregation method specified: '{method}', must be one of the following: {list(self.possible_methods.keys())}"
                )

        self._validate_method(method)
        self.method = method
        self.kwargs = kwargs

    @staticmethod
    def _validate_method(method) -> None:
        if not callable(method):
            raise TypeError(f"Expected callable method, got {type(method)}")

    @staticmethod
    def _validate_scores(scores: np.ndarray) -> None:
        if not (isinstance(scores, np.ndarray) and scores.ndim == 2):
            raise ValueError(
                f"Expected 2D array for scores, got {type(scores)} with shape {scores.shape}"
            )

    def __call__(self, scores: np.ndarray, **kwargs) -> np.ndarray:
        self._validate_scores(scores)
        kwargs["axis"] = 1
        updated_kwargs = {**self.kwargs, **kwargs}
        return self.method(scores, **updated_kwargs)

    def __repr__(self):
        return f"Aggregator(method={self.method.__name__}, kwargs={self.kwargs})"

class MultilabelScorer:

    def __init__(
        self,
        base_scorer: ClassLabelScorer = ClassLabelScorer.SELF_CONFIDENCE,
        aggregator: Union[Aggregator, Callable] = Aggregator(exponential_moving_average, alpha=0.8),
        *,
        strict: bool = True,
    ):
        self.base_scorer = base_scorer
        if not isinstance(aggregator, Aggregator):
            self.aggregator = Aggregator(aggregator)
        else:
            self.aggregator = aggregator
        self.strict = strict

    def __call__(
        self,
        labels: np.ndarray,
        pred_probs: np.ndarray,
        base_scorer_kwargs: Optional[dict] = None,
        **aggregator_kwargs,
    ) -> np.ndarray:
        
        if self.strict:
            self._validate_labels_and_pred_probs(labels, pred_probs)
        scores = self.get_class_label_quality_scores(labels, pred_probs, base_scorer_kwargs)
        return self.aggregate(scores, **aggregator_kwargs)

    def aggregate(
        self,
        class_label_quality_scores: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        
        return self.aggregator(class_label_quality_scores, **kwargs)

    def get_class_label_quality_scores(
        self,
        labels: np.ndarray,
        pred_probs: np.ndarray,
        base_scorer_kwargs: Optional[dict] = None,
    ) -> np.ndarray:
        
        class_label_quality_scores = np.zeros(shape=labels.shape)
        if base_scorer_kwargs is None:
            base_scorer_kwargs = {}
        for i, (label_i, pred_prob_i) in enumerate(zip(labels.T, pred_probs.T)):
            pred_prob_i_two_columns = stack_complement(pred_prob_i)
            class_label_quality_scores[:, i] = self.base_scorer(
                label_i, pred_prob_i_two_columns, **base_scorer_kwargs
            )
        return class_label_quality_scores

    @staticmethod
    def _validate_labels_and_pred_probs(labels: np.ndarray, pred_probs: np.ndarray) -> None:
        # Only allow dense matrices for labels for now
        if not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a numpy array.")
        if not _is_multilabel(labels):
            raise ValueError("Labels must be in multi-label format.")
        if labels.shape != pred_probs.shape:
            raise ValueError("Labels and predicted probabilities must have the same shape.")

def get_label_quality_scores(
    labels,
    pred_probs,
    *,
    method: MultilabelScorer = MultilabelScorer(),
    base_scorer_kwargs: Optional[dict] = None,
    **aggregator_kwargs,
) -> np.ndarray:
    
    return method(labels, pred_probs, base_scorer_kwargs=base_scorer_kwargs, **aggregator_kwargs)

# Probabilities

def multilabel_py(y: np.ndarray) -> np.ndarray:

    N, _ = y.shape
    fraction_0 = np.sum(y == 0, axis=0) / N
    fraction_1 = 1 - fraction_0
    py = np.column_stack((fraction_0, fraction_1))
    return py

# Cross-validation helpers

def _get_split_generator(labels, cv):
    _, multilabel_ids = np.unique(labels, axis=0, return_inverse=True)
    split_generator = cv.split(X=multilabel_ids, y=multilabel_ids)
    return split_generator

def get_cross_validated_multilabel_pred_probs(X, labels: np.ndarray, *, clf, cv) -> np.ndarray:
    split_generator = _get_split_generator(labels, cv)
    pred_probs = cross_val_predict(clf, X, labels, cv=split_generator, method="predict_proba")
    return pred_probs
