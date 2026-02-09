import numpy as np
from sklearn.metrics import log_loss
from typing import List, Optional
import warnings

from premortemml.internal.validation import assert_valid_inputs
from premortemml.internal.constants import (
    CLIPPING_LOWER_BOUND,
)  # lower-bound clipping threshold to prevents 0 in logs and division

from premortemml.internal.label_quality_utils import (
    _subtract_confident_thresholds,
    get_normalized_entropy,
)

def get_label_quality_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
) -> np.ndarray:
    

    assert_valid_inputs(
        X=None, y=labels, pred_probs=pred_probs, multi_label=False, allow_one_class=True
    )
    return _compute_label_quality_scores(
        labels=labels, pred_probs=pred_probs, method=method, adjust_pred_probs=adjust_pred_probs
    )

def _compute_label_quality_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
    confident_thresholds: Optional[np.ndarray] = None,
) -> np.ndarray:
    
    scoring_funcs = {
        "self_confidence": get_self_confidence_for_each_label,
        "normalized_margin": get_normalized_margin_for_each_label,
        "confidence_weighted_entropy": get_confidence_weighted_entropy_for_each_label,
    }
    try:
        scoring_func = scoring_funcs[method]
    except KeyError:
        raise ValueError(
            f
        )
    if adjust_pred_probs:
        if method == "confidence_weighted_entropy":
            raise ValueError(f"adjust_pred_probs is not currently supported for {method}.")
        pred_probs = _subtract_confident_thresholds(
            labels=labels, pred_probs=pred_probs, confident_thresholds=confident_thresholds
        )

    scoring_inputs = {"labels": labels, "pred_probs": pred_probs}
    label_quality_scores = scoring_func(**scoring_inputs)
    return label_quality_scores

def get_label_quality_ensemble_scores(
    labels: np.ndarray,
    pred_probs_list: List[np.ndarray],
    *,
    method: str = "self_confidence",
    adjust_pred_probs: bool = False,
    weight_ensemble_members_by: str = "accuracy",
    custom_weights: Optional[np.ndarray] = None,
    log_loss_search_T_values: List[float] = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 2e2],
    verbose: bool = True,
) -> np.ndarray:
    

    # Check pred_probs_list for errors
    assert isinstance(
        pred_probs_list, list
    ), f"pred_probs_list needs to be a list. Provided pred_probs_list is a {type(pred_probs_list)}"

    assert len(pred_probs_list) > 0, "pred_probs_list is empty."

    if len(pred_probs_list) == 1:
        warnings.warn(
            
        )

    for pred_probs in pred_probs_list:
        assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs, multi_label=False)

    # Raise ValueError if user passed custom_weights array but did not choose weight_ensemble_members_by="custom"
    if custom_weights is not None and weight_ensemble_members_by != "custom":
        raise ValueError(
            f
        )

    # This weighting scheme performs search of t in log_loss_search_T_values for "best" log loss
    if weight_ensemble_members_by == "log_loss_search":
        # Initialize variables for log loss search
        pred_probs_avg_log_loss_weighted = None
        neg_log_loss_weights = None
        best_eval_log_loss = float("inf")

        for t in log_loss_search_T_values:
            neg_log_loss_list = []

            # pred_probs for each model
            for pred_probs in pred_probs_list:
                pred_probs_clipped = np.clip(
                    pred_probs, a_min=CLIPPING_LOWER_BOUND, a_max=None
                )  # lower-bound clipping threshold to prevents 0 in logs when calculating log loss
                pred_probs_clipped /= pred_probs_clipped.sum(axis=1)[:, np.newaxis]  # renormalize

                neg_log_loss = np.exp(-t * log_loss(labels, pred_probs_clipped))
                neg_log_loss_list.append(neg_log_loss)

            # weights using negative log loss
            neg_log_loss_weights_temp = np.array(neg_log_loss_list) / sum(neg_log_loss_list)

            # weighted average using negative log loss
            pred_probs_avg_log_loss_weighted_temp = sum(
                [neg_log_loss_weights_temp[i] * p for i, p in enumerate(pred_probs_list)]
            )
            # evaluate log loss with this weighted average pred_probs
            eval_log_loss = log_loss(labels, pred_probs_avg_log_loss_weighted_temp)

            # check if eval_log_loss is the best so far (lower the better)
            if best_eval_log_loss > eval_log_loss:
                best_eval_log_loss = eval_log_loss
                pred_probs_avg_log_loss_weighted = pred_probs_avg_log_loss_weighted_temp
                neg_log_loss_weights = neg_log_loss_weights_temp.copy()

    # Generate scores for each model's pred_probs
    scores_list = []
    accuracy_list = []
    for pred_probs in pred_probs_list:
        # Calculate scores and accuracy
        scores = get_label_quality_scores(
            labels=labels,
            pred_probs=pred_probs,
            method=method,
            adjust_pred_probs=adjust_pred_probs,
        )
        scores_list.append(scores)

        # Only compute if weighting by accuracy
        if weight_ensemble_members_by == "accuracy":
            accuracy = (pred_probs.argmax(axis=1) == labels).mean()
            accuracy_list.append(accuracy)

    if verbose:
        print(f"Weighting scheme for ensemble: {weight_ensemble_members_by}")

    # Transform list of scores into an array of shape (N, M) where M is the number of models in the ensemble
    scores_ensemble = np.vstack(scores_list).T

    # Aggregate scores with chosen weighting scheme
    if weight_ensemble_members_by == "uniform":
        label_quality_scores = scores_ensemble.mean(axis=1)  # Uniform weights (simple average)

    elif weight_ensemble_members_by == "accuracy":
        weights = np.array(accuracy_list) / sum(accuracy_list)  # Weight by relative accuracy
        if verbose:
            print("Ensemble members will be weighted by their relative accuracy")
            for i, acc in enumerate(accuracy_list):
                print(f"  Model {i} accuracy : {acc}")
                print(f"  Model {i} weight   : {weights[i]}")

        # Aggregate scores with weighted average
        label_quality_scores = (scores_ensemble * weights).sum(axis=1)

    elif weight_ensemble_members_by == "log_loss_search":
        assert neg_log_loss_weights is not None
        weights = neg_log_loss_weights  # Weight by exp(t * -log_loss) where t is found by searching through log_loss_search_T_values
        if verbose:
            print(
                "Ensemble members will be weighted by log-loss between their predicted probabilities and given labels"
            )
            for i, weight in enumerate(weights):
                print(f"  Model {i} weight   : {weight}")

        # Aggregate scores with weighted average
        label_quality_scores = (scores_ensemble * weights).sum(axis=1)

    elif weight_ensemble_members_by == "custom":
        # Check custom_weights for errors
        assert (
            custom_weights is not None
        ), "custom_weights is None! Please pass a valid custom_weights."

        assert len(custom_weights) == len(
            pred_probs_list
        ), "Length of custom_weights array must match the number of models: len(pred_probs_list)."

        # Aggregate scores with custom weights
        label_quality_scores = (scores_ensemble * custom_weights).sum(axis=1)

    else:
        raise ValueError(
            f
        )

    return label_quality_scores

def find_top_issues(quality_scores: np.ndarray, *, top: int = 10) -> np.ndarray:

    if top is None or top > len(quality_scores):
        top = len(quality_scores)

    top_outlier_indices = quality_scores.argsort()[:top]
    return top_outlier_indices

def order_label_issues(
    label_issues_mask: np.ndarray,
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    rank_by: str = "self_confidence",
    rank_by_kwargs: dict = {},
) -> np.ndarray:
    

    allow_one_class = False
    if isinstance(labels, np.ndarray) or all(isinstance(lab, int) for lab in labels):
        if set(labels) == {0}:  # occurs with missing classes in multi-label settings
            allow_one_class = True
    assert_valid_inputs(
        X=None,
        y=labels,
        pred_probs=pred_probs,
        multi_label=False,
        allow_one_class=allow_one_class,
    )

    # Convert bool mask to index mask
    label_issues_idx = np.arange(len(labels))[label_issues_mask]

    # Calculate label quality scores
    label_quality_scores = get_label_quality_scores(
        labels, pred_probs, method=rank_by, **rank_by_kwargs
    )

    # Get label quality scores for label issues
    label_quality_scores_issues = label_quality_scores[label_issues_mask]

    return label_issues_idx[np.argsort(label_quality_scores_issues)]

def get_self_confidence_for_each_label(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> np.ndarray:
    

    # To make this work for multi-label (but it will slow down runtime), return:
    # np.array([np.mean(pred_probs[i, l]) for i, l in enumerate(labels)])
    return pred_probs[np.arange(labels.shape[0]), labels]

def get_normalized_margin_for_each_label(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> np.ndarray:
    

    self_confidence = get_self_confidence_for_each_label(labels, pred_probs)
    N, K = pred_probs.shape
    del_indices = np.arange(N) * K + labels
    max_prob_not_label = np.max(
        np.delete(pred_probs, del_indices, axis=None).reshape(N, K - 1), axis=-1
    )
    label_quality_scores = (self_confidence - max_prob_not_label + 1) / 2
    return label_quality_scores

def get_confidence_weighted_entropy_for_each_label(
    labels: np.ndarray, pred_probs: np.ndarray
) -> np.ndarray:
    

    self_confidence = get_self_confidence_for_each_label(labels, pred_probs)
    self_confidence = np.clip(self_confidence, a_min=CLIPPING_LOWER_BOUND, a_max=None)

    # Divide entropy by self confidence
    label_quality_scores = get_normalized_entropy(pred_probs) / self_confidence

    # Rescale
    clipped_scores = np.clip(label_quality_scores, a_min=CLIPPING_LOWER_BOUND, a_max=None)
    label_quality_scores = np.log(label_quality_scores + 1) / clipped_scores

    return label_quality_scores
