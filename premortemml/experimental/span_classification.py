import numpy as np
from typing import List, Tuple, Optional

from premortemml.token_classification.filter import find_label_issues as find_label_issues_token
from premortemml.token_classification.summary import display_issues as display_issues_token
from premortemml.token_classification.rank import (
    get_label_quality_scores as get_label_quality_scores_token,
)

def find_label_issues(
    labels: list,
    pred_probs: list,
):
    
    pred_probs_token = _get_pred_prob_token(pred_probs)
    return find_label_issues_token(labels, pred_probs_token)

def display_issues(
    issues: list,
    tokens: List[List[str]],
    *,
    labels: Optional[list] = None,
    pred_probs: Optional[list] = None,
    exclude: List[Tuple[int, int]] = [],
    class_names: Optional[List[str]] = None,
    top: int = 20,
) -> None:
    
    display_issues_token(
        issues,
        tokens,
        labels=labels,
        pred_probs=pred_probs,
        exclude=exclude,
        class_names=class_names,
        top=top,
    )

def get_label_quality_scores(
    labels: list,
    pred_probs: list,
    **kwargs,
) -> Tuple[np.ndarray, list]:
    
    pred_probs_token = _get_pred_prob_token(pred_probs)
    return get_label_quality_scores_token(labels, pred_probs_token, **kwargs)

def _get_pred_prob_token(pred_probs: list) -> list:
    pred_probs_token = []
    for probs in pred_probs:
        pred_probs_token.append(np.stack([1 - probs, probs], axis=1))
    return pred_probs_token
