import numpy as np
from typing import List, Tuple
import warnings

from premortemml.filter import find_label_issues as find_label_issues_main
from premortemml.experimental.label_issues_batched import find_label_issues_batched

def find_label_issues(
    labels: list,
    pred_probs: list,
    *,
    return_indices_ranked_by: str = "self_confidence",
    low_memory: bool = False,
    **kwargs,
) -> List[Tuple[int, int]]:
    
    labels_flatten = [l for label in labels for l in label]
    pred_probs_flatten = np.array([pred for pred_prob in pred_probs for pred in pred_prob])

    if low_memory:
        for arg_name, _ in kwargs.items():
            warnings.warn(f"`{arg_name}` is not used when `low_memory=True`.")
        quality_score_kwargs = {"method": return_indices_ranked_by}
        issues_main = find_label_issues_batched(
            labels_flatten, pred_probs_flatten, quality_score_kwargs=quality_score_kwargs
        )
    else:
        issues_main = find_label_issues_main(
            labels_flatten,
            pred_probs_flatten,
            return_indices_ranked_by=return_indices_ranked_by,
            **kwargs,
        )

    lengths = [len(label) for label in labels]
    mapping = [[(i, j) for j in range(length)] for i, length in enumerate(lengths)]
    mapping_flatten = [index for indicies in mapping for index in indicies]

    issues = [mapping_flatten[issue] for issue in issues_main]
    return issues
