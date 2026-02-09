from typing import List, Optional, Tuple

import numpy as np

from premortemml.internal.util import get_num_classes

def _is_multilabel(y: np.ndarray) -> bool:
    if not (isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] > 1):
        return False
    return np.array_equal(np.unique(y), [0, 1])

def stack_complement(pred_prob_slice: np.ndarray) -> np.ndarray:
    return np.vstack((1 - pred_prob_slice, pred_prob_slice)).T

def get_onehot_num_classes(
    labels: list, pred_probs: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, int]:
    
    num_classes = get_num_classes(labels=labels, pred_probs=pred_probs)
    try:
        y_one = int2onehot(labels, K=num_classes)
    except TypeError:
        raise ValueError(
            "wrong format for labels, should be a list of list[indices], please check the documentation in find_label_issues for further information"
        )
    return y_one, num_classes

def int2onehot(labels: list, K: int) -> np.ndarray:

    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer(classes=range(K))
    return mlb.fit_transform(labels)

def onehot2int(onehot_matrix: np.ndarray) -> List[List[int]]:

    return [np.where(row)[0].tolist() for row in onehot_matrix]
