import warnings
import numpy as np
from typing import Tuple

from premortemml.internal.util import value_counts, clip_values, clip_noise_rates
from premortemml.internal.constants import TINY_VALUE, CLIPPING_LOWER_BOUND

def compute_ps_py_inv_noise_matrix(
    labels, noise_matrix
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    

    ps = value_counts(labels) / float(len(labels))  # p(labels=k)
    py, inverse_noise_matrix = compute_py_inv_noise_matrix(ps, noise_matrix)
    return ps, py, inverse_noise_matrix

def compute_py_inv_noise_matrix(ps, noise_matrix) -> Tuple[np.ndarray, np.ndarray]:

    # 'py' is p(true_labels=k) = noise_matrix^(-1) * p(labels=k)
    # because in *vector computation*: P(label=k|true_label=k) * p(true_label=k) = P(label=k)
    # The pseudo-inverse is used when noise_matrix is not invertible.
    py = np.linalg.inv(noise_matrix).dot(ps)

    # No class should have probability 0, so we use .000001
    # Make sure valid probabilities that sum to 1.0
    py = clip_values(py, low=CLIPPING_LOWER_BOUND, high=1.0, new_sum=1.0)

    # All the work is done in this function (below)
    return py, compute_inv_noise_matrix(py=py, noise_matrix=noise_matrix, ps=ps)

def compute_inv_noise_matrix(py, noise_matrix, *, ps=None) -> np.ndarray:

    joint = noise_matrix * py
    ps = joint.sum(axis=1) if ps is None else ps
    inverse_noise_matrix = joint.T / np.clip(ps, a_min=TINY_VALUE, a_max=None)

    # Clip inverse noise rates P(true_label=k_s|true_label=k_y) into proper range [0,1)
    return clip_noise_rates(inverse_noise_matrix)

def compute_noise_matrix_from_inverse(ps, inverse_noise_matrix, *, py=None) -> np.ndarray:

    joint = (inverse_noise_matrix * ps).T
    py = joint.sum(axis=0) if py is None else py
    noise_matrix = joint / np.clip(py, a_min=TINY_VALUE, a_max=None)

    # Clip inverse noise rates P(true_label=k_y|true_label=k_s) into proper range [0,1)
    return clip_noise_rates(noise_matrix)

def compute_py(
    ps, noise_matrix, inverse_noise_matrix, *, py_method="cnt", true_labels_class_counts=None
) -> np.ndarray:
    

    if len(np.shape(ps)) > 2 or (len(np.shape(ps)) == 2 and np.shape(ps)[0] != 1):
        w = "Input parameter np.ndarray ps has shape " + str(np.shape(ps))
        w += ", but shape should be (K, ) or (1, K)"
        warnings.warn(w)

    if py_method == "marginal" and true_labels_class_counts is None:
        msg = (
            'py_method == "marginal" requires true_labels_class_counts, '
            "but true_labels_class_counts is None. "
        )
        msg += " Provide parameter true_labels_class_counts."
        raise ValueError(msg)

    if py_method == "cnt":
        # Computing py this way avoids dividing by zero noise rates.
        # More robust bc error est_p(true_label|labels) / est_p(labels|y) ~ p(true_label|labels) / p(labels|y)
        py = (
            inverse_noise_matrix.diagonal()
            / np.clip(noise_matrix.diagonal(), a_min=TINY_VALUE, a_max=None)
            * ps
        )
        # Equivalently: py = (true_labels_class_counts / labels_class_counts) * ps
    elif py_method == "eqn":
        py = np.linalg.inv(noise_matrix).dot(ps)
    elif py_method == "marginal":
        py = true_labels_class_counts / np.clip(
            float(sum(true_labels_class_counts)), a_min=TINY_VALUE, a_max=None
        )
    elif py_method == "marginal_ps":
        py = np.dot(inverse_noise_matrix, ps)
    else:
        err = "py_method {}".format(py_method)
        err += " should be in [cnt, eqn, marginal, marginal_ps]"
        raise ValueError(err)

    # Clip py (0,1), s.t. no class should have prob 0, hence 1e-6
    py = clip_values(py, low=CLIPPING_LOWER_BOUND, high=1.0, new_sum=1.0)
    return py

def compute_pyx(pred_probs, noise_matrix, inverse_noise_matrix):

    if len(np.shape(pred_probs)) != 2:
        raise ValueError(
            "Input parameter np.ndarray 'pred_probs' has shape "
            + str(np.shape(pred_probs))
            + ", but shape should be (N, K)"
        )

    pyx = (
        pred_probs
        * inverse_noise_matrix.diagonal()
        / np.clip(noise_matrix.diagonal(), a_min=TINY_VALUE, a_max=None)
    )
    # Make sure valid probabilities that sum to 1.0
    return np.apply_along_axis(
        func1d=clip_values, axis=1, arr=pyx, **{"low": 0.0, "high": 1.0, "new_sum": 1.0}
    )
