import numpy as np
from sklearn.metrics import confusion_matrix
import multiprocessing
import sys
import warnings
from typing import Any, Dict, Optional, Tuple, List
from functools import reduce
import platform

from premortemml.count import calibrate_confident_joint, num_label_issues, _reduce_issues
from premortemml.rank import order_label_issues, get_label_quality_scores
from premortemml.internal.validation import assert_valid_inputs
from premortemml.internal.util import (
    value_counts_fill_missing_classes,
    round_preserving_row_totals,
    get_num_classes,
)
from premortemml.internal.multilabel_utils import stack_complement, get_onehot_num_classes, int2onehot
from premortemml.typing import LabelLike
from premortemml.multilabel_classification.filter import find_multilabel_issues_per_class

# tqdm is a package to print time-to-complete when multiprocessing is used.
# This package is not necessary, but when installed improves user experience for large datasets.
try:
    import tqdm.auto as tqdm

    tqdm_exists = True
except ImportError as e:  # pragma: no cover
    tqdm_exists = False

    warnings.warn(w)

# psutil is a package used to count physical cores for multiprocessing
# This package is not necessary, because we can always fall back to logical cores as the default
try:
    import psutil

    psutil_exists = True
except ImportError as e:  # pragma: no cover
    psutil_exists = False

# global variable for find_label_issues multiprocessing
pred_probs_by_class: Dict[int, np.ndarray]
prune_count_matrix_cols: Dict[int, np.ndarray]

def find_label_issues(
    labels: LabelLike,
    pred_probs: np.ndarray,
    *,
    return_indices_ranked_by: Optional[str] = None,
    rank_by_kwargs: Optional[Dict[str, Any]] = None,
    filter_by: str = "prune_by_noise_rate",
    frac_noise: float = 1.0,
    num_to_remove_per_class: Optional[List[int]] = None,
    min_examples_per_class=1,
    confident_joint: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    multi_label: bool = False,
) -> np.ndarray:
    
    if not rank_by_kwargs:
        rank_by_kwargs = {}

    assert filter_by in [
        "low_normalized_margin",
        "low_self_confidence",
        "prune_by_noise_rate",
        "prune_by_class",
        "both",
        "confident_learning",
        "predicted_neq_given",
    ]
    allow_one_class = False
    if isinstance(labels, np.ndarray) or all(isinstance(lab, int) for lab in labels):
        if set(labels) == {0}:  # occurs with missing classes in multi-label settings
            allow_one_class = True
    assert_valid_inputs(
        X=None,
        y=labels,
        pred_probs=pred_probs,
        multi_label=multi_label,
        allow_one_class=allow_one_class,
    )

    if filter_by in [
        "confident_learning",
        "predicted_neq_given",
        "low_normalized_margin",
        "low_self_confidence",
    ] and (frac_noise != 1.0 or num_to_remove_per_class is not None):
        warn_str = (
            "frac_noise and num_to_remove_per_class parameters are only supported"
            " for filter_by 'prune_by_noise_rate', 'prune_by_class', and 'both'. They "
            "are not supported for methods 'confident_learning', 'predicted_neq_given', "
            "'low_normalized_margin' or 'low_self_confidence'."
        )
        warnings.warn(warn_str)
    if (num_to_remove_per_class is not None) and (
        filter_by
        in [
            "confident_learning",
            "predicted_neq_given",
            "low_normalized_margin",
            "low_self_confidence",
        ]
    ):
        raise ValueError(
            "filter_by 'confident_learning', 'predicted_neq_given', 'low_normalized_margin' "
            "or 'low_self_confidence' is not supported (yet) when setting 'num_to_remove_per_class'"
        )
    if filter_by == "confident_learning" and isinstance(confident_joint, np.ndarray):
        warn_str = (
            "The supplied `confident_joint` is ignored when `filter_by = 'confident_learning'`; confident joint will be "
            "re-estimated from the given labels. To use your supplied `confident_joint`, please specify a different "
            "`filter_by` value."
        )
        warnings.warn(warn_str)

    K = get_num_classes(
        labels=labels, pred_probs=pred_probs, label_matrix=confident_joint, multi_label=multi_label
    )
    # Boolean set to true if dataset is large
    big_dataset = K * len(labels) > 1e8

    # Set-up number of multiprocessing threads
    # On Windows/macOS, when multi_label is True, multiprocessing is much slower
    # even for faily large input arrays, so we default to n_jobs=1 in this case
    os_name = platform.system()
    if n_jobs is None:
        if multi_label and os_name != "Linux":
            n_jobs = 1
        else:
            if psutil_exists:
                n_jobs = psutil.cpu_count(logical=False)  # physical cores
            elif big_dataset:
                print(
                    "To default `n_jobs` to the number of physical cores for multiprocessing in find_label_issues(), please: `pip install psutil`.\n"
                    "Note: You can safely ignore this message. `n_jobs` only affects runtimes, results will be the same no matter its value.\n"
                    "Since psutil is not installed, `n_jobs` was set to the number of logical cores by default.\n"
                    "Disable this message by either installing psutil or specifying the `n_jobs` argument."
                )  # pragma: no cover
            if not n_jobs:
                # either psutil does not exist
                # or psutil can return None when physical cores cannot be determined
                # switch to logical cores
                n_jobs = multiprocessing.cpu_count()
    else:
        assert n_jobs >= 1

    if multi_label:
        if not isinstance(labels, list):
            raise TypeError("`labels` must be list when `multi_label=True`.")
        warnings.warn(
            "The multi_label argument to filter.find_label_issues() is deprecated and will be removed in future versions. Please use `multilabel_classification.filter.find_label_issues()` instead.",
            DeprecationWarning,
        )
        return _find_label_issues_multilabel(
            labels,
            pred_probs,
            return_indices_ranked_by,
            rank_by_kwargs,
            filter_by,
            frac_noise,
            num_to_remove_per_class,
            min_examples_per_class,
            confident_joint,
            n_jobs,
            verbose,
        )

    # Else this is standard multi-class classification
    # Number of examples in each class of labels
    label_counts = value_counts_fill_missing_classes(labels, K, multi_label=multi_label)
    # Ensure labels are of type np.ndarray()
    labels = np.asarray(labels)
    if confident_joint is None or filter_by == "confident_learning":
        from premortemml.count import compute_confident_joint

        confident_joint, cl_error_indices = compute_confident_joint(
            labels=labels,
            pred_probs=pred_probs,
            multi_label=multi_label,
            return_indices_of_off_diagonals=True,
        )

    if filter_by in ["low_normalized_margin", "low_self_confidence"]:
        scores = get_label_quality_scores(
            labels,
            pred_probs,
            method=filter_by[4:],
            adjust_pred_probs=False,
        )
        num_errors = num_label_issues(
            labels, pred_probs, multi_label=multi_label
        )
        # Find label issues O(nlogn) solution (mapped to boolean mask later in the method)
        cl_error_indices = np.argsort(scores)[:num_errors]
        # The following is the O(n) fastest solution (check for one-off errors), but the problem is if lots of the scores are identical you will overcount,
        # you can end up returning more or less and they aren't ranked in the boolean form so there's no way to drop the highest scores randomly

    if filter_by in ["prune_by_noise_rate", "prune_by_class", "both"]:
        # Create `prune_count_matrix` with the number of examples to remove in each class and
        # leave at least min_examples_per_class examples per class.
        # `prune_count_matrix` is transposed relative to the confident_joint.
        prune_count_matrix = _keep_at_least_n_per_class(
            prune_count_matrix=confident_joint.T,
            n=min_examples_per_class,
            frac_noise=frac_noise,
        )

        if num_to_remove_per_class is not None:
            # Estimate joint probability distribution over label issues
            psy = prune_count_matrix / np.sum(prune_count_matrix, axis=1)
            noise_per_s = psy.sum(axis=1) - psy.diagonal()
            # Calibrate labels.t. noise rates sum to num_to_remove_per_class
            tmp = (psy.T * num_to_remove_per_class / noise_per_s).T
            np.fill_diagonal(tmp, label_counts - num_to_remove_per_class)
            prune_count_matrix = round_preserving_row_totals(tmp)

        # Prepare multiprocessing shared data
        # On Linux with Python <3.14, multiprocessing is started with fork,
        # so data can be shared with global variables + COW
        # On Window/macOS, processes are started with spawn,
        # so data will need to be pickled to the subprocesses through input args
        # In Python 3.14+, global variable sharing is no longer reliable even on Linux
        chunksize = max(1, K // n_jobs)
        use_global_vars = n_jobs == 1 or (os_name == "Linux" and sys.version_info < (3, 14))
        if use_global_vars:
            global pred_probs_by_class, prune_count_matrix_cols
            pred_probs_by_class = {k: pred_probs[labels == k] for k in range(K)}
            prune_count_matrix_cols = {k: prune_count_matrix[:, k] for k in range(K)}
            args = [[k, min_examples_per_class, None] for k in range(K)]
        else:
            args = [
                [k, min_examples_per_class, [pred_probs[labels == k], prune_count_matrix[:, k]]]
                for k in range(K)
            ]

    # Perform Pruning with threshold probabilities from BFPRT algorithm in O(n)
    # Operations are parallelized across all CPU processes
    if filter_by == "prune_by_class" or filter_by == "both":
        if n_jobs > 1:
            with multiprocessing.Pool(n_jobs) as p:
                if verbose:  # pragma: no cover
                    print("Parallel processing label issues by class.")
                sys.stdout.flush()
                if big_dataset and tqdm_exists:
                    label_issues_masks_per_class = list(
                        tqdm.tqdm(p.imap(_prune_by_class, args, chunksize=chunksize), total=K)
                    )
                else:
                    label_issues_masks_per_class = p.map(_prune_by_class, args, chunksize=chunksize)
        else:
            label_issues_masks_per_class = [_prune_by_class(arg) for arg in args]

        label_issues_mask = np.zeros(len(labels), dtype=bool)
        for k, mask in enumerate(label_issues_masks_per_class):
            if len(mask) > 1:
                label_issues_mask[labels == k] = mask

    if filter_by == "both":
        label_issues_mask_by_class = label_issues_mask

    if filter_by == "prune_by_noise_rate" or filter_by == "both":
        if n_jobs > 1:
            with multiprocessing.Pool(n_jobs) as p:
                if verbose:  # pragma: no cover
                    print("Parallel processing label issues by noise rate.")
                sys.stdout.flush()
                if big_dataset and tqdm_exists:
                    label_issues_masks_per_class = list(
                        tqdm.tqdm(p.imap(_prune_by_count, args, chunksize=chunksize), total=K)
                    )
                else:
                    label_issues_masks_per_class = p.map(_prune_by_count, args, chunksize=chunksize)
        else:
            label_issues_masks_per_class = [_prune_by_count(arg) for arg in args]

        label_issues_mask = np.zeros(len(labels), dtype=bool)
        for k, mask in enumerate(label_issues_masks_per_class):
            if len(mask) > 1:
                label_issues_mask[labels == k] = mask

    if filter_by == "both":
        label_issues_mask = label_issues_mask & label_issues_mask_by_class

    if filter_by in ["confident_learning", "low_normalized_margin", "low_self_confidence"]:
        label_issues_mask = np.zeros(len(labels), dtype=bool)
        label_issues_mask[cl_error_indices] = True

    if filter_by == "predicted_neq_given":
        label_issues_mask = find_predicted_neq_given(labels, pred_probs, multi_label=multi_label)

    if filter_by not in ["low_self_confidence", "low_normalized_margin"]:
        # Remove label issues if model prediction is close to given label
        mask = _reduce_issues(pred_probs=pred_probs, labels=labels)
        label_issues_mask[mask] = False

    if verbose:
        print("Number of label issues found: {}".format(sum(label_issues_mask)))

    if return_indices_ranked_by is not None:
        er = order_label_issues(
            label_issues_mask=label_issues_mask,
            labels=labels,
            pred_probs=pred_probs,
            rank_by=return_indices_ranked_by,
            rank_by_kwargs=rank_by_kwargs,
        )
        return er
    return label_issues_mask

def _find_label_issues_multilabel(
    labels: list,
    pred_probs: np.ndarray,
    return_indices_ranked_by: Optional[str] = None,
    rank_by_kwargs={},
    filter_by: str = "prune_by_noise_rate",
    frac_noise: float = 1.0,
    num_to_remove_per_class: Optional[List[int]] = None,
    min_examples_per_class=1,
    confident_joint: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    low_memory: bool = False,
) -> np.ndarray:
    
    if filter_by in ["low_normalized_margin", "low_self_confidence"] and not low_memory:
        num_errors = sum(
            find_label_issues(
                labels=labels,
                pred_probs=pred_probs,
                confident_joint=confident_joint,
                multi_label=True,
                filter_by="confident_learning",
            )
        )

        y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
        label_quality_scores = ml_scorer.get_label_quality_scores(
            labels=y_one,
            pred_probs=pred_probs,
        )

        cl_error_indices = np.argsort(label_quality_scores)[:num_errors]
        label_issues_mask = np.zeros(len(labels), dtype=bool)
        label_issues_mask[cl_error_indices] = True

        if return_indices_ranked_by is not None:
            label_quality_scores_issues = ml_scorer.get_label_quality_scores(
                labels=y_one[label_issues_mask],
                pred_probs=pred_probs[label_issues_mask],
                method=ml_scorer.MultilabelScorer(
                    base_scorer=ml_scorer.ClassLabelScorer.from_str(return_indices_ranked_by),
                ),
                base_scorer_kwargs=rank_by_kwargs,
            )
            return cl_error_indices[np.argsort(label_quality_scores_issues)]

        return label_issues_mask

    per_class_issues = find_multilabel_issues_per_class(
        labels,
        pred_probs,
        return_indices_ranked_by,
        rank_by_kwargs,
        filter_by,
        frac_noise,
        num_to_remove_per_class,
        min_examples_per_class,
        confident_joint,
        n_jobs,
        verbose,
        low_memory,
    )
    if return_indices_ranked_by is None:
        assert isinstance(per_class_issues, np.ndarray)
        return per_class_issues.sum(axis=1) >= 1
    else:
        label_issues_list, labels_list, pred_probs_list = per_class_issues
        label_issues_idx = reduce(np.union1d, label_issues_list)
        y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
        label_quality_scores = ml_scorer.get_label_quality_scores(
            labels=y_one,
            pred_probs=pred_probs,
            method=ml_scorer.MultilabelScorer(
                base_scorer=ml_scorer.ClassLabelScorer.from_str(return_indices_ranked_by),
            ),
            base_scorer_kwargs=rank_by_kwargs,
        )
        label_quality_scores_issues = label_quality_scores[label_issues_idx]
        return label_issues_idx[np.argsort(label_quality_scores_issues)]

def _keep_at_least_n_per_class(
    prune_count_matrix: np.ndarray, n: int, *, frac_noise: float = 1.0
) -> np.ndarray:
    

    prune_count_matrix_diagonal = np.diagonal(prune_count_matrix)

    # Set diagonal terms less than n, to n.
    new_diagonal = np.maximum(prune_count_matrix_diagonal, n)

    # Find how much diagonal terms were increased.
    diff_per_col = new_diagonal - prune_count_matrix_diagonal

    # Count non-zero, non-diagonal items per column
    # np.maximum(*, 1) makes this never 0 (we divide by this next)
    num_noise_rates_per_col = np.maximum(
        np.count_nonzero(prune_count_matrix, axis=0) - 1.0,
        1.0,
    )

    # Uniformly decrease non-zero noise rates by the same amount
    # that the diagonal items were increased
    new_mat = prune_count_matrix - diff_per_col / num_noise_rates_per_col

    # Originally zero noise rates will now be negative, fix them back to zero
    new_mat[new_mat < 0] = 0

    # Round diagonal terms (correctly labeled examples)
    np.fill_diagonal(new_mat, new_diagonal)

    # Reduce (multiply) all noise rates (non-diagonal) by frac_noise and
    # increase diagonal by the total amount reduced in each column
    # to preserve column counts.
    new_mat = _reduce_prune_counts(new_mat, frac_noise)

    # These are counts, so return a matrix of ints.
    return round_preserving_row_totals(new_mat).astype(int)

def _reduce_prune_counts(prune_count_matrix: np.ndarray, frac_noise: float = 1.0) -> np.ndarray:

    new_mat = prune_count_matrix * frac_noise
    np.fill_diagonal(new_mat, prune_count_matrix.diagonal())
    np.fill_diagonal(
        new_mat,
        prune_count_matrix.diagonal() + np.sum(prune_count_matrix - new_mat, axis=0),
    )

    # These are counts, so return a matrix of ints.
    return new_mat.astype(int)

def find_predicted_neq_given(
    labels: LabelLike, pred_probs: np.ndarray, *, multi_label: bool = False
) -> np.ndarray:
    

    assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs, multi_label=multi_label)
    if multi_label:
        if not isinstance(labels, list):
            raise TypeError("`labels` must be list when `multi_label=True`.")
        else:
            return _find_predicted_neq_given_multilabel(labels=labels, pred_probs=pred_probs)
    else:
        return np.argmax(pred_probs, axis=1) != np.asarray(labels)

def _find_predicted_neq_given_multilabel(labels: list, pred_probs: np.ndarray) -> np.ndarray:
    y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
    pred_neq: np.ndarray = np.zeros(y_one.shape).astype(bool)
    for class_num, (label, pred_prob_for_class) in enumerate(zip(y_one.T, pred_probs.T)):
        pred_probs_binary = stack_complement(pred_prob_for_class)
        pred_neq[:, class_num] = find_predicted_neq_given(
            labels=label, pred_probs=pred_probs_binary
        )
    return pred_neq.sum(axis=1) >= 1

def find_label_issues_using_argmax_confusion_matrix(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    calibrate: bool = True,
    filter_by: str = "prune_by_noise_rate",
) -> np.ndarray:
    

    assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs, multi_label=False)
    confident_joint = confusion_matrix(np.argmax(pred_probs, axis=1), labels).T
    if calibrate:
        confident_joint = calibrate_confident_joint(confident_joint, labels)
    return find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        confident_joint=confident_joint,
        filter_by=filter_by,
    )

# Multiprocessing helper functions:

mp_params: Dict[str, Any] = {}  # Globals to be shared across threads in multiprocessing

def _to_np_array(
    mp_arr: bytearray, dtype="int32", shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:  # pragma: no cover
    
    arr = np.frombuffer(mp_arr, dtype=dtype)
    if shape is None:
        return arr
    return arr.reshape(shape)

def _init(
    __labels,
    __label_counts,
    __prune_count_matrix,
    __pcm_shape,
    __pred_probs,
    __pred_probs_shape,
    __multi_label,
    __min_examples_per_class,
):  # pragma: no cover
    

    mp_params["labels"] = __labels
    mp_params["label_counts"] = __label_counts
    mp_params["prune_count_matrix"] = __prune_count_matrix
    mp_params["pcm_shape"] = __pcm_shape
    mp_params["pred_probs"] = __pred_probs
    mp_params["pred_probs_shape"] = __pred_probs_shape
    mp_params["multi_label"] = __multi_label
    mp_params["min_examples_per_class"] = __min_examples_per_class

def _get_shared_data() -> Any:  # pragma: no cover

    label_counts = _to_np_array(mp_params["label_counts"])
    prune_count_matrix = _to_np_array(
        mp_arr=mp_params["prune_count_matrix"],
        shape=mp_params["pcm_shape"],
    )
    pred_probs = _to_np_array(
        mp_arr=mp_params["pred_probs"],
        dtype="float32",
        shape=mp_params["pred_probs_shape"],
    )
    min_examples_per_class = mp_params["min_examples_per_class"]
    multi_label = mp_params["multi_label"]
    labels = _to_np_array(mp_params["labels"])  # type: ignore
    return (
        labels,
        label_counts,
        prune_count_matrix,
        pred_probs,
        multi_label,
        min_examples_per_class,
    )

def _prune_by_class(args: list) -> np.ndarray:

    k, min_examples_per_class, arrays = args
    if arrays is None:
        pred_probs = pred_probs_by_class[k]
        prune_count_matrix = prune_count_matrix_cols[k]
    else:
        pred_probs = arrays[0]
        prune_count_matrix = arrays[1]

    label_counts = pred_probs.shape[0]
    label_issues = np.zeros(label_counts, dtype=bool)
    if label_counts > min_examples_per_class:  # No prune if not at least min_examples_per_class
        num_issues = label_counts - prune_count_matrix[k]
        # Get return_indices_ranked_by of the smallest prob of class k for examples with noisy label k
        if num_issues >= 1:
            class_probs = pred_probs[:, k]
            order = np.argsort(class_probs)
            label_issues[order[:num_issues]] = True
        return label_issues

    warnings.warn(
        f"May not flag all label issues in class: {k}, it has too few examples (see argument: `min_examples_per_class`)"
    )
    return label_issues

def _prune_by_count(args: list) -> np.ndarray:

    k, min_examples_per_class, arrays = args
    if arrays is None:
        pred_probs = pred_probs_by_class[k]
        prune_count_matrix = prune_count_matrix_cols[k]
    else:
        pred_probs = arrays[0]
        prune_count_matrix = arrays[1]

    label_counts = pred_probs.shape[0]
    label_issues_mask = np.zeros(label_counts, dtype=bool)
    if label_counts <= min_examples_per_class:
        warnings.warn(
            f"May not flag all label issues in class: {k}, it has too few examples (see `min_examples_per_class` argument)"
        )
        return label_issues_mask

    K = pred_probs.shape[1]
    if K < 1:
        raise ValueError("Must have at least 1 class.")
    for j in range(K):
        num2prune = prune_count_matrix[j]
        # Only prune for noise rates, not diagonal entries
        if k != j and num2prune > 0:
            # num2prune's largest p(true class k) - p(noisy class k)
            # for x with true label j
            margin = pred_probs[:, j] - pred_probs[:, k]
            order = np.argsort(-margin)
            label_issues_mask[order[:num2prune]] = True
    return label_issues_mask

def _multiclass_crossval_predict(
    labels: list, pred_probs: np.ndarray
) -> np.ndarray:  # pragma: no cover
    

    from sklearn.metrics import f1_score

    boundaries = np.arange(0.05, 0.9, 0.05)
    K = get_num_classes(
        labels=labels,
        pred_probs=pred_probs,
        multi_label=True,
    )
    labels_one_hot = int2onehot(labels, K)
    f1s = [
        f1_score(
            labels_one_hot,
            (pred_probs > boundary).astype(np.uint8),
            average="micro",
        )
        for boundary in boundaries
    ]
    boundary = boundaries[np.argmax(f1s)]
    pred = (pred_probs > boundary).astype(np.uint8)
    return pred
