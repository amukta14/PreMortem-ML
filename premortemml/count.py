import warnings
from typing import Optional, Tuple, Union

import numpy as np
import sklearn.base
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from premortemml.internal.constants import (
    CONFIDENT_THRESHOLDS_LOWER_BOUND,
    FLOATING_POINT_COMPARISON,
    TINY_VALUE,
)
from premortemml.internal.latent_algebra import (
    compute_inv_noise_matrix,
    compute_noise_matrix_from_inverse,
    compute_py,
)
from premortemml.internal.multilabel_utils import get_onehot_num_classes, stack_complement
from premortemml.internal.util import (
    append_extra_datapoint,
    clip_noise_rates,
    clip_values,
    get_num_classes,
    get_unique_classes,
    is_torch_dataset,
    round_preserving_row_totals,
    train_val_split,
    value_counts_fill_missing_classes,
)
from premortemml.internal.validation import assert_valid_inputs, labels_to_array
from premortemml.typing import LabelLike

def num_label_issues(
    labels: LabelLike,
    pred_probs: np.ndarray,
    *,
    confident_joint: Optional[np.ndarray] = None,
    estimation_method: str = "off_diagonal",
    multi_label: bool = False,
) -> int:
    
    valid_methods = ["off_diagonal", "off_diagonal_calibrated", "off_diagonal_custom"]
    if isinstance(confident_joint, np.ndarray) and estimation_method != "off_diagonal_custom":
        warn_str = (
            "The supplied `confident_joint` is ignored as `confident_joint` is recomuputed internally using "
            "the supplied `labels` and `pred_probs`. If you still want to use custom `confident_joint` call function "
            "with `estimation_method='off_diagonal_custom'`."
        )
        warnings.warn(warn_str)

    if multi_label:
        return _num_label_issues_multilabel(
            labels=labels,
            pred_probs=pred_probs,
            confident_joint=confident_joint,
        )
    labels = labels_to_array(labels)
    assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs)

    if estimation_method == "off_diagonal":
        _, cl_error_indices = compute_confident_joint(
            labels=labels,
            pred_probs=pred_probs,
            calibrate=False,
            return_indices_of_off_diagonals=True,
        )

        label_issues_mask = np.zeros(len(labels), dtype=bool)
        label_issues_mask[cl_error_indices] = True

        # Remove label issues if model prediction is close to given label
        mask = _reduce_issues(pred_probs=pred_probs, labels=labels)
        label_issues_mask[mask] = False
        num_issues = np.sum(label_issues_mask)
    elif estimation_method == "off_diagonal_calibrated":
        calculated_confident_joint = compute_confident_joint(
            labels=labels,
            pred_probs=pred_probs,
            calibrate=True,
        )
        assert isinstance(calculated_confident_joint, np.ndarray)
        # Estimate_joint calibrates the row sums to match the prior distribution of given labels and normalizes to sum to 1
        joint = estimate_joint(labels, pred_probs, confident_joint=calculated_confident_joint)
        frac_issues = 1.0 - joint.trace()
        num_issues = np.rint(frac_issues * len(labels)).astype(int)
    elif estimation_method == "off_diagonal_custom":
        if not isinstance(confident_joint, np.ndarray):
            raise ValueError(
                f
            )
        else:
            joint = estimate_joint(labels, pred_probs, confident_joint=confident_joint)
            frac_issues = 1.0 - joint.trace()
            num_issues = np.rint(frac_issues * len(labels)).astype(int)
    else:
        raise ValueError(
            f
        )

    return num_issues

def _num_label_issues_multilabel(
    labels: LabelLike,
    pred_probs: np.ndarray,
    confident_joint: Optional[np.ndarray] = None,
) -> int:
    

    from premortemml.filter import find_label_issues

    issues_idx = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        confident_joint=confident_joint,
        multi_label=True,
        filter_by="confident_learning",  # specified to match num_label_issues
    )
    return sum(issues_idx)

def _reduce_issues(pred_probs, labels):
    pred_probs_copy = np.copy(pred_probs)  # Make a copy of the original array
    pred_probs_copy[np.arange(len(labels)), labels] += FLOATING_POINT_COMPARISON
    pred = pred_probs_copy.argmax(axis=1)
    mask = pred == labels
    del pred_probs_copy  # Delete copy
    return mask

def calibrate_confident_joint(
    confident_joint: np.ndarray, labels: LabelLike, *, multi_label: bool = False
) -> np.ndarray:
    

    if multi_label:
        if not isinstance(labels, list):
            raise TypeError("`labels` must be list when `multi_label=True`.")
        else:
            return _calibrate_confident_joint_multilabel(confident_joint, labels)
    else:
        num_classes = len(confident_joint)
        label_counts = value_counts_fill_missing_classes(labels, num_classes, multi_label=False)
    # Calibrate confident joint to have correct p(labels) prior on noisy labels.
    calibrated_cj = (
        confident_joint.T
        / np.clip(confident_joint.sum(axis=1), a_min=TINY_VALUE, a_max=None)
        * label_counts
    ).T
    # Calibrate confident joint to sum to:
    # The number of examples (for single labeled datasets)
    # The number of total labels (for multi-labeled datasets)
    calibrated_cj = (
        calibrated_cj
        / np.clip(np.sum(calibrated_cj), a_min=TINY_VALUE, a_max=None)
        * sum(label_counts)
    )
    return round_preserving_row_totals(calibrated_cj)

def _calibrate_confident_joint_multilabel(confident_joint: np.ndarray, labels: list) -> np.ndarray:
    y_one, num_classes = get_onehot_num_classes(labels)
    calibrate_confident_joint_list: np.ndarray = np.ndarray(
        shape=(num_classes, 2, 2), dtype=np.int64
    )
    for class_num, (cj, y) in enumerate(zip(confident_joint, y_one.T)):
        calibrate_confident_joint_list[class_num] = calibrate_confident_joint(cj, labels=y)

    return calibrate_confident_joint_list

def estimate_joint(
    labels: LabelLike,
    pred_probs: np.ndarray,
    *,
    confident_joint: Optional[np.ndarray] = None,
    multi_label: bool = False,
) -> np.ndarray:
    

    if confident_joint is None:
        calibrated_cj = compute_confident_joint(
            labels,
            pred_probs,
            calibrate=True,
            multi_label=multi_label,
        )
    else:
        if labels is not None:
            calibrated_cj = calibrate_confident_joint(
                confident_joint, labels, multi_label=multi_label
            )
        else:
            calibrated_cj = confident_joint

    assert isinstance(calibrated_cj, np.ndarray)
    if multi_label:
        if not isinstance(labels, list):
            raise TypeError("`labels` must be list when `multi_label=True`.")
        else:
            return _estimate_joint_multilabel(
                labels=labels, pred_probs=pred_probs, confident_joint=confident_joint
            )
    else:
        return calibrated_cj / np.clip(float(np.sum(calibrated_cj)), a_min=TINY_VALUE, a_max=None)

def _estimate_joint_multilabel(
    labels: list, pred_probs: np.ndarray, *, confident_joint: Optional[np.ndarray] = None
) -> np.ndarray:
    
    y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
    if confident_joint is None:
        calibrated_cj = compute_confident_joint(
            labels,
            pred_probs,
            calibrate=True,
            multi_label=True,
        )
    else:
        calibrated_cj = confident_joint
    assert isinstance(calibrated_cj, np.ndarray)
    calibrated_cf: np.ndarray = np.ndarray((num_classes, 2, 2))
    for class_num, (label, pred_prob_for_class) in enumerate(zip(y_one.T, pred_probs.T)):
        pred_probs_binary = stack_complement(pred_prob_for_class)
        calibrated_cf[class_num] = estimate_joint(
            labels=label,
            pred_probs=pred_probs_binary,
            confident_joint=calibrated_cj[class_num],
        )

    return calibrated_cf

def compute_confident_joint(
    labels: LabelLike,
    pred_probs: np.ndarray,
    *,
    thresholds: Optional[Union[np.ndarray, list]] = None,
    calibrate: bool = True,
    multi_label: bool = False,
    return_indices_of_off_diagonals: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    

    if multi_label:
        if not isinstance(labels, list):
            raise TypeError("`labels` must be list when `multi_label=True`.")

        return _compute_confident_joint_multi_label(
            labels=labels,
            pred_probs=pred_probs,
            thresholds=thresholds,
            calibrate=calibrate,
            return_indices_of_off_diagonals=return_indices_of_off_diagonals,
        )

    # labels needs to be a numpy array
    labels = np.asarray(labels)

    # Estimate the probability thresholds for confident counting
    if thresholds is None:
        # P(we predict the given noisy label is k | given noisy label is k)
        thresholds = get_confident_thresholds(labels, pred_probs, multi_label=multi_label)
    thresholds = np.asarray(thresholds)

    # Compute confident joint (vectorized for speed).

    # pred_probs_bool is a bool matrix where each row represents a training example as a boolean vector of
    # size num_classes, with True if the example confidently belongs to that class and False if not.
    pred_probs_bool = pred_probs >= thresholds - 1e-6
    num_confident_bins = pred_probs_bool.sum(axis=1)
    # The indices where this is false, are often outliers (not confident of any label)
    at_least_one_confident = num_confident_bins > 0
    more_than_one_confident = num_confident_bins > 1
    pred_probs_argmax = pred_probs.argmax(axis=1)
    # Note that confident_argmax is meaningless for rows of all False
    confident_argmax = pred_probs_bool.argmax(axis=1)
    # For each example, choose the confident class (greater than threshold)
    # When there is 2+ confident classes, choose the class with largest prob.
    true_label_guess = np.where(
        more_than_one_confident,
        pred_probs_argmax,
        confident_argmax,
    )
    # true_labels_confident omits meaningless all-False rows
    true_labels_confident = true_label_guess[at_least_one_confident]
    labels_confident = labels[at_least_one_confident]

    # Handle case where no examples are confident (sklearn >=1.8.0 compatibility)
    # In sklearn <1.8.0, confusion_matrix with empty inputs returned zeros matrix
    # In sklearn >=1.8.0, it raises ValueError - we emulate the old behavior
    if len(true_labels_confident) == 0:
        confident_joint = np.zeros((pred_probs.shape[1], pred_probs.shape[1]), dtype=int)
    else:
        confident_joint = confusion_matrix(
            y_true=true_labels_confident,
            y_pred=labels_confident,
            labels=range(pred_probs.shape[1]),
        ).T
    # Guarantee at least one correctly labeled example is represented in every class
    np.fill_diagonal(confident_joint, confident_joint.diagonal().clip(min=1))
    if calibrate:
        confident_joint = calibrate_confident_joint(confident_joint, labels)

    if return_indices_of_off_diagonals:
        true_labels_neq_given_labels = true_labels_confident != labels_confident
        indices = np.arange(len(labels))[at_least_one_confident][true_labels_neq_given_labels]

        return confident_joint, indices

    return confident_joint

def _compute_confident_joint_multi_label(
    labels: list,
    pred_probs: np.ndarray,
    *,
    thresholds: Optional[Union[np.ndarray, list]] = None,
    calibrate: bool = True,
    return_indices_of_off_diagonals: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    

    y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
    confident_joint_list: np.ndarray = np.ndarray(shape=(num_classes, 2, 2), dtype=np.int64)
    indices_off_diagonal = []
    for class_num, (label, pred_prob_for_class) in enumerate(zip(y_one.T, pred_probs.T)):
        pred_probs_binary = stack_complement(pred_prob_for_class)
        if return_indices_of_off_diagonals:
            cj, ind = compute_confident_joint(
                labels=label,
                pred_probs=pred_probs_binary,
                multi_label=False,
                thresholds=thresholds,
                calibrate=calibrate,
                return_indices_of_off_diagonals=return_indices_of_off_diagonals,
            )
            indices_off_diagonal.append(ind)
        else:
            cj = compute_confident_joint(
                labels=label,
                pred_probs=pred_probs_binary,
                multi_label=False,
                thresholds=thresholds,
                calibrate=calibrate,
                return_indices_of_off_diagonals=return_indices_of_off_diagonals,
            )
        confident_joint_list[class_num] = cj

    if return_indices_of_off_diagonals:
        return confident_joint_list, indices_off_diagonal

    return confident_joint_list

def estimate_latent(
    confident_joint: np.ndarray,
    labels: np.ndarray,
    *,
    py_method: str = "cnt",
    converge_latent_estimates: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    

    num_classes = len(confident_joint)
    label_counts = value_counts_fill_missing_classes(labels, num_classes)
    # 'ps' is p(labels=k)
    ps = label_counts / float(len(labels))
    # Number of training examples confidently counted from each noisy class
    labels_class_counts = confident_joint.sum(axis=1).astype(float)
    # Number of training examples confidently counted into each true class
    true_labels_class_counts = confident_joint.sum(axis=0).astype(float)
    # p(label=k_s|true_label=k_y) ~ |label=k_s and true_label=k_y| / |true_label=k_y|
    noise_matrix = confident_joint / np.clip(true_labels_class_counts, a_min=TINY_VALUE, a_max=None)
    # p(true_label=k_y|label=k_s) ~ |true_label=k_y and label=k_s| / |label=k_s|
    inv_noise_matrix = confident_joint.T / np.clip(
        labels_class_counts, a_min=TINY_VALUE, a_max=None
    )
    # Compute the prior p(y), the latent (uncorrupted) class distribution.
    py = compute_py(
        ps,
        noise_matrix,
        inv_noise_matrix,
        py_method=py_method,
        true_labels_class_counts=true_labels_class_counts,
    )
    # Clip noise rates to be valid probabilities.
    noise_matrix = clip_noise_rates(noise_matrix)
    inv_noise_matrix = clip_noise_rates(inv_noise_matrix)
    # Make latent estimates mathematically agree in their algebraic relations.
    if converge_latent_estimates:
        py, noise_matrix, inv_noise_matrix = _converge_estimates(
            ps, py, noise_matrix, inv_noise_matrix
        )
        # Again clip py and noise rates into proper range [0,1)
        py = clip_values(py, low=1e-5, high=1.0, new_sum=1.0)
        noise_matrix = clip_noise_rates(noise_matrix)
        inv_noise_matrix = clip_noise_rates(inv_noise_matrix)

    return py, noise_matrix, inv_noise_matrix

def estimate_py_and_noise_matrices_from_probabilities(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    thresholds: Optional[Union[np.ndarray, list]] = None,
    converge_latent_estimates: bool = True,
    py_method: str = "cnt",
    calibrate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    

    confident_joint = compute_confident_joint(
        labels=labels,
        pred_probs=pred_probs,
        thresholds=thresholds,
        calibrate=calibrate,
    )
    assert isinstance(confident_joint, np.ndarray)
    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        labels=labels,
        py_method=py_method,
        converge_latent_estimates=converge_latent_estimates,
    )
    assert isinstance(confident_joint, np.ndarray)

    return py, noise_matrix, inv_noise_matrix, confident_joint

def estimate_confident_joint_and_cv_pred_proba(
    X,
    labels,
    clf=LogReg(solver="lbfgs"),
    *,
    cv_n_folds=5,
    thresholds=None,
    seed=None,
    calibrate=True,
    clf_kwargs={},
    validation_func=None,
) -> Tuple[np.ndarray, np.ndarray]:
    

    assert_valid_inputs(X, labels)
    labels = labels_to_array(labels)
    num_classes = get_num_classes(
        labels=labels
    )  # This method definitely only works if all classes are present.

    # Create cross-validation object for out-of-sample predicted probabilities.
    # CV folds preserve the fraction of noisy positive and
    # noisy negative examples in each class.
    kf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=seed)

    # Initialize pred_probs array
    pred_probs = np.zeros(shape=(len(labels), num_classes))

    # Split X and labels into "cv_n_folds" stratified folds.
    # CV indices only require labels: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    # Only split based on labels because X may have various formats:
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(X=labels, y=labels)):
        try:
            clf_copy = sklearn.base.clone(clf)  # fresh untrained copy of the model
        except Exception:
            raise ValueError(
                "`clf` must be clonable via: sklearn.base.clone(clf). "
                "You can either implement instance method `clf.get_params()` to produce a fresh untrained copy of this model, "
            )
        # Select the training and holdout cross-validated sets.
        X_train_cv, X_holdout_cv, s_train_cv, s_holdout_cv = train_val_split(
            X, labels, cv_train_idx, cv_holdout_idx
        )

        # dict with keys: which classes missing, values: index of holdout data from this class that is duplicated:
        missing_class_inds = {}
        if not is_torch_dataset(X):
            # Ensure no missing classes in training set.
            train_cv_classes = set(s_train_cv)
            all_classes = set(range(num_classes))
            if len(train_cv_classes) != len(all_classes):
                missing_classes = all_classes.difference(train_cv_classes)
                warnings.warn(
                    "Duplicated some data across multiple folds to ensure training does not fail "
                    f"because these classes do not have enough data for proper cross-validation: {missing_classes}."
                )
                for missing_class in missing_classes:
                    # Duplicate one instance of missing_class from holdout data to the training data:
                    holdout_inds = np.where(s_holdout_cv == missing_class)[0]
                    dup_idx = holdout_inds[0]
                    s_train_cv = np.append(s_train_cv, s_holdout_cv[dup_idx])
                    # labels are always np.ndarray so don't have to consider .iloc above
                    X_train_cv = append_extra_datapoint(
                        to_data=X_train_cv, from_data=X_holdout_cv, index=dup_idx
                    )
                    missing_class_inds[missing_class] = dup_idx

        # Map validation data into appropriate format to pass into classifier clf
        if validation_func is None:
            validation_kwargs = {}
        elif callable(validation_func):
            validation_kwargs = validation_func(X_holdout_cv, s_holdout_cv)
        else:
            raise TypeError("validation_func must be callable function with args: X_val, y_val")

        # Fit classifier clf to training set, predict on holdout set, and update pred_probs.
        clf_copy.fit(X_train_cv, s_train_cv, **clf_kwargs, **validation_kwargs)
        pred_probs_cv = clf_copy.predict_proba(X_holdout_cv)  # P(labels = k|x) # [:,1]

        # Replace predictions for duplicated indices with dummy predictions:
        for missing_class in missing_class_inds:
            dummy_pred = np.zeros(pred_probs_cv[0].shape)
            dummy_pred[missing_class] = 1.0  # predict given label with full confidence
            dup_idx = missing_class_inds[missing_class]
            pred_probs_cv[dup_idx] = dummy_pred

        pred_probs[cv_holdout_idx] = pred_probs_cv

    # Compute the confident counts, a num_classes x num_classes matrix for all pairs of labels.
    confident_joint = compute_confident_joint(
        labels=labels,
        pred_probs=pred_probs,  # P(labels = k|x)
        thresholds=thresholds,
        calibrate=calibrate,
    )
    assert isinstance(confident_joint, np.ndarray)
    assert isinstance(pred_probs, np.ndarray)

    return confident_joint, pred_probs

def estimate_py_noise_matrices_and_cv_pred_proba(
    X,
    labels,
    clf=LogReg(solver="lbfgs"),
    *,
    cv_n_folds=5,
    thresholds=None,
    converge_latent_estimates=False,
    py_method="cnt",
    seed=None,
    clf_kwargs={},
    validation_func=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    confident_joint, pred_probs = estimate_confident_joint_and_cv_pred_proba(
        X=X,
        labels=labels,
        clf=clf,
        cv_n_folds=cv_n_folds,
        thresholds=thresholds,
        seed=seed,
        clf_kwargs=clf_kwargs,
        validation_func=validation_func,
    )

    py, noise_matrix, inv_noise_matrix = estimate_latent(
        confident_joint=confident_joint,
        labels=labels,
        py_method=py_method,
        converge_latent_estimates=converge_latent_estimates,
    )

    return py, noise_matrix, inv_noise_matrix, confident_joint, pred_probs

def estimate_cv_predicted_probabilities(
    X,
    labels,
    clf=LogReg(solver="lbfgs"),
    *,
    cv_n_folds=5,
    seed=None,
    clf_kwargs={},
    validation_func=None,
) -> np.ndarray:
    

    return estimate_py_noise_matrices_and_cv_pred_proba(
        X=X,
        labels=labels,
        clf=clf,
        cv_n_folds=cv_n_folds,
        seed=seed,
        clf_kwargs=clf_kwargs,
        validation_func=validation_func,
    )[-1]

def estimate_noise_matrices(
    X,
    labels,
    clf=LogReg(solver="lbfgs"),
    *,
    cv_n_folds=5,
    thresholds=None,
    converge_latent_estimates=True,
    seed=None,
    clf_kwargs={},
    validation_func=None,
) -> Tuple[np.ndarray, np.ndarray]:
    

    return estimate_py_noise_matrices_and_cv_pred_proba(
        X=X,
        labels=labels,
        clf=clf,
        cv_n_folds=cv_n_folds,
        thresholds=thresholds,
        converge_latent_estimates=converge_latent_estimates,
        seed=seed,
        clf_kwargs=clf_kwargs,
        validation_func=validation_func,
    )[1:-2]

def _converge_estimates(
    ps: np.ndarray,
    py: np.ndarray,
    noise_matrix: np.ndarray,
    inverse_noise_matrix: np.ndarray,
    *,
    inv_noise_matrix_iterations: int = 5,
    noise_matrix_iterations: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    

    for j in range(noise_matrix_iterations):
        for i in range(inv_noise_matrix_iterations):
            inverse_noise_matrix = compute_inv_noise_matrix(py=py, noise_matrix=noise_matrix, ps=ps)
            py = compute_py(ps, noise_matrix, inverse_noise_matrix)
        noise_matrix = compute_noise_matrix_from_inverse(
            ps=ps, inverse_noise_matrix=inverse_noise_matrix, py=py
        )

    return py, noise_matrix, inverse_noise_matrix

def get_confident_thresholds(
    labels: LabelLike,
    pred_probs: np.ndarray,
    multi_label: bool = False,
) -> np.ndarray:
    
    if multi_label:
        assert isinstance(labels, list)
        return _get_confident_thresholds_multilabel(labels=labels, pred_probs=pred_probs)
    else:
        # When all_classes != unique_classes the class threshold for the missing classes is set to
        # BIG_VALUE such that no valid prob >= BIG_VALUE (no example will be counted in missing classes)
        # REQUIRES: pred_probs.max() >= 1
        labels = labels_to_array(labels)
        all_classes = range(pred_probs.shape[1])
        unique_classes = get_unique_classes(labels, multi_label=multi_label)
        BIG_VALUE = 2
        confident_thresholds = [
            np.mean(pred_probs[:, k][labels == k]) if k in unique_classes else BIG_VALUE
            for k in all_classes
        ]
        confident_thresholds = np.clip(
            confident_thresholds, a_min=CONFIDENT_THRESHOLDS_LOWER_BOUND, a_max=None
        )
        return confident_thresholds

def _get_confident_thresholds_multilabel(
    labels: list,
    pred_probs: np.ndarray,
):
    
    y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
    confident_thresholds: np.ndarray = np.ndarray((num_classes, 2))
    for class_num, (label_for_class, pred_prob_for_class) in enumerate(zip(y_one.T, pred_probs.T)):
        pred_probs_binary = stack_complement(pred_prob_for_class)
        confident_thresholds[class_num] = get_confident_thresholds(
            pred_probs=pred_probs_binary, labels=label_for_class
        )
    return confident_thresholds
