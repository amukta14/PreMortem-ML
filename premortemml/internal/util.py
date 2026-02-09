from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from premortemml.internal.constants import FLOATING_POINT_COMPARISON, TINY_VALUE
from premortemml.internal.validation import labels_to_array
from premortemml.typing import DatasetLike, LabelLike

def remove_noise_from_class(noise_matrix: np.ndarray, class_without_noise: int) -> np.ndarray:

    # Number of classes
    K = len(noise_matrix)

    cwn = class_without_noise
    x = np.copy(noise_matrix)

    # Set P( labels = cwn | y != cwn) = 0 (no noise)
    class_arange = np.arange(K)
    x[cwn, class_arange[class_arange != cwn]] = 0.0

    # Normalize columns by increasing diagonal terms
    # Ensures noise_matrix is a valid probability matrix
    np.fill_diagonal(x, 1 - (np.sum(x, axis=0) - np.diag(x)))
    return x

def clip_noise_rates(noise_matrix: np.ndarray) -> np.ndarray:

    # Preserve because diagonal entries are not noise rates.
    diagonal = np.diagonal(noise_matrix)

    # Clip all noise rates (efficiently).
    noise_matrix = np.clip(noise_matrix, 0, 0.9999)

    # Put unmodified diagonal back.
    np.fill_diagonal(noise_matrix, diagonal)

    # Re-normalized noise_matrix so that columns sum to one.
    noise_matrix = noise_matrix / np.clip(noise_matrix.sum(axis=0), a_min=TINY_VALUE, a_max=None)
    return noise_matrix

def clip_values(x, low=0.0, high=1.0, new_sum: Optional[float] = None) -> np.ndarray:

    if len(x.shape) > 1:
        raise TypeError(
            f"only size-1 arrays can be converted to Python scalars but 'x' had shape {x.shape}"
        )
    prev_sum = np.sum(x) if new_sum is None else new_sum  # Store previous sum
    x = np.clip(x, low, high)  # Clip all values (efficiently)
    x = (
        x * prev_sum / np.clip(np.sum(x), a_min=TINY_VALUE, a_max=None)
    )  # Re-normalized values to sum to previous sum
    return x

def value_counts(x, *, num_classes: Optional[int] = None, multi_label=False) -> np.ndarray:

    # Efficient method if x is pd.Series, np.ndarray, or list
    if multi_label:
        x = [z for lst in x for z in lst]  # Flatten
    unique_classes, counts = np.unique(x, return_counts=True)

    # Early exit if num_classes is not provided or redundant
    if num_classes is None or num_classes == len(unique_classes):
        return counts

    # Else, there are missing classes
    labels_are_integers = np.issubdtype(np.array(x).dtype, np.integer)
    if labels_are_integers and num_classes <= np.max(unique_classes):
        raise ValueError(f"Required: num_classes > max(x), but {num_classes} <= {np.max(x)}.")

    # Add zero counts for all missing classes in [0, 1,..., num_classes-1]
    total_counts = np.zeros(num_classes, dtype=int)
    # Fill in counts for classes that are present.
    # If labels are integers, unique_classes can be used directly as indices to place counts
    # into the correct positions in total_counts array.
    # If labels are strings, use a slice to fill counts sequentially since strings do not map to indices.
    count_ids = unique_classes if labels_are_integers else slice(len(unique_classes))
    total_counts[count_ids] = counts

    # Return counts with zeros for all missing classes.
    return total_counts

def value_counts_fill_missing_classes(x, num_classes, *, multi_label=False) -> np.ndarray:

    return value_counts(x, num_classes=num_classes, multi_label=multi_label)

def get_missing_classes(labels, *, pred_probs=None, num_classes=None, multi_label=False):
    if pred_probs is None and num_classes is None:
        raise ValueError("Both pred_probs and num_classes are None. You must provide exactly one.")
    if pred_probs is not None and num_classes is not None:
        raise ValueError("Both pred_probs and num_classes are not None. Only one may be provided.")
    if num_classes is None:
        num_classes = pred_probs.shape[1]
    unique_classes = get_unique_classes(labels, multi_label=multi_label)
    return sorted(set(range(num_classes)).difference(unique_classes))

def round_preserving_sum(iterable) -> np.ndarray:

    floats = np.asarray(iterable, dtype=float)
    ints = floats.round()
    orig_sum = np.sum(floats).round()
    int_sum = np.sum(ints).round()
    # Adjust the integers so that they sum to orig_sum
    while abs(int_sum - orig_sum) > FLOATING_POINT_COMPARISON:
        diff = np.round(orig_sum - int_sum)
        increment = -1 if int(diff < 0.0) else 1
        changes = min(int(abs(diff)), len(iterable))
        # Orders indices by difference. Increments # of changes.
        indices = np.argsort(floats - ints)[::-increment][:changes]
        for i in indices:
            ints[i] = ints[i] + increment
        int_sum = np.sum(ints).round()
    return ints.astype(int)

def round_preserving_row_totals(confident_joint) -> np.ndarray:

    return np.apply_along_axis(
        func1d=round_preserving_sum,
        axis=1,
        arr=confident_joint,
    ).astype(int)

def estimate_pu_f1(s, prob_s_eq_1) -> float:

    pred = np.asarray(prob_s_eq_1) >= 0.5
    true_positives = sum((np.asarray(s) == 1) & (np.asarray(pred) == 1))
    all_positives = sum(s)
    recall = true_positives / float(all_positives)
    frac_positive = sum(pred) / float(len(s))
    return recall**2 / (2.0 * frac_positive) if frac_positive != 0 else np.nan

def confusion_matrix(true, pred) -> np.ndarray:

    assert len(true) == len(pred)
    true_classes = np.unique(true)
    pred_classes = np.unique(pred)
    K_true = len(true_classes)  # Number of classes in true
    K_pred = len(pred_classes)  # Number of classes in pred
    map_true = dict(zip(true_classes, range(K_true)))
    map_pred = dict(zip(pred_classes, range(K_pred)))

    result = np.zeros((K_true, K_pred))
    for i in range(len(true)):
        result[map_true[true[i]]][map_pred[pred[i]]] += 1

    return result

def print_square_matrix(
    matrix,
    left_name="s",
    top_name="y",
    title=" A square matrix",
    short_title="s,y",
    round_places=2,
):
    

    short_title = short_title[:6]
    K = len(matrix)  # Number of classes
    # Make sure matrix is 2d array
    if len(np.shape(matrix)) == 1:
        matrix = np.array([matrix])
    print()
    print(title, "of shape", matrix.shape)
    print(" " + short_title + "".join(["\t" + top_name + "=" + str(i) for i in range(K)]))
    print("\t---" * K)
    for i in range(K):
        entry = "\t".join([str(z) for z in list(matrix.round(round_places)[i, :])])
        print(left_name + "=" + str(i) + " |\t" + entry)
    print("\tTrace(matrix) =", np.round(np.trace(matrix), round_places))
    print()

def print_noise_matrix(noise_matrix, round_places=2):
    print_square_matrix(
        noise_matrix,
        title=" Noise Matrix (aka Noisy Channel) P(given_label|true_label)",
        short_title="p(s|y)",
        round_places=round_places,
    )

def print_inverse_noise_matrix(inverse_noise_matrix, round_places=2):
    print_square_matrix(
        inverse_noise_matrix,
        left_name="y",
        top_name="s",
        title=" Inverse Noise Matrix P(true_label|given_label)",
        short_title="p(y|s)",
        round_places=round_places,
    )

def print_joint_matrix(joint_matrix, round_places=2):
    print_square_matrix(
        joint_matrix,
        title=" Joint Label Noise Distribution Matrix P(given_label, true_label)",
        short_title="p(s,y)",
        round_places=round_places,
    )

def compress_int_array(int_array, num_possible_values) -> np.ndarray:
    try:
        compressed_type = None
        if num_possible_values < np.iinfo(np.dtype("int16")).max:
            compressed_type = "int16"
        elif num_possible_values < np.iinfo(np.dtype("int32")).max:  # pragma: no cover
            compressed_type = "int32"  # pragma: no cover
        if compressed_type is not None:
            int_array = int_array.astype(compressed_type)
        return int_array
    except Exception:  # int_array may not even be numpy array, keep as is then
        return int_array

def train_val_split(
    X, labels, train_idx, holdout_idx
) -> Tuple[DatasetLike, DatasetLike, LabelLike, LabelLike]:
    
    labels_train, labels_holdout = (
        labels[train_idx],
        labels[holdout_idx],
    )  # labels are always np.ndarray
    split_completed = False
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X_train, X_holdout = X.iloc[train_idx], X.iloc[holdout_idx]
        split_completed = True
    if not split_completed:
        try:  # check if X is pytorch Dataset object using lazy import
            import torch

            if isinstance(X, torch.utils.data.Dataset):  # special splitting for pytorch Dataset
                X_train = torch.utils.data.Subset(X, train_idx)
                X_holdout = torch.utils.data.Subset(X, holdout_idx)
                split_completed = True
        except Exception:
            pass
    if not split_completed:
        try:
            X_train, X_holdout = X[train_idx], X[holdout_idx]
        except Exception:
            raise ValueError(
                "PreMortemML cannot split this form of dataset (required for cross-validation). "
                "Try a different data format, "
                "or implement the cross-validation yourself and instead provide out-of-sample `pred_probs`"
            )

    return X_train, X_holdout, labels_train, labels_holdout

def subset_X_y(X, labels, mask) -> Tuple[DatasetLike, LabelLike]:
    labels = subset_labels(labels, mask)
    X = subset_data(X, mask)
    return X, labels

def subset_labels(labels, mask) -> Union[list, np.ndarray, pd.Series]:
    try:  # filtering labels as if it is array or DataFrame
        return labels[mask]
    except Exception:
        try:  # filtering labels as if it is list
            return [l for idx, l in enumerate(labels) if mask[idx]]
        except Exception:
            raise TypeError("labels must be 1D np.ndarray, list, or pd.Series.")

def subset_data(X, mask) -> DatasetLike:
    try:
        import torch

        if isinstance(X, torch.utils.data.Dataset):
            mask_idx_list = list(np.nonzero(mask)[0])
            return torch.utils.data.Subset(X, mask_idx_list)
    except Exception:
        pass
    try:
        return X[mask]
    except Exception:
        raise TypeError("Data features X must be subsettable with boolean mask array: X[mask]")

def is_torch_dataset(X) -> bool:
    try:
        import torch

        if isinstance(X, torch.utils.data.Dataset):
            return True
    except Exception:
        pass
    return False  # assumes this cannot be torch dataset if torch cannot be imported

def csr_vstack(a, b) -> DatasetLike:
    a.data = np.hstack((a.data, b.data))
    a.indices = np.hstack((a.indices, b.indices))
    a.indptr = np.hstack((a.indptr, (b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0] + b.shape[0], b.shape[1])
    return a

def append_extra_datapoint(to_data, from_data, index) -> DatasetLike:
    if not (type(from_data) is type(to_data)):
        raise ValueError("Cannot append datapoint from different type of data object.")

    if isinstance(to_data, np.ndarray):
        return np.vstack([to_data, from_data[index]])
    elif isinstance(from_data, (pd.DataFrame, pd.Series)):
        X_extra = from_data.iloc[[index]]  # type: ignore
        to_data = pd.concat([to_data, X_extra])
        return to_data.reset_index(drop=True)
    else:
        try:
            X_extra = from_data[index]
            try:
                return to_data.append(X_extra)
            except Exception:  # special append for sparse matrix
                return csr_vstack(to_data, X_extra)
        except Exception:
            raise TypeError("Data features X must support: X.append(X[i])")

def get_num_classes(labels=None, pred_probs=None, label_matrix=None, multi_label=None) -> int:
    if pred_probs is not None:  # pred_probs is number 1 source of truth
        return pred_probs.shape[1]

    if label_matrix is not None:  # matrix dimension is number 2 source of truth
        if label_matrix.shape[0] != label_matrix.shape[1]:
            raise ValueError(f"label matrix must be K x K, not {label_matrix.shape}")
        else:
            return label_matrix.shape[0]

    if labels is None:
        raise ValueError("Cannot determine number of classes from None input")

    return num_unique_classes(labels, multi_label=multi_label)

def num_unique_classes(labels, multi_label=None) -> int:
    return len(get_unique_classes(labels, multi_label))

def get_unique_classes(labels, multi_label=None) -> set:
    if multi_label is None:
        multi_label = any(isinstance(l, list) for l in labels)
    if multi_label:
        return set(l for grp in labels for l in list(grp))
    else:
        return set(labels)

def format_labels(labels: LabelLike) -> Tuple[np.ndarray, dict]:
    labels = labels_to_array(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be 1D numpy array.")

    unique_labels = np.unique(labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    formatted_labels = np.array([label_map[l] for l in labels])
    inverse_map = {i: label for label, i in label_map.items()}

    return formatted_labels, inverse_map

def smart_display_dataframe(df):  # pragma: no cover
    try:
        from IPython.display import display

        display(df)
    except Exception:
        print(df)

def force_two_dimensions(X) -> DatasetLike:
    if X is not None and len(X.shape) > 2:
        X = X.reshape((len(X), -1))
    return X
