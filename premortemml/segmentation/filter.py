from typing import Optional, Tuple

import numpy as np

from premortemml.experimental.label_issues_batched import LabelInspector
from premortemml.internal.segmentation_utils import _check_input, _get_valid_optional_params

def find_label_issues(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    batch_size: Optional[int] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
    **kwargs,
) -> np.ndarray:
    
    batch_size, n_jobs = _get_valid_optional_params(batch_size, n_jobs)
    downsample = kwargs.get("downsample", 1)

    def downsample_arrays(
        labels: np.ndarray, pred_probs: np.ndarray, factor: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        if factor == 1:
            return labels, pred_probs

        num_image, num_classes, h, w = pred_probs.shape

        # Check if possible to downsample
        if h % downsample != 0 or w % downsample != 0:
            raise ValueError(
                f"Height {h} and width {w} not divisible by downsample value of {downsample}. Set kwarg downsample to 1 to avoid downsampling."
            )
        small_labels = np.round(
            labels.reshape((num_image, h // factor, factor, w // factor, factor)).mean((4, 2))
        )
        small_pred_probs = pred_probs.reshape(
            (num_image, num_classes, h // factor, factor, w // factor, factor)
        ).mean((5, 3))

        # We want to make sure that pred_probs are renormalized
        row_sums = small_pred_probs.sum(axis=1)
        renorm_small_pred_probs = small_pred_probs / np.expand_dims(row_sums, 1)

        return small_labels, renorm_small_pred_probs

    def flatten_and_preprocess_masks(
        labels: np.ndarray, pred_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, num_classes, _, _ = pred_probs.shape
        labels_flat = labels.flatten().astype(int)
        pred_probs_flat = np.moveaxis(pred_probs, 0, 1).reshape(num_classes, -1)

        return labels_flat, pred_probs_flat.T

    ##
    _check_input(labels, pred_probs)

    # Added Downsampling
    pre_labels, pre_pred_probs = downsample_arrays(labels, pred_probs, downsample)

    num_image, _, h, w = pre_pred_probs.shape

    ### This section is a modified version of find_label_issues_batched()
    lab = LabelInspector(
        num_class=pre_pred_probs.shape[1],
        verbose=verbose,
        n_jobs=n_jobs,
        quality_score_kwargs=None,
        num_issue_kwargs=None,
    )
    n = len(pre_labels)

    if verbose:
        from tqdm.auto import tqdm

        pbar = tqdm(desc="number of examples processed for estimating thresholds", total=n)

    # Precompute the size of each image in the batch
    image_size = np.prod(pre_pred_probs.shape[1:])
    images_per_batch = max(batch_size // image_size, 1)

    for start_index in range(0, n, images_per_batch):
        end_index = min(start_index + images_per_batch, n)
        labels_batch, pred_probs_batch = flatten_and_preprocess_masks(
            pre_labels[start_index:end_index], pre_pred_probs[start_index:end_index]
        )
        lab.update_confident_thresholds(labels_batch, pred_probs_batch)
        if verbose:
            pbar.update(end_index - start_index)

    if verbose:
        pbar.close()
        pbar = tqdm(desc="number of examples processed for checking labels", total=n)

    for start_index in range(0, n, images_per_batch):
        end_index = min(start_index + images_per_batch, n)
        labels_batch, pred_probs_batch = flatten_and_preprocess_masks(
            pre_labels[start_index:end_index], pre_pred_probs[start_index:end_index]
        )
        _ = lab.score_label_quality(labels_batch, pred_probs_batch)
        if verbose:
            pbar.update(end_index - start_index)

    if verbose:
        pbar.close()

    ranked_label_issues = lab.get_label_issues()
    ### End find_label_issues_batched() section

    # Upsample carefully maintaining indicies
    label_issues = np.full((num_image, h, w), False)

    # only want to call it an error if pred_probs doesnt match the label at those pixels
    for i in range(0, ranked_label_issues.shape[0], batch_size):
        issues_batch = ranked_label_issues[i : i + batch_size]
        # Finding the right indicies
        image_batch, batch_coor_i, batch_coor_j = _get_indexes_from_ranked_issues(
            issues_batch, h, w
        )
        label_issues[image_batch, batch_coor_i, batch_coor_j] = True
        if downsample == 1:
            # check if pred_probs matches the label at those pixels
            pred_argmax = np.argmax(pred_probs[image_batch, :, batch_coor_i, batch_coor_j], axis=1)
            mask = pred_argmax == labels[image_batch, batch_coor_i, batch_coor_j]
            label_issues[image_batch[mask], batch_coor_i[mask], batch_coor_j[mask]] = False

    if downsample != 1:
        label_issues = label_issues.repeat(downsample, axis=1).repeat(downsample, axis=2)

        for i in range(0, ranked_label_issues.shape[0], batch_size):
            issues_batch = ranked_label_issues[i : i + batch_size]
            image_batch, batch_coor_i, batch_coor_j = _get_indexes_from_ranked_issues(
                issues_batch, h, w
            )
            # Upsample the coordinates
            upsampled_ii = batch_coor_i * downsample
            upsampled_jj = batch_coor_j * downsample
            # Iterate over the upsampled region
            for i in range(downsample):
                for j in range(downsample):
                    rows = upsampled_ii + i
                    cols = upsampled_jj + j
                    pred_argmax = np.argmax(pred_probs[image_batch, :, rows, cols], axis=1)
                    # Check if the predicted class (argmax) at the identified issue location matches the true label
                    mask = pred_argmax == labels[image_batch, rows, cols]
                    # If they match, set the corresponding entries in the label_issues array to False
                    label_issues[image_batch[mask], rows[mask], cols[mask]] = False

    return label_issues

def _get_indexes_from_ranked_issues(
    ranked_label_issues: np.ndarray, h: int, w: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hw = h * w
    relative_index = ranked_label_issues % hw
    pixel_coor_i, pixel_coor_j = np.unravel_index(relative_index, (h, w))
    image_batch = ranked_label_issues // hw
    return image_batch, pixel_coor_i, pixel_coor_j
