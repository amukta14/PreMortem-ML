import warnings
from typing import Optional, Tuple

import numpy as np

from premortemml.internal.segmentation_utils import _check_input, _get_valid_optional_params
from premortemml.segmentation.filter import find_label_issues

def get_label_quality_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    method: str = "softmin",
    batch_size: Optional[int] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    
    batch_size, n_jobs = _get_valid_optional_params(batch_size, n_jobs)
    _check_input(labels, pred_probs)

    softmin_temperature = kwargs.get("temperature", 0.1)
    downsample_num_pixel_issues = kwargs.get("downsample", 1)

    if method == "num_pixel_issues":
        _, K, _, _ = pred_probs.shape
        labels_expanded = labels[:, np.newaxis, :, :]
        mask = np.arange(K)[np.newaxis, :, np.newaxis, np.newaxis] == labels_expanded
        # Calculate pixel_scores
        masked_pred_probs = np.where(mask, pred_probs, 0)
        pixel_scores = masked_pred_probs.sum(axis=1)
        scores = find_label_issues(
            labels,
            pred_probs,
            downsample=downsample_num_pixel_issues,
            n_jobs=n_jobs,
            verbose=verbose,
            batch_size=batch_size,
        )
        img_scores = 1 - np.mean(scores, axis=(1, 2))
        return (img_scores, pixel_scores)

    if downsample_num_pixel_issues != 1:
        warnings.warn(
            f"image will not downsample for method {method} is only for method: num_pixel_issues"
        )

    num_im, num_class, h, w = pred_probs.shape
    image_scores = np.empty((num_im,))
    pixel_scores = np.empty((num_im, h, w))
    if verbose:
        from tqdm.auto import tqdm

        pbar = tqdm(desc=f"images processed using {method}", total=num_im)

    h_array = np.arange(h)[:, None]
    w_array = np.arange(w)

    for image in range(num_im):
        image_probs = pred_probs[image][
            labels[image],
            h_array,
            w_array,
        ]
        pixel_scores[image, :, :] = image_probs
        image_scores[image] = _get_label_quality_per_image(
            image_probs.flatten(), method=method, temperature=softmin_temperature
        )
        if verbose:
            pbar.update(1)
    return image_scores, pixel_scores

def issues_from_scores(
    image_scores: np.ndarray, pixel_scores: Optional[np.ndarray] = None, threshold: float = 0.1
) -> np.ndarray:
    

    if image_scores is None:
        raise ValueError("pixel_scores must be provided")
    if threshold < 0 or threshold > 1 or threshold is None:
        raise ValueError("threshold must be between 0 and 1")

    if pixel_scores is not None:
        return pixel_scores < threshold

    ranking = np.argsort(image_scores)
    cutoff = np.searchsorted(image_scores[ranking], threshold)
    return ranking[: cutoff + 1]

def _get_label_quality_per_image(pixel_scores, method=None, temperature=0.1):
    from premortemml.internal.multilabel_scorer import softmin

    
    if pixel_scores is None or pixel_scores.size == 0:
        raise Exception("Invalid Input: pixel_scores cannot be None or an empty list")

    if temperature == 0 or temperature is None:
        raise Exception("Invalid Input: temperature cannot be zero or None")

    pixel_scores_64 = pixel_scores.astype("float64")
    if method == "softmin":
        if len(pixel_scores_64) > 0:
            return softmin(
                np.expand_dims(pixel_scores_64, axis=0), axis=1, temperature=temperature
            )[0]
        else:
            raise Exception("Invalid Input: pixel_scores is empty")
    else:
        raise Exception("Invalid Method: Specify correct method. Currently only supports 'softmin'")
