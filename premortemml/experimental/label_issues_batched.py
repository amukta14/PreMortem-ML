import numpy as np
from typing import Optional, List, Tuple, Any

from premortemml.count import get_confident_thresholds, _reduce_issues
from premortemml.rank import find_top_issues, _compute_label_quality_scores
from premortemml.typing import LabelLike
from premortemml.internal.util import value_counts_fill_missing_classes
from premortemml.internal.constants import (
    CONFIDENT_THRESHOLDS_LOWER_BOUND,
    FLOATING_POINT_COMPARISON,
    CLIPPING_LOWER_BOUND,
)

import platform
import multiprocessing as mp

try:
    import psutil

    PSUTIL_EXISTS = True
except ImportError:  # pragma: no cover
    PSUTIL_EXISTS = False

# global variable for multiproc on linux
adj_confident_thresholds_shared: np.ndarray
labels_shared: LabelLike
pred_probs_shared: np.ndarray

def find_label_issues_batched(
    labels: Optional[LabelLike] = None,
    pred_probs: Optional[np.ndarray] = None,
    *,
    labels_file: Optional[str] = None,
    pred_probs_file: Optional[str] = None,
    batch_size: int = 10000,
    n_jobs: Optional[int] = 1,
    verbose: bool = True,
    quality_score_kwargs: Optional[dict] = None,
    num_issue_kwargs: Optional[dict] = None,
    return_mask: bool = False,
) -> np.ndarray:
    
    if labels_file is not None:
        if labels is not None:
            raise ValueError("only specify one of: `labels` or `labels_file`")
        if not isinstance(labels_file, str):
            raise ValueError(
                "labels_file must be str specifying path to .npy file containing the array of labels"
            )
        labels = np.load(labels_file, mmap_mode="r")
        assert isinstance(labels, np.ndarray)

    if pred_probs_file is not None:
        if pred_probs is not None:
            raise ValueError("only specify one of: `pred_probs` or `pred_probs_file`")
        if not isinstance(pred_probs_file, str):
            raise ValueError(
                "pred_probs_file must be str specifying path to .npy file containing 2D array of pred_probs"
            )
        pred_probs = np.load(pred_probs_file, mmap_mode="r")
        assert isinstance(pred_probs, np.ndarray)
        if verbose:
            print(
                f"mmap-loaded numpy arrays have: {len(pred_probs)} examples, {pred_probs.shape[1]} classes"
            )
    if labels is None:
        raise ValueError("must provide one of: `labels` or `labels_file`")
    if pred_probs is None:
        raise ValueError("must provide one of: `pred_probs` or `pred_probs_file`")

    assert pred_probs is not None
    if len(labels) != len(pred_probs):
        raise ValueError(
            f"len(labels)={len(labels)} does not match len(pred_probs)={len(pred_probs)}. Perhaps an issue loading mmap numpy arrays from file."
        )
    lab = LabelInspector(
        num_class=pred_probs.shape[1],
        verbose=verbose,
        n_jobs=n_jobs,
        quality_score_kwargs=quality_score_kwargs,
        num_issue_kwargs=num_issue_kwargs,
    )
    n = len(labels)
    if verbose:
        from tqdm.auto import tqdm

        pbar = tqdm(desc="number of examples processed for estimating thresholds", total=n)
    i = 0
    while i < n:
        end_index = i + batch_size
        labels_batch = labels[i:end_index]
        pred_probs_batch = pred_probs[i:end_index, :]
        i = end_index
        lab.update_confident_thresholds(labels_batch, pred_probs_batch)
        if verbose:
            pbar.update(batch_size)

    # Next evaluate the quality of the labels (run this on full dataset you want to evaluate):
    if verbose:
        pbar.close()
        pbar = tqdm(desc="number of examples processed for checking labels", total=n)
    i = 0
    while i < n:
        end_index = i + batch_size
        labels_batch = labels[i:end_index]
        pred_probs_batch = pred_probs[i:end_index, :]
        i = end_index
        _ = lab.score_label_quality(labels_batch, pred_probs_batch)
        if verbose:
            pbar.update(batch_size)

    if verbose:
        pbar.close()

    label_issues_indices = lab.get_label_issues()
    label_issues_mask = np.zeros(len(labels), dtype=bool)
    label_issues_mask[label_issues_indices] = True
    mask = _reduce_issues(pred_probs=pred_probs, labels=labels)
    label_issues_mask[mask] = False
    if return_mask:
        return label_issues_mask
    return np.where(label_issues_mask)[0]

class LabelInspector:

    def __init__(
        self,
        *,
        num_class: int,
        store_results: bool = True,
        verbose: bool = True,
        quality_score_kwargs: Optional[dict] = None,
        num_issue_kwargs: Optional[dict] = None,
        n_jobs: Optional[int] = 1,
    ):
        if quality_score_kwargs is None:
            quality_score_kwargs = {}
        if num_issue_kwargs is None:
            num_issue_kwargs = {}

        self.num_class = num_class
        self.store_results = store_results
        self.verbose = verbose
        self.quality_score_kwargs = quality_score_kwargs  # extra arguments for ``rank.get_label_quality_scores()`` to control label quality scoring
        self.num_issue_kwargs = num_issue_kwargs  # extra arguments for ``count.num_label_issues()`` to control estimation of the number of label issues (only supported argument for now is: `estimation_method`).
        self.off_diagonal_calibrated = False
        if num_issue_kwargs.get("estimation_method") == "off_diagonal_calibrated":
            # store extra attributes later needed for calibration:
            self.off_diagonal_calibrated = True
            self.prune_counts = np.zeros(self.num_class)
            self.class_counts = np.zeros(self.num_class)
            self.normalization = np.zeros(self.num_class)
        else:
            self.prune_count = 0  # number of label issues estimated based on data seen so far (only used when estimation_method is not calibrated)

        if self.store_results:
            self.label_quality_scores: List[float] = []

        self.confident_thresholds = np.zeros(
            (num_class,)
        )  # current estimate of thresholds based on data seen so far
        self.examples_per_class = np.zeros(
            (num_class,)
        )  # current counts of examples with each given label seen so far
        self.examples_processed_thresh = (
            0  # number of examples seen so far for estimating thresholds
        )
        self.examples_processed_quality = 0  # number of examples seen so far for estimating label quality and number of label issues
        # Determine number of cores for multiprocessing:
        self.n_jobs: Optional[int] = None
        os_name = platform.system()
        if os_name != "Linux":
            self.n_jobs = 1
            if n_jobs is not None and n_jobs != 1 and self.verbose:
                print(
                    "n_jobs is overridden to 1 because multiprocessing is only supported for Linux."
                )
        elif n_jobs is not None:
            self.n_jobs = n_jobs
        else:
            if PSUTIL_EXISTS:
                self.n_jobs = psutil.cpu_count(logical=False)  # physical cores
            if not self.n_jobs:
                # switch to logical cores
                self.n_jobs = mp.cpu_count()
                if self.verbose:
                    print(
                        f"Multiprocessing will default to using the number of logical cores ({self.n_jobs}). To default to number of physical cores: pip install psutil"
                    )

    def get_confident_thresholds(self, silent: bool = False) -> np.ndarray:
        if self.examples_processed_thresh < 1:
            raise ValueError(
                "Have not computed any confident_thresholds yet. Call `update_confident_thresholds()` first."
            )
        else:
            if self.verbose and not silent:
                print(
                    f"Total number of examples used to estimate confident thresholds: {self.examples_processed_thresh}"
                )
            return self.confident_thresholds

    def get_num_issues(self, silent: bool = False) -> int:
        if self.examples_processed_quality < 1:
            raise ValueError(
                "Have not evaluated any labels yet. Call `score_label_quality()` first."
            )
        else:
            if self.verbose and not silent:
                print(
                    f"Total number of examples whose labels have been evaluated: {self.examples_processed_quality}"
                )
            if self.off_diagonal_calibrated:
                calibrated_prune_counts = (
                    self.prune_counts
                    * self.class_counts
                    / np.clip(self.normalization, a_min=CLIPPING_LOWER_BOUND, a_max=None)
                )  # avoid division by 0
                return np.rint(np.sum(calibrated_prune_counts)).astype("int")
            else:  # not calibrated
                return self.prune_count

    def get_quality_scores(self) -> np.ndarray:
        if not self.store_results:
            raise ValueError(
                "Must initialize the LabelInspector with `store_results` == True. "
                "Otherwise you can assemble the label quality scores yourself based on "
                "the scores returned for each batch of data from `score_label_quality()`"
            )
        else:
            return np.asarray(self.label_quality_scores)

    def get_label_issues(self) -> np.ndarray:
        if not self.store_results:
            raise ValueError(
                "Must initialize the LabelInspector with `store_results` == True. "
                "Otherwise you can identify label issues yourself based on the scores from all "
                "the batches of data and the total number of issues returned by `get_num_issues()`"
            )
        if self.examples_processed_quality < 1:
            raise ValueError(
                "Have not evaluated any labels yet. Call `score_label_quality()` first."
            )
        if self.verbose:
            print(
                f"Total number of examples whose labels have been evaluated: {self.examples_processed_quality}"
            )
        return find_top_issues(self.get_quality_scores(), top=self.get_num_issues(silent=True))

    def update_confident_thresholds(self, labels: LabelLike, pred_probs: np.ndarray):
        labels = _batch_check(labels, pred_probs, self.num_class)
        batch_size = len(labels)
        batch_thresholds = get_confident_thresholds(
            labels, pred_probs
        )  # values for missing classes may exceed 1 but should not matter since we multiply by this class counts in the batch
        batch_class_counts = value_counts_fill_missing_classes(labels, num_classes=self.num_class)
        self.confident_thresholds = (
            self.examples_per_class * self.confident_thresholds
            + batch_class_counts * batch_thresholds
        ) / np.clip(
            self.examples_per_class + batch_class_counts, a_min=1, a_max=None
        )  # avoid division by 0
        self.confident_thresholds = np.clip(
            self.confident_thresholds, a_min=CONFIDENT_THRESHOLDS_LOWER_BOUND, a_max=None
        )
        self.examples_per_class += batch_class_counts
        self.examples_processed_thresh += batch_size

    def score_label_quality(
        self,
        labels: LabelLike,
        pred_probs: np.ndarray,
        *,
        update_num_issues: bool = True,
    ) -> np.ndarray:
        
        labels = _batch_check(labels, pred_probs, self.num_class)
        batch_size = len(labels)
        scores = _compute_label_quality_scores(
            labels,
            pred_probs,
            confident_thresholds=self.get_confident_thresholds(silent=True),
            **self.quality_score_kwargs,
        )
        class_counts = value_counts_fill_missing_classes(labels, num_classes=self.num_class)
        if update_num_issues:
            self._update_num_label_issues(labels, pred_probs, **self.num_issue_kwargs)
        self.examples_processed_quality += batch_size
        if self.store_results:
            self.label_quality_scores += list(scores)

        return scores

    def _update_num_label_issues(
        self,
        labels: LabelLike,
        pred_probs: np.ndarray,
        **kwargs,
    ):
        

        # whether to match the output of count.num_label_issues exactly
        # default is False, which gives significant speedup on large batches
        # and empirically matches num_label_issues even on input sizes of
        # 1M x 10k
        thorough = False
        if self.examples_processed_thresh < 1:
            raise ValueError(
                "Have not computed any confident_thresholds yet. Call `update_confident_thresholds()` first."
            )

        if self.n_jobs == 1:
            adj_confident_thresholds = self.confident_thresholds - FLOATING_POINT_COMPARISON
            pred_class = np.argmax(pred_probs, axis=1)
            batch_size = len(labels)
            if thorough:
                # add margin for floating point comparison operations:
                pred_gt_thresholds = pred_probs >= adj_confident_thresholds
                max_ind = np.argmax(pred_probs * pred_gt_thresholds, axis=1)
                if not self.off_diagonal_calibrated:
                    mask = (max_ind != labels) & (pred_class != labels)
                else:
                    # calibrated
                    # should we change to above?
                    mask = pred_class != labels
            else:
                max_ind = pred_class
                mask = pred_class != labels

            if not self.off_diagonal_calibrated:
                prune_count_batch = np.sum(
                    (
                        pred_probs[np.arange(batch_size), max_ind]
                        >= adj_confident_thresholds[max_ind]
                    )
                    & mask
                )
                self.prune_count += prune_count_batch
            else:  # calibrated
                self.class_counts += value_counts_fill_missing_classes(
                    labels, num_classes=self.num_class
                )
                to_increment = (
                    pred_probs[np.arange(batch_size), max_ind] >= adj_confident_thresholds[max_ind]
                )
                for class_label in range(self.num_class):
                    labels_equal_to_class = labels == class_label
                    self.normalization[class_label] += np.sum(labels_equal_to_class & to_increment)
                    self.prune_counts[class_label] += np.sum(
                        labels_equal_to_class
                        & to_increment
                        & (max_ind != labels)
                        # & (pred_class != labels)
                        # This is not applied in num_label_issues(..., estimation_method="off_diagonal_custom"). Do we want to add it?
                    )
        else:  # multiprocessing implementation
            global adj_confident_thresholds_shared
            adj_confident_thresholds_shared = self.confident_thresholds - FLOATING_POINT_COMPARISON

            global labels_shared, pred_probs_shared
            labels_shared = labels
            pred_probs_shared = pred_probs

            # good values for this are ~1000-10000 in benchmarks where pred_probs has 1B entries:
            processes = 5000
            if len(labels) <= processes:
                chunksize = 1
            else:
                chunksize = len(labels) // processes
            inds = split_arr(np.arange(len(labels)), chunksize)

            if thorough:
                use_thorough = np.ones(len(inds), dtype=bool)
            else:
                use_thorough = np.zeros(len(inds), dtype=bool)
            args = zip(inds, use_thorough)

            # Use fork method explicitly for Python 3.14+ compatibility
            # Falls back to default method if fork is not available
            try:
                ctx = mp.get_context("fork")
                pool_class = ctx.Pool
            except (RuntimeError, ValueError):
                # fork not available (Windows) or already set, use default
                pool_class = mp.Pool

            with pool_class(self.n_jobs) as pool:
                if not self.off_diagonal_calibrated:
                    prune_count_batch = np.sum(
                        np.asarray(list(pool.imap_unordered(_compute_num_issues, args)))
                    )
                    self.prune_count += prune_count_batch
                else:
                    results = list(pool.imap_unordered(_compute_num_issues_calibrated, args))
                    for result in results:
                        class_label = result[0]
                        self.class_counts[class_label] += 1
                        self.normalization[class_label] += result[1]
                        self.prune_counts[class_label] += result[2]

def split_arr(arr: np.ndarray, chunksize: int) -> List[np.ndarray]:
    return np.split(arr, np.arange(chunksize, arr.shape[0], chunksize), axis=0)

def _compute_num_issues(arg: Tuple[np.ndarray, bool]) -> int:
    ind = arg[0]
    thorough = arg[1]
    label = labels_shared[ind]
    pred_prob = pred_probs_shared[ind, :]
    pred_class = np.argmax(pred_prob, axis=-1)
    batch_size = len(label)

    if thorough:
        pred_gt_thresholds = pred_prob >= adj_confident_thresholds_shared
        max_ind = np.argmax(pred_prob * pred_gt_thresholds, axis=-1)
        prune_count_batch = np.sum(
            (pred_prob[np.arange(batch_size), max_ind] >= adj_confident_thresholds_shared[max_ind])
            & (max_ind != label)
            & (pred_class != label)
        )
    else:
        prune_count_batch = np.sum(
            (
                pred_prob[np.arange(batch_size), pred_class]
                >= adj_confident_thresholds_shared[pred_class]
            )
            & (pred_class != label)
        )
    return prune_count_batch

def _compute_num_issues_calibrated(arg: Tuple[np.ndarray, bool]) -> Tuple[Any, int, int]:
    ind = arg[0]
    thorough = arg[1]
    label = labels_shared[ind]
    pred_prob = pred_probs_shared[ind, :]
    batch_size = len(label)

    pred_class = np.argmax(pred_prob, axis=-1)
    if thorough:
        pred_gt_thresholds = pred_prob >= adj_confident_thresholds_shared
        max_ind = np.argmax(pred_prob * pred_gt_thresholds, axis=-1)
        to_inc = (
            pred_prob[np.arange(batch_size), max_ind] >= adj_confident_thresholds_shared[max_ind]
        )

        prune_count_batch = to_inc & (max_ind != label)
        normalization_batch = to_inc
    else:
        to_inc = (
            pred_prob[np.arange(batch_size), pred_class]
            >= adj_confident_thresholds_shared[pred_class]
        )
        normalization_batch = to_inc
        prune_count_batch = to_inc & (pred_class != label)

    return (label, normalization_batch, prune_count_batch)

def _batch_check(labels: LabelLike, pred_probs: np.ndarray, num_class: int) -> np.ndarray:
    batch_size = pred_probs.shape[0]
    labels = np.asarray(labels)
    if len(labels) != batch_size:
        raise ValueError("labels and pred_probs must have same length")
    if pred_probs.shape[1] != num_class:
        raise ValueError("num_class must equal pred_probs.shape[1]")

    return labels
