import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors

from premortemml.count import get_confident_thresholds
from premortemml.internal.label_quality_utils import (
    _subtract_confident_thresholds,
    get_normalized_entropy,
)
from premortemml.internal.neighbor.knn_graph import correct_knn_distances_and_indices, features_to_knn
from premortemml.internal.numerics import softmax
from premortemml.internal.outlier import correct_precision_errors, transform_distances_to_scores
from premortemml.internal.validation import assert_valid_inputs, labels_to_array
from premortemml.typing import LabelLike

class OutOfDistribution:

    OUTLIER_PARAMS = {"k", "t", "knn"}
    OOD_PARAMS = {"confident_thresholds", "adjust_pred_probs", "method", "M", "gamma"}
    DEFAULT_PARAM_DICT: Dict[str, Union[str, int, float, None, np.ndarray]] = {
        "k": None,  # param for feature based outlier detection (number of neighbors)
        "t": 1,  # param for feature based outlier detection (controls transformation of outlier scores to 0-1 range)
        "knn": None,  # param for features based outlier detection (precomputed nearest neighbors graph to use)
        "method": "entropy",  # param specifying which pred_probs-based outlier detection method to use
        "adjust_pred_probs": True,  # param for pred_probs based outlier detection (whether to adjust the probabilities by class thresholds or not)
        "confident_thresholds": None,  # param for pred_probs based outlier detection (precomputed confident thresholds to use for adjustment)
        "M": 100,  # param for GEN method for pred_probs based outlier detection
        "gamma": 0.1,  # param for GEN method for pred_probs based outlier detection
    }

    def __init__(self, params: Optional[dict] = None) -> None:
        self._assert_valid_params(params, self.DEFAULT_PARAM_DICT)
        self.params = self.DEFAULT_PARAM_DICT.copy()
        if params is not None:
            self.params.update(params)
        if self.params["adjust_pred_probs"] and self.params["method"] == "gen":
            print(
                "CAUTION: GEN method is not recommended for use with adjusted pred_probs. "
                "To use GEN, we recommend setting: params['adjust_pred_probs'] = False"
            )

        # scaling_factor internally used to rescale distances based on mean distances to k nearest neighbors
        self.params["scaling_factor"] = None

    def fit_score(
        self,
        *,
        features: Optional[np.ndarray] = None,
        pred_probs: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> np.ndarray:
        
        scores = self._shared_fit(
            features=features,
            pred_probs=pred_probs,
            labels=labels,
            verbose=verbose,
        )

        if scores is None:  # Fit was called on already fitted object so we just score vals instead
            scores = self.score(features=features, pred_probs=pred_probs)

        return scores

    def fit(
        self,
        *,
        features: Optional[np.ndarray] = None,
        pred_probs: Optional[np.ndarray] = None,
        labels: Optional[LabelLike] = None,
        verbose: bool = True,
    ):
        
        _ = self._shared_fit(
            features=features,
            pred_probs=pred_probs,
            labels=labels,
            verbose=verbose,
        )

    def score(
        self, *, features: Optional[np.ndarray] = None, pred_probs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        
        self._assert_valid_inputs(features, pred_probs)

        if features is not None:
            if self.params["knn"] is None:
                raise ValueError(
                    "OOD estimator needs to be fit on features first. Call `fit()` or `fit_scores()` before this function."
                )
            scores, _ = self._get_ood_features_scores(
                features, **self._get_params(self.OUTLIER_PARAMS)
            )

        if pred_probs is not None:
            if self.params["confident_thresholds"] is None and self.params["adjust_pred_probs"]:
                raise ValueError(
                    "OOD estimator needs to be fit on pred_probs first since params['adjust_pred_probs']=True. Call `fit()` or `fit_scores()` before this function."
                )
            scores, _ = _get_ood_predictions_scores(pred_probs, **self._get_params(self.OOD_PARAMS))

        return scores

    def _get_params(self, param_keys) -> dict:
        return {k: v for k, v in self.params.items() if k in param_keys}

    @staticmethod
    def _assert_valid_params(params, param_keys):
        if params is not None:
            wrong_params = list(set(params.keys()).difference(set(param_keys)))
            if len(wrong_params) > 0:
                raise ValueError(
                    f"Passed in params dict can only contain {param_keys}. Remove {wrong_params} from params dict."
                )

    @staticmethod
    def _assert_valid_inputs(features, pred_probs):
        if features is None and pred_probs is None:
            raise ValueError(
                "Not enough information to compute scores. Pass in either features or pred_probs."
            )

        if features is not None and pred_probs is not None:
            raise ValueError(
                "Cannot fit to OOD Estimator to both features and pred_probs. Pass in either one or the other."
            )

        if features is not None and len(features.shape) != 2:
            raise ValueError(
                "Feature array needs to be of shape (N, M), where N is the number of examples and M is the "
                "number of features used to represent each example. "
            )

    def _shared_fit(
        self,
        *,
        features: Optional[np.ndarray] = None,
        pred_probs: Optional[np.ndarray] = None,
        labels: Optional[LabelLike] = None,
        verbose: bool = True,
    ) -> Optional[np.ndarray]:
        
        self._assert_valid_inputs(features, pred_probs)
        scores = None  # If none scores are returned, fit was skipped

        if features is not None:
            if self.params["knn"] is not None:
                # No fitting twice if knn object already fit
                warnings.warn(
                    "A KNN estimator has previously already been fit, call score() to apply it to data, or create a new OutOfDistribution object to fit a different estimator.",
                    UserWarning,
                )
            else:
                # Get ood features scores
                if verbose:
                    print("Fitting OOD estimator based on provided features ...")
                scores, knn = self._get_ood_features_scores(
                    features, **self._get_params(self.OUTLIER_PARAMS)
                )
                self.params["knn"] = knn

        if pred_probs is not None:
            if self.params["confident_thresholds"] is not None:
                # No fitting twice if confident_thresholds object already fit
                warnings.warn(
                    "Confident thresholds have previously already been fit, call score() to apply them to data, or create a new OutOfDistribution object to fit a different estimator.",
                    UserWarning,
                )
            else:
                # Get ood predictions scores
                if verbose:
                    print("Fitting OOD estimator based on provided pred_probs ...")
                scores, confident_thresholds = _get_ood_predictions_scores(
                    pred_probs,
                    labels=labels,
                    **self._get_params(self.OOD_PARAMS),
                )
                if confident_thresholds is None:
                    warnings.warn(
                        "No estimates need to be be fit under the provided params, so you could directly call "
                        "score() as an alternative.",
                        UserWarning,
                    )
                else:
                    self.params["confident_thresholds"] = confident_thresholds
        return scores

    def _get_ood_features_scores(
        self,
        features: Optional[np.ndarray] = None,
        knn: Optional[NearestNeighbors] = None,
        k: Optional[int] = None,
        t: int = 1,
    ) -> Tuple[np.ndarray, Optional[NearestNeighbors]]:
        
        DEFAULT_K = 10
        # fit skip over (if knn is not None) then skipping fit and suggest score else fit.
        distance_metric = None
        correct_knn = False
        if knn is None:  # setup default KNN estimator
            # Make sure both knn and features are not None
            knn = features_to_knn(features, n_neighbors=k)
            correct_knn = True
            features = None  # features should be None in knn.kneighbors(features) to avoid counting duplicate data points
            # Log knn metric as string to ensure compatibility for score correction
            distance_metric = (
                metric if isinstance((metric := knn.metric), str) else str(metric.__name__)
            )
            k = knn.n_neighbors

        elif k is None:
            k = knn.n_neighbors

        max_k = knn.n_neighbors  # number of neighbors previously used in NearestNeighbors object
        if k > max_k:  # if k provided is too high, use max possible number of nearest neighbors
            warnings.warn(
                f"Chosen k={k} cannot be greater than n_neighbors={max_k} which was used when fitting "
                f"NearestNeighbors object! Value of k changed to k={max_k}.",
                UserWarning,
            )
            k = max_k

        # Fit knn estimator on the features if a non-fitted estimator is passed in
        try:
            knn.kneighbors(features)
        except NotFittedError:
            knn.fit(features)

        # Get distances to k-nearest neighbors Note that the knn object contains the specification of distance metric
        # and n_neighbors (k value) If our query set of features matches the training set used to fit knn, the nearest
        # neighbor of each point is the point itself, at a distance of zero.
        distances, indices = knn.kneighbors(features)
        if (
            correct_knn
        ):  # This should only happen if knn is None at the start of this function. Will NEVER happen for approximate KNN provided by user.
            _features_for_correction = (
                knn._fit_X if features is None else features
            )  # Hacky way to get features (training or test). Storing np.unique results is a hassle. ONLY WORKS WITH sklearn NearestNeighbors object
            distances, _ = correct_knn_distances_and_indices(
                features=_features_for_correction,
                distances=distances,
                indices=indices,
            )

        # Calculate average distance to k-nearest neighbors
        avg_knn_distances = distances[:, :k].mean(axis=1)

        if self.params["scaling_factor"] is None:
            self.params["scaling_factor"] = float(
                max(np.median(avg_knn_distances), 100 * np.finfo(np.float64).eps)
            )
        scaling_factor = self.params["scaling_factor"]

        if not isinstance(scaling_factor, float):
            raise ValueError(f"Scaling factor must be a float. Got {type(scaling_factor)} instead.")

        ood_features_scores = transform_distances_to_scores(
            avg_knn_distances, t, scaling_factor=scaling_factor
        )
        distance_metric = distance_metric or (
            metric if isinstance((metric := knn.metric), str) else metric.__name__
        )
        p = None
        if distance_metric == "minkowski":
            p = knn.p
        ood_features_scores = correct_precision_errors(
            ood_features_scores, avg_knn_distances, distance_metric, p=p
        )
        return (ood_features_scores, knn)

def _get_ood_predictions_scores(
    pred_probs: np.ndarray,
    *,
    labels: Optional[LabelLike] = None,
    confident_thresholds: Optional[np.ndarray] = None,
    adjust_pred_probs: bool = True,
    method: str = "entropy",
    M: int = 100,
    gamma: float = 0.1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    
    valid_methods = (
        "entropy",
        "least_confidence",
        "gen",
    )

    if (confident_thresholds is not None or labels is not None) and not adjust_pred_probs:
        warnings.warn(
            "OOD scores are not adjusted with confident thresholds. If scores need to be adjusted set "
            "params['adjusted_pred_probs'] = True. Otherwise passing in confident_thresholds and/or labels does not change "
            "score calculation.",
            UserWarning,
        )

    if adjust_pred_probs:
        if confident_thresholds is None:
            if labels is None:
                raise ValueError(
                    "Cannot calculate adjust_pred_probs without labels. Either pass in labels parameter or set "
                    "params['adjusted_pred_probs'] = False. "
                )
            labels = labels_to_array(labels)
            assert_valid_inputs(X=None, y=labels, pred_probs=pred_probs, multi_label=False)
            confident_thresholds = get_confident_thresholds(labels, pred_probs, multi_label=False)

        pred_probs = _subtract_confident_thresholds(
            None, pred_probs, multi_label=False, confident_thresholds=confident_thresholds
        )

    # Scores are flipped so ood scores are closer to 0. Scores reflect confidence example is in-distribution.
    if method == "entropy":
        ood_predictions_scores = 1.0 - get_normalized_entropy(pred_probs)
    elif method == "least_confidence":
        ood_predictions_scores = pred_probs.max(axis=1)
    elif method == "gen":
        if pred_probs.shape[1] < M:  # pragma: no cover
            warnings.warn(
                f"GEN with the default hyperparameter settings is intended for datasets with at least {M} classes. You can adjust params['M'] according to the number of classes in your dataset.",
                UserWarning,
            )
        probs = softmax(pred_probs, axis=1)
        probs_sorted = np.sort(probs, axis=1)[:, -M:]
        ood_predictions_scores = (
            1 - np.sum(probs_sorted**gamma * (1 - probs_sorted) ** (gamma), axis=1) / M
        )  # Use 1 + original gen score/M to make the scores lie in 0-1
    else:
        raise ValueError(
            f
        )

    return (
        ood_predictions_scores,
        confident_thresholds,
    )
