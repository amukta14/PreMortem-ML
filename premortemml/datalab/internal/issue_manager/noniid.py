from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Optional, Union, cast
import itertools

from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from premortemml.datalab.internal.issue_manager import IssueManager
from premortemml.datalab.internal.issue_manager.knn_graph_helpers import knn_exists, set_knn_graph

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from premortemml.datalab.datalab import Datalab

def simplified_kolmogorov_smirnov_test(
    neighbor_histogram: npt.NDArray[np.float64],
    non_neighbor_histogram: npt.NDArray[np.float64],
) -> float:
    

    neighbor_cdf = np.cumsum(neighbor_histogram)
    non_neighbor_cdf = np.cumsum(non_neighbor_histogram)

    statistic = np.max(np.abs(neighbor_cdf - non_neighbor_cdf))
    return statistic

class NonIIDIssueManager(IssueManager):

    description: ClassVar[
        str
    ] = ""
    issue_name: ClassVar[str] = "non_iid"
    verbosity_levels = {
        0: ["p-value"],
        1: [],
        2: [],
    }

    def __init__(
        self,
        datalab: Datalab,
        metric: Optional[Union[str, Callable]] = None,
        k: int = 10,
        num_permutations: int = 25,
        seed: Optional[int] = 0,
        significance_threshold: float = 0.05,
        **_,
    ):
        super().__init__(datalab)
        self.metric = metric
        self.k = k
        self.num_permutations = num_permutations
        self.tests = {
            "ks": simplified_kolmogorov_smirnov_test,
        }
        self.background_distribution = None
        self.seed = seed
        self.significance_threshold = significance_threshold

        self._skip_storing_knn_graph_for_pred_probs: bool = False

    @staticmethod
    def _determine_optional_features(
        features: Optional[npt.NDArray],
        pred_probs: Optional[np.ndarray],
    ) -> Optional[npt.NDArray]:
        
        if features is not None:
            return features

        if pred_probs is not None:
            return pred_probs

        return None

    def find_issues(
        self,
        features: Optional[npt.NDArray] = None,
        pred_probs: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        statistics = self.datalab.get_info("statistics")

        # Crucial when building knn graphs with pred_probs instead of features, where only the
        # latter is preferred for storage.
        self._determine_if_knn_graph_storage_should_be_skipped(
            features, pred_probs, kwargs, statistics, self.k
        )

        knn_graph, self.metric, _ = set_knn_graph(
            features=self._determine_optional_features(features, pred_probs),
            find_issues_kwargs=kwargs,
            metric=self.metric,
            k=self.k,
            statistics=statistics,
        )

        self.neighbor_index_choices = self._get_neighbors(knn_graph=knn_graph)

        self.num_neighbors = self.k

        indices = np.arange(self.N)
        self.neighbor_index_distances = np.abs(indices.reshape(-1, 1) - self.neighbor_index_choices)

        self.statistics = self._get_statistics(self.neighbor_index_distances)

        self.p_value = self._permutation_test(num_permutations=self.num_permutations)

        scores = self._score_dataset()
        issue_mask = np.zeros(self.N, dtype=bool)
        if self.p_value < self.significance_threshold:
            issue_mask[scores.argmin()] = True
        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": issue_mask,
                self.issue_score_key: scores,
            },
        )

        self.summary = self.make_summary(score=self.p_value)

        self.info = self.collect_info(knn_graph=knn_graph)

    def _determine_if_knn_graph_storage_should_be_skipped(
        self, features, pred_probs, kwargs, statistics, k
    ) -> None:
        
        sufficient_knn_graph_available = knn_exists(kwargs, statistics, k)
        pred_probs_needed = (
            not sufficient_knn_graph_available and features is None and pred_probs is not None
        )
        if pred_probs_needed:
            self._skip_storing_knn_graph_for_pred_probs = True

    def collect_info(self, knn_graph: csr_matrix) -> dict:
        issues_dict = {
            "p-value": self.p_value,
        }

        params_dict = {
            "metric": self.metric,
            "k": self.k,
        }

        statistics_dict = self._build_statistics_dictionary(knn_graph=knn_graph)

        info_dict = {
            **issues_dict,
            **params_dict,  # type: ignore[arg-type]
            **statistics_dict,  # type: ignore[arg-type]
        }
        return info_dict

    def _build_statistics_dictionary(self, knn_graph: csr_matrix) -> Dict[str, Dict[str, Any]]:
        statistics_dict: Dict[str, Dict[str, Any]] = {"statistics": {}}

        if self._skip_storing_knn_graph_for_pred_probs:
            return statistics_dict
        # Add the knn graph as a statistic if necessary
        graph_key = "weighted_knn_graph"
        old_knn_graph = self.datalab.get_info("statistics").get(graph_key, None)
        old_graph_exists = old_knn_graph is not None
        prefer_new_graph = (
            (knn_graph is not None and not old_graph_exists)
            or (old_knn_graph is not None and knn_graph.nnz > old_knn_graph.nnz)
            or self.metric != self.datalab.get_info("statistics").get("knn_metric", None)
        )
        if prefer_new_graph:
            statistics_dict["statistics"][graph_key] = knn_graph
            if self.metric is not None:
                statistics_dict["statistics"]["knn_metric"] = self.metric

        return statistics_dict

    def _permutation_test(self, num_permutations) -> float:
        N = self.N

        if self.seed is not None:
            np.random.seed(self.seed)
        perms = np.fromiter(
            itertools.chain.from_iterable(
                np.random.permutation(N) for i in range(num_permutations)
            ),
            dtype=int,
        ).reshape(num_permutations, N)

        neighbor_index_choices = self.neighbor_index_choices
        neighbor_index_choices = neighbor_index_choices.reshape(1, *neighbor_index_choices.shape)
        perm_neighbor_choices = perms[:, neighbor_index_choices].reshape(
            num_permutations, *neighbor_index_choices.shape[1:]
        )
        neighbor_index_distances = np.abs(perms[..., None] - perm_neighbor_choices).reshape(
            num_permutations, -1
        )

        statistics = []
        for neighbor_index_dist in neighbor_index_distances:
            stats = self._get_statistics(
                neighbor_index_dist,
            )
            statistics.append(stats)

        ks_stats = np.array([stats["ks"] for stats in statistics])
        ks_stats_kde = gaussian_kde(ks_stats)
        p_value = ks_stats_kde.integrate_box(self.statistics["ks"], 100)

        return p_value

    def _score_dataset(self) -> npt.NDArray[np.float64]:
        N = self.N

        sorted_neighbors = np.sort(self.neighbor_index_distances, axis=1)

        # find the maximum distance that occurs with double probability
        middle_idx = np.floor((N - 1) / 2).astype(int)
        double_distances = np.arange(N).reshape(N, 1)
        double_distances[double_distances > middle_idx] -= N - 1
        double_distances = np.abs(double_distances)

        sorted_neighbors = np.hstack([sorted_neighbors, np.ones((N, 1)) * (N - 1)]).astype(int)

        # the set of distances that are less than the double distance threshold
        set_beginning = sorted_neighbors <= double_distances
        # the set of distances that are greater than the double distance threshold but have nonzero probability
        set_middle = (sorted_neighbors > double_distances) & (
            sorted_neighbors <= (N - double_distances - 1)
        )
        # the set of distances that occur with 0 probability
        set_end = sorted_neighbors > (N - double_distances - 1)

        shifted_neighbors = np.zeros(sorted_neighbors.shape)
        shifted_neighbors[:, 1:] = sorted_neighbors[:, :-1]
        diffs = sorted_neighbors - shifted_neighbors  # the distances between the sorted indices

        area_beginning = (double_distances**2) / (N - 1)
        length = N - 2 * double_distances - 1
        a = 2 * double_distances / (N - 1)
        area_middle = 0.5 * (a + 1) * length

        # compute the area under the CDF for each of the indices in sorted_neighbors
        background_area = np.zeros(diffs.shape)
        background_diffs = np.zeros(diffs.shape)
        background_area[set_beginning] = ((sorted_neighbors**2) / (N - 1))[set_beginning]
        background_area[set_middle] = (
            area_beginning
            + 0.5
            * (
                (sorted_neighbors + 3 * double_distances)
                * (sorted_neighbors - double_distances)
                / (N - 1)
            )
        )[set_middle]
        background_area[set_end] = (
            area_beginning + area_middle + (sorted_neighbors - (N - double_distances - 1) * 1.0)
        )[set_end]

        # compute the area under the CDF between indices in sorted_neighbors
        shifted_background = np.zeros(background_area.shape)
        shifted_background[:, 1:] = background_area[:, :-1]
        background_diffs = background_area - shifted_background

        # compute the foreground CDF and AUC between indices in sorted_neighbors
        foreground_cdf = np.arange(sorted_neighbors.shape[1]) / (sorted_neighbors.shape[1] - 1)
        foreground_diffs = foreground_cdf.reshape(1, -1) * diffs

        # compute the differences between foreground and background area intervals
        area_diffs = np.abs(foreground_diffs - background_diffs)
        stats = np.sum(area_diffs, axis=1)

        # normalize scores by the index and transform to [0, 1]
        indices = np.arange(N)
        reverse = N - indices
        normalizer = np.where(indices > reverse, indices, reverse)

        scores = stats / normalizer
        scores = np.tanh(-1 * scores) + 1
        return scores

    def _get_neighbors(self, knn_graph: csr_matrix) -> np.ndarray:
        self.N = knn_graph.shape[0]
        kneighbors = knn_graph.indices.reshape(self.N, -1)
        return kneighbors

    def _get_statistics(
        self,
        neighbor_index_distances,
    ) -> dict[str, float]:
        neighbor_index_distances = neighbor_index_distances.flatten()
        sorted_neighbors = np.sort(neighbor_index_distances)
        sorted_neighbors = np.hstack([sorted_neighbors, np.ones((1)) * (self.N - 1)]).astype(int)

        if self.background_distribution is None:
            self.background_distribution = (self.N - np.arange(1, self.N)) / (
                self.N * (self.N - 1) / 2
            )

        background_distribution = cast(np.ndarray, self.background_distribution)
        background_cdf = np.cumsum(background_distribution)

        foreground_cdf = np.arange(sorted_neighbors.shape[0]) / (sorted_neighbors.shape[0] - 1)

        statistic = np.max(np.abs(foreground_cdf - background_cdf[sorted_neighbors - 1]))
        statistics = {"ks": statistic}
        return statistics
