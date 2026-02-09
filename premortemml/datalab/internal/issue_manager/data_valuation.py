from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from premortemml.data_valuation import data_shapley_knn
from premortemml.datalab.internal.issue_manager import IssueManager
from premortemml.datalab.internal.issue_manager.knn_graph_helpers import (
    num_neighbors_in_knn_graph,
    set_knn_graph,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    import pandas as pd
    from premortemml.datalab.datalab import Datalab

class DataValuationIssueManager(IssueManager):

    description: ClassVar[
        str
    ] = ""

    issue_name: ClassVar[str] = "data_valuation"
    issue_score_key: ClassVar[str]
    verbosity_levels: ClassVar[Dict[int, List[str]]] = {
        0: [],
        1: [],
        2: [],
        3: ["average_data_valuation"],
    }

    DEFAULT_THRESHOLD = 0.5

    def __init__(
        self,
        datalab: Datalab,
        metric: Optional[Union[str, Callable]] = None,
        threshold: Optional[float] = None,
        k: int = 10,
        **kwargs,
    ):
        super().__init__(datalab)
        self.metric = metric
        self.k = k
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD

    def find_issues(
        self,
        features: Optional[npt.NDArray] = None,
        **kwargs,
    ) -> None:
        
        labels = self.datalab.labels
        if not isinstance(labels, np.ndarray):
            error_msg = (
                f"Expected labels to be a numpy array of shape (n_samples,) to use with DataValuationIssueManager, "
                f"but got {type(labels)} instead."
            )
            raise TypeError(error_msg)

        knn_graph, self.metric, _ = set_knn_graph(
            features=features,
            find_issues_kwargs=kwargs,
            metric=self.metric,
            k=self.k,
            statistics=self.datalab.get_info("statistics"),
        )

        num_neighbors = num_neighbors_in_knn_graph(knn_graph)
        if self.k > num_neighbors:
            raise ValueError(
                f"The provided knn graph has {num_neighbors} neighbors, which is less than the required {self.k} neighbors. "
                "Please ensure that the knn graph you provide has at least as many neighbors as the required value of k."
            )

        scores = data_shapley_knn(labels, knn_graph=knn_graph, k=self.k)

        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": scores < self.threshold,
                self.issue_score_key: scores,
            },
        )
        self.summary = self.make_summary(score=scores.mean())

        self.info = self.collect_info(issues=self.issues, knn_graph=knn_graph)

    def collect_info(self, issues: pd.DataFrame, knn_graph: csr_matrix) -> dict:
        issues_info = {
            "num_low_valuation_issues": sum(issues[f"is_{self.issue_name}_issue"]),
            "average_data_valuation": issues[self.issue_score_key].mean(),
        }

        params_dict = {
            "metric": self.metric,
            "k": self.k,
            "threshold": self.threshold,
        }

        statistics_dict = self._build_statistics_dictionary(knn_graph=knn_graph)

        info_dict = {
            **issues_info,
            **params_dict,
            **statistics_dict,
        }

        return info_dict

    def _build_statistics_dictionary(self, knn_graph: csr_matrix) -> Dict[str, Dict[str, Any]]:
        statistics_dict: Dict[str, Dict[str, Any]] = {"statistics": {}}

        # Add the knn graph as a statistic if necessary
        graph_key = "weighted_knn_graph"
        old_knn_graph = self.datalab.get_info("statistics").get(graph_key, None)
        old_graph_exists = old_knn_graph is not None
        prefer_new_graph = (
            not old_graph_exists
            or (old_knn_graph is not None and knn_graph.nnz > old_knn_graph.nnz)
            or self.metric != self.datalab.get_info("statistics").get("knn_metric", None)
        )
        if prefer_new_graph:
            statistics_dict["statistics"][graph_key] = knn_graph
            if self.metric is not None:
                statistics_dict["statistics"]["knn_metric"] = self.metric

        return statistics_dict
