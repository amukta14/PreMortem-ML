from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, List

import pandas as pd

from premortemml.datalab.internal.issue_manager import IssueManager
from premortemml.internal.multilabel_utils import onehot2int
from premortemml.multilabel_classification.filter import find_label_issues
from premortemml.multilabel_classification.rank import get_label_quality_scores

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    import pandas as pd

    from premortemml.datalab.datalab import Datalab

class MultilabelIssueManager(IssueManager):

    description: ClassVar[
        str
    ] = ""

    _PREDICTED_LABEL_THRESH = 0.5
    

    issue_name: ClassVar[str] = "label"
    verbosity_levels = {
        0: [],
        1: [],
        2: [],
        3: [],
    }

    def __init__(
        self,
        datalab: Datalab,
        **_,
    ):
        super().__init__(datalab)

    @staticmethod
    def _process_find_label_issues_kwargs(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
        accepted_kwargs = [
            "filter_by",
            "frac_noise",
            "num_to_remove_per_class",
            "min_examples_per_class",
            "confident_joint",
            "n_jobs",
            "verbose",
            "low_memory",
        ]
        return {k: v for k, v in kwargs.items() if k in accepted_kwargs and v is not None}

    @staticmethod
    def _process_get_label_quality_scores_kwargs(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
        accepted_kwargs = ["method", "adjust_pred_probs", "aggregator_kwargs"]
        return {k: v for k, v in kwargs.items() if k in accepted_kwargs and v is not None}

    def find_issues(
        self,
        pred_probs: npt.NDArray,
        **kwargs,
    ) -> None:
        
        predicted_labels = onehot2int(pred_probs > self._PREDICTED_LABEL_THRESH)

        # Find examples with label issues
        assert isinstance(self.datalab.labels, List)  # Type Narrowing
        is_issue_column = find_label_issues(
            labels=self.datalab.labels,
            pred_probs=pred_probs,
            **self._process_find_label_issues_kwargs(**kwargs),
        )
        scores = get_label_quality_scores(
            labels=self.datalab.labels,
            pred_probs=pred_probs,
            **self._process_get_label_quality_scores_kwargs(**kwargs),
        )

        self.issues = pd.DataFrame(
            {
                f"is_{self.issue_name}_issue": is_issue_column,
                self.issue_score_key: scores,
            },
        )
        # Get a summarized dataframe of the label issues
        self.summary = self.make_summary(score=scores.mean())

        # Collect info about the label issues
        self.info = self.collect_info(self.datalab.labels, predicted_labels)

    def collect_info(
        self, given_labels: List[List[int]], predicted_labels: List[List[int]]
    ) -> Dict[str, Any]:
        issues_info = {
            "given_label": given_labels,
            "predicted_label": predicted_labels,
        }
        return issues_info
