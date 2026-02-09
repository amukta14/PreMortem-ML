from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional
import numpy as np
import pandas as pd

from premortemml.regression.learn import CleanLearning
from premortemml.datalab.internal.issue_manager import IssueManager
from premortemml.regression.rank import get_label_quality_scores

if TYPE_CHECKING:  # pragma: no cover
    from premortemml.datalab.datalab import Datalab

class RegressionLabelIssueManager(IssueManager):

    description: ClassVar[
        str
    ] = ""

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
        clean_learning_kwargs: Optional[Dict[str, Any]] = None,
        threshold: float = 0.05,
        health_summary_parameters: Optional[Dict[str, Any]] = None,
        **_,
    ):
        super().__init__(datalab)
        self.cl = CleanLearning(**(clean_learning_kwargs or {}))
        # This is a field for prioritizing features only when using a custom model
        self._uses_custom_model = "model" in (clean_learning_kwargs or {})
        self.threshold = threshold

    def find_issues(
        self,
        features: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        
        if features is None and predictions is None:
            raise ValueError(
                "Regression requires numerical `features` or `predictions` "
                "to be passed in as an argument to `find_issues`."
            )
        if features is None and self._uses_custom_model:
            raise ValueError(
                "Regression requires numerical `features` to be passed in as an argument to `find_issues` "
                "when using a custom model."
            )
        # If features are provided and either a custom model is used or no predictions are provided
        use_features = features is not None and (self._uses_custom_model or predictions is None)
        labels = self.datalab.labels
        if not isinstance(labels, np.ndarray):
            error_msg = (
                f"Expected labels to be a numpy array of shape (n_samples,) to use with RegressionLabelIssueManager, "
                f"but got {type(labels)} instead."
            )
            raise TypeError(error_msg)
        if use_features:
            assert features is not None  # mypy won't narrow the type for some reason
            self.issues = find_issues_with_features(
                features=features,
                y=labels,
                cl=self.cl,
                **kwargs,  # function sanitizes kwargs
            )
            self.issues.rename(columns={"label_quality": self.issue_score_key}, inplace=True)

        # Otherwise, if predictions are provided, process them
        else:
            assert predictions is not None  # mypy won't narrow the type for some reason
            self.issues = find_issues_with_predictions(
                predictions=predictions,
                y=labels,
                **{**kwargs, **{"threshold": self.threshold}},  # function sanitizes kwargs
            )

        # Get a summarized dataframe of the label issues
        self.summary = self.make_summary(score=self.issues[self.issue_score_key].mean())

        # Collect info about the label issues
        self.info = self.collect_info(issues=self.issues)

        # Drop columns from issues that are in the info
        self.issues = self.issues.drop(columns=["given_label", "predicted_label"])

    def collect_info(self, issues: pd.DataFrame) -> dict:
        issues_info = {
            "num_label_issues": sum(issues[f"is_{self.issue_name}_issue"]),
            "average_label_quality": issues[self.issue_score_key].mean(),
            "given_label": issues["given_label"].tolist(),
            "predicted_label": issues["predicted_label"].tolist(),
        }

        # health_summary_info, cl_info kept just for consistency with classification, but it could be just return issues_info
        health_summary_info: dict = {}
        cl_info: dict = {}

        info_dict = {
            **issues_info,
            **health_summary_info,
            **cl_info,
        }

        return info_dict

def find_issues_with_predictions(
    predictions: np.ndarray,
    y: np.ndarray,
    threshold: float,
    **kwargs,
) -> pd.DataFrame:
    
    _accepted_kwargs = ["method"]
    _kwargs = {k: kwargs.get(k) for k in _accepted_kwargs}
    _kwargs = {k: v for k, v in _kwargs.items() if v is not None}
    quality_scores = get_label_quality_scores(labels=y, predictions=predictions, **_kwargs)

    median_score = np.median(quality_scores)
    is_label_issue_mask = quality_scores < median_score * threshold

    issues = pd.DataFrame(
        {
            "is_label_issue": is_label_issue_mask,
            "label_score": quality_scores,
            "given_label": y,
            "predicted_label": predictions,
        }
    )
    return issues

def find_issues_with_features(
    features: np.ndarray,
    y: np.ndarray,
    cl: CleanLearning,
    **kwargs,
) -> pd.DataFrame:
    
    _accepted_kwargs = [
        "uncertainty",
        "coarse_search_range",
        "fine_search_size",
        "save_space",
        "model_kwargs",
    ]
    _kwargs = {k: v for k, v in kwargs.items() if k in _accepted_kwargs and v is not None}
    return cl.find_label_issues(X=features, y=y, **_kwargs)
