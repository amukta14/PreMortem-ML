from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union
import numpy as np

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from premortemml.datalab.internal.data import Data
    from premortemml.datalab.internal.issue_manager import IssueManager
    from cleanvision import Imagelab

class _InfoStrategy(ABC):

    @staticmethod
    @abstractmethod
    def get_info(
        data: Data,
        info: Dict[str, Dict[str, Any]],
        issue_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        
        pass  # pragma: no cover

    @staticmethod
    def _get_info_helper(
        info: Dict[str, Dict[str, Any]],
        issue_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if issue_name is None:
            return None
        if issue_name not in info:
            raise ValueError(
                f"issue_name {issue_name} not found in self.info. These have not been computed yet."
            )
        info = info[issue_name].copy()
        return info

class _ClassificationInfoStrategy(_InfoStrategy):

    @staticmethod
    def get_info(
        data: Data,
        info: Dict[str, Dict[str, Any]],
        issue_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        info_extracted = _InfoStrategy._get_info_helper(info=info, issue_name=issue_name)
        info = info_extracted if info_extracted is not None else info
        if issue_name in ["label", "class_imbalance"]:
            if data.labels.is_available is False:
                raise ValueError(
                    "The labels are not available. "
                    "Most likely, no label column was provided when creating the Data object."
                )
            # Labels that are stored as integers may need to be converted to strings.
            label_map = data.labels.label_map
            if not label_map:
                raise ValueError("The label map is not available.")
            for key in ["given_label", "predicted_label"]:
                labels = info.get(key, None)
                if labels is not None:
                    info[key] = np.vectorize(label_map.get)(labels)
            info["class_names"] = list(label_map.values())  # type: ignore
        return info

class _RegressionInfoStrategy(_InfoStrategy):

    @staticmethod
    def get_info(
        data: Data,
        info: Dict[str, Dict[str, Any]],
        issue_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        info_extracted = _InfoStrategy._get_info_helper(info=info, issue_name=issue_name)
        info = info_extracted if info_extracted is not None else info
        if issue_name == "label":
            for key in ["given_label", "predicted_label"]:
                labels = info.get(key, None)
                if labels is not None:
                    info[key] = labels
        return info

class _MultilabelInfoStrategy(_InfoStrategy):

    @staticmethod
    def get_info(
        data: Data,
        info: Dict[str, Dict[str, Any]],
        issue_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        info_extracted = _InfoStrategy._get_info_helper(info=info, issue_name=issue_name)
        info = info_extracted if info_extracted is not None else info
        if issue_name == "label":
            if data.labels.is_available is False:
                raise ValueError(
                    "The labels are not available. "
                    "Most likely, no label column was provided when creating the Data object."
                )
            # Labels that are stored as integers may need to be converted to strings.
            label_map = data.labels.label_map
            if not label_map:
                raise ValueError("The label map is not available.")
            for key in ["given_label", "predicted_label"]:
                labels = info.get(key, None)
                if labels is not None:
                    info[key] = [list(map(label_map.get, label)) for label in labels]  # type: ignore
            info["class_names"] = list(label_map.values())  # type: ignore
        return info

class DataIssues:

    def __init__(self, data: Data, strategy: Type[_InfoStrategy]) -> None:
        self.issues: pd.DataFrame = pd.DataFrame(index=range(len(data)))
        self.issue_summary: pd.DataFrame = pd.DataFrame(
            columns=["issue_type", "score", "num_issues"]
        ).astype({"score": np.float64, "num_issues": np.int64})
        self.info: Dict[str, Dict[str, Any]] = {
            "statistics": get_data_statistics(data),
        }
        self._data = data
        self._strategy = strategy

    def get_info(self, issue_name: Optional[str] = None) -> Dict[str, Any]:
        return self._strategy.get_info(data=self._data, info=self.info, issue_name=issue_name)

    @property
    def statistics(self) -> Dict[str, Any]:
        return self.info["statistics"]

    def get_issues(self, issue_name: Optional[str] = None) -> pd.DataFrame:
        if self.issues.empty:
            raise ValueError(
                
            )
        if issue_name is None:
            return self.issues

        columns = [col for col in self.issues.columns if issue_name in col]
        if not columns:
            raise ValueError(
                f
            )
        specific_issues = self.issues[columns]
        info = self.get_info(issue_name=issue_name)

        if issue_name == "label":
            specific_issues = specific_issues.assign(
                given_label=info["given_label"], predicted_label=info["predicted_label"]
            )

        if issue_name == "near_duplicate":
            column_dict = {
                k: info.get(k)
                for k in ["near_duplicate_sets", "distance_to_nearest_neighbor"]
                if info.get(k) is not None
            }
            specific_issues = specific_issues.assign(**column_dict)

        if issue_name == "class_imbalance":
            specific_issues = specific_issues.assign(given_label=info["given_label"])
        return specific_issues

    def get_issue_summary(self, issue_name: Optional[str] = None) -> pd.DataFrame:
        if self.issue_summary.empty:
            raise ValueError(
                "No issues found in the dataset. "
                "Call `find_issues` before calling `get_issue_summary`."
            )

        if issue_name is None:
            return self.issue_summary

        row_mask = self.issue_summary["issue_type"] == issue_name
        if not any(row_mask):
            raise ValueError(f"Issue type {issue_name} not found in the summary.")
        return self.issue_summary[row_mask].reset_index(drop=True)

    def collect_statistics(self, issue_manager: Union[IssueManager, "Imagelab"]) -> None:
        key = "statistics"
        statistics: Dict[str, Any] = issue_manager.info.get(key, {})
        if statistics:
            self.info[key].update(statistics)

    def _update_issues(self, issue_manager):
        overlapping_columns = list(set(self.issues.columns) & set(issue_manager.issues.columns))
        if overlapping_columns:
            warnings.warn(
                f"Overwriting columns {overlapping_columns} in self.issues with "
                f"columns from issue manager {issue_manager}."
            )
            self.issues.drop(columns=overlapping_columns, inplace=True)
        self.issues = self.issues.join(issue_manager.issues, how="outer")

    def _update_issue_info(self, issue_name, new_info):
        if issue_name in self.info:
            warnings.warn(f"Overwriting key {issue_name} in self.info")
        self.info[issue_name] = new_info

    def collect_issues_from_issue_manager(self, issue_manager: IssueManager) -> None:
        self._update_issues(issue_manager)

        if issue_manager.issue_name in self.issue_summary["issue_type"].values:
            warnings.warn(
                f"Overwriting row in self.issue_summary with "
                f"row from issue manager {issue_manager}."
            )
            self.issue_summary = self.issue_summary[
                self.issue_summary["issue_type"] != issue_manager.issue_name
            ]
        issue_column_name: str = f"is_{issue_manager.issue_name}_issue"
        num_issues: int = int(issue_manager.issues[issue_column_name].sum())
        self.issue_summary = pd.concat(
            [
                self.issue_summary,
                issue_manager.summary.assign(num_issues=num_issues),
            ],
            axis=0,
            ignore_index=True,
        )
        self._update_issue_info(issue_manager.issue_name, issue_manager.info)

    def collect_issues_from_imagelab(self, imagelab: "Imagelab", issue_types: List[str]) -> None:
        pass  # pragma: no cover

    def set_health_score(self) -> None:
        self.info["statistics"]["health_score"] = self.issue_summary["score"].mean()

def get_data_statistics(data: Data) -> Dict[str, Any]:
    statistics: Dict[str, Any] = {
        "num_examples": len(data),
        "multi_label": False,
        "health_score": None,
    }
    if data.labels.is_available:
        class_names = data.class_names
        statistics["class_names"] = class_names
        statistics["num_classes"] = len(class_names)
    return statistics
