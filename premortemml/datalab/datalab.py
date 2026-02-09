from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from premortemml.datalab.internal.adapter.constants import DEFAULT_CLEANVISION_ISSUES
from premortemml.datalab.internal.adapter.imagelab import create_imagelab
from premortemml.datalab.internal.data import Data
from premortemml.datalab.internal.display import _Displayer
from premortemml.datalab.internal.helper_factory import (
    _DataIssuesBuilder,
    issue_finder_factory,
    report_factory,
)
from premortemml.datalab.internal.issue_manager_factory import (
    list_default_issue_types as _list_default_issue_types,
    list_possible_issue_types as _list_possible_issue_types,
)
from premortemml.datalab.internal.serialize import _Serializer
from premortemml.datalab.internal.task import Task

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from datasets.arrow_dataset import Dataset
    from scipy.sparse import csr_matrix

    DatasetLike = Union[Dataset, pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], str]

__all__ = ["Datalab"]

class Datalab:

    def __init__(
        self,
        data: "DatasetLike",
        task: str = "classification",
        label_name: Optional[str] = None,
        image_key: Optional[str] = None,
        verbosity: int = 1,
    ) -> None:
        # Assume continuous values of labels for regression task
        # Map labels to integers for classification task
        self.task = Task.from_str(task)
        self._data = Data(data, self.task, label_name)
        self.data = self._data._data
        self._labels = self._data.labels
        self._label_map = self._labels.label_map
        self.label_name = self._labels.label_name
        self._data_hash = self._data._data_hash
        import premortemml
        self.version = premortemml.__version__
        self.verbosity = verbosity
        self._imagelab = create_imagelab(dataset=self.data, image_key=image_key)

        # Create the builder for DataIssues
        builder = _DataIssuesBuilder(self._data)
        builder.set_imagelab(self._imagelab).set_task(self.task)
        self.data_issues = builder.build()

    def __repr__(self) -> str:
        return _Displayer(data_issues=self.data_issues, task=self.task).__repr__()

    def __str__(self) -> str:
        return _Displayer(data_issues=self.data_issues, task=self.task).__str__()

    @property
    def labels(self) -> Union[np.ndarray, List[List[int]]]:
        return self._labels.labels

    @property
    def has_labels(self) -> bool:
        return self._labels.is_available

    @property
    def class_names(self) -> List[str]:
        return self._labels.class_names

    def find_issues(
        self,
        *,
        pred_probs: Optional[np.ndarray] = None,
        features: Optional[npt.NDArray] = None,
        knn_graph: Optional[csr_matrix] = None,
        issue_types: Optional[Dict[str, Any]] = None,
    ) -> None:
        

        if issue_types is not None and not issue_types:
            warnings.warn(
                "No issue types were specified so no issues will be found in the dataset. Set `issue_types` as None to consider a default set of issues."
            )
            return None
        issue_finder = issue_finder_factory(self._imagelab)(
            datalab=self, task=self.task, verbosity=self.verbosity
        )
        issue_finder.find_issues(
            pred_probs=pred_probs,
            features=features,
            knn_graph=knn_graph,
            issue_types=issue_types,
        )

        if self.verbosity:
            print(
                f"\nAudit complete. {self.data_issues.issue_summary['num_issues'].sum()} issues found in the dataset."
            )

    def report(
        self,
        *,
        num_examples: int = 5,
        verbosity: Optional[int] = None,
        include_description: bool = True,
        show_summary_score: bool = False,
        show_all_issues: bool = False,
    ) -> None:
        
        if verbosity is None:
            verbosity = self.verbosity
        if self.data_issues.issue_summary.empty:
            print("Please specify some `issue_types` in datalab.find_issues() to see a report.\n")
            return

        reporter = report_factory(self._imagelab)(
            data_issues=self.data_issues,
            task=self.task,
            verbosity=verbosity,
            include_description=include_description,
            show_summary_score=show_summary_score,
            show_all_issues=show_all_issues,
            imagelab=self._imagelab,
        )
        reporter.report(num_examples=num_examples)

    @property
    def issues(self) -> pd.DataFrame:
        return self.data_issues.issues

    @issues.setter
    def issues(self, issues: pd.DataFrame) -> None:
        self.data_issues.issues = issues

    @property
    def issue_summary(self) -> pd.DataFrame:
        return self.data_issues.issue_summary

    @issue_summary.setter
    def issue_summary(self, issue_summary: pd.DataFrame) -> None:
        self.data_issues.issue_summary = issue_summary

    @property
    def info(self) -> Dict[str, Dict[str, Any]]:
        return self.data_issues.info

    @info.setter
    def info(self, info: Dict[str, Dict[str, Any]]) -> None:
        self.data_issues.info = info

    def get_issues(self, issue_name: Optional[str] = None) -> pd.DataFrame:

        # Validate issue_name
        if issue_name is not None and issue_name not in self.list_possible_issue_types():
            raise ValueError(
                f
            )
        return self.data_issues.get_issues(issue_name=issue_name)

    def get_issue_summary(self, issue_name: Optional[str] = None) -> pd.DataFrame:
        return self.data_issues.get_issue_summary(issue_name=issue_name)

    def get_info(self, issue_name: Optional[str] = None) -> Dict[str, Any]:
        return self.data_issues.get_info(issue_name)

    def list_possible_issue_types(self) -> List[str]:
        possible_issue_types = _list_possible_issue_types(task=self.task)
        if self._imagelab is not None:
            possible_issue_types.extend(DEFAULT_CLEANVISION_ISSUES.keys())
        return possible_issue_types

    def list_default_issue_types(self) -> List[str]:
        default_issue_types = _list_default_issue_types(task=self.task)
        if self._imagelab is not None:
            default_issue_types.extend(DEFAULT_CLEANVISION_ISSUES.keys())
        return default_issue_types

    def save(self, path: str, force: bool = False) -> None:
        _Serializer.serialize(path=path, datalab=self, force=force)
        save_message = f"Saved Datalab to folder: {path}"
        print(save_message)

    @staticmethod
    def load(path: str, data: Optional[Dataset] = None) -> "Datalab":
        datalab = _Serializer.deserialize(path=path, data=data)
        load_message = f"Datalab loaded from folder: {path}"
        print(load_message)
        return datalab
