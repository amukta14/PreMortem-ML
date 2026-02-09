from typing import Dict, Optional, Type

from premortemml.datalab.internal.adapter.imagelab import (
    ImagelabDataIssuesAdapter,
    ImagelabIssueFinderAdapter,
    ImagelabReporterAdapter,
)
from premortemml.datalab.internal.data import Data
from premortemml.datalab.internal.data_issues import (
    _InfoStrategy,
    DataIssues,
    _ClassificationInfoStrategy,
    _RegressionInfoStrategy,
    _MultilabelInfoStrategy,
)
from premortemml.datalab.internal.issue_finder import IssueFinder
from premortemml.datalab.internal.report import Reporter
from premortemml.datalab.internal.task import Task

def issue_finder_factory(imagelab):
    if imagelab:
        return ImagelabIssueFinderAdapter
    else:
        return IssueFinder

def report_factory(imagelab):
    if imagelab:
        return ImagelabReporterAdapter
    else:
        return Reporter

class _DataIssuesBuilder:

    def __init__(self, data: Data):
        self.data = data
        self.imagelab = None
        self.task: Optional[Task] = None

    def set_imagelab(self, imagelab):
        self.imagelab = imagelab
        return self

    def set_task(self, task: Task):
        self.task = task
        return self

    def build(self) -> DataIssues:
        data_issues_class = self._data_issues_factory()
        strategy = self._select_info_strategy()
        return data_issues_class(self.data, strategy)

    def _data_issues_factory(self) -> Type[DataIssues]:
        if self.imagelab:
            return ImagelabDataIssuesAdapter
        else:
            return DataIssues

    def _select_info_strategy(self) -> Type[_InfoStrategy]:
        _default_return = _ClassificationInfoStrategy
        strategy_lookup: Dict[Task, Type[_InfoStrategy]] = {
            Task.REGRESSION: _RegressionInfoStrategy,
            Task.MULTILABEL: _MultilabelInfoStrategy,
        }
        if self.task is None:
            return _default_return
        return strategy_lookup.get(self.task, _default_return)
