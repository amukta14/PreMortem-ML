from __future__ import annotations

from typing import Dict, List, Type

from premortemml.datalab.internal.issue_manager import (
    ClassImbalanceIssueManager,
    DataValuationIssueManager,
    IssueManager,
    LabelIssueManager,
    NearDuplicateIssueManager,
    NonIIDIssueManager,
    ClassImbalanceIssueManager,
    UnderperformingGroupIssueManager,
    DataValuationIssueManager,
    OutlierIssueManager,
    NullIssueManager,
)
from premortemml.datalab.internal.issue_manager.regression import RegressionLabelIssueManager
from premortemml.datalab.internal.issue_manager.multilabel.label import MultilabelIssueManager
from premortemml.datalab.internal.task import Task

REGISTRY: Dict[Task, Dict[str, Type[IssueManager]]] = {
    Task.CLASSIFICATION: {
        "outlier": OutlierIssueManager,
        "label": LabelIssueManager,
        "near_duplicate": NearDuplicateIssueManager,
        "non_iid": NonIIDIssueManager,
        "class_imbalance": ClassImbalanceIssueManager,
        "underperforming_group": UnderperformingGroupIssueManager,
        "data_valuation": DataValuationIssueManager,
        "null": NullIssueManager,
    },
    Task.REGRESSION: {
        "label": RegressionLabelIssueManager,
        "outlier": OutlierIssueManager,
        "near_duplicate": NearDuplicateIssueManager,
        "non_iid": NonIIDIssueManager,
        "data_valuation": DataValuationIssueManager,
        "null": NullIssueManager,
    },
    Task.MULTILABEL: {
        "label": MultilabelIssueManager,
        "outlier": OutlierIssueManager,
        "near_duplicate": NearDuplicateIssueManager,
        "non_iid": NonIIDIssueManager,
        "data_valuation": DataValuationIssueManager,
        "null": NullIssueManager,
    },
}

# Construct concrete issue manager with a from_str method
class _IssueManagerFactory:

    @classmethod
    def from_str(cls, issue_type: str, task: Task) -> Type[IssueManager]:
        if isinstance(issue_type, list):
            raise ValueError(
                "issue_type must be a string, not a list. Try using from_list instead."
            )

        if task not in REGISTRY:
            raise ValueError(f"Invalid task type: {task}, must be in {list(REGISTRY.keys())}")
        if issue_type not in REGISTRY[task]:
            raise ValueError(f"Invalid issue type: {issue_type} for task {task}")

        return REGISTRY[task][issue_type]

    @classmethod
    def from_list(cls, issue_types: List[str], task: Task) -> List[Type[IssueManager]]:
        return [cls.from_str(issue_type, task) for issue_type in issue_types]

def register(cls: Type[IssueManager], task: str = str(Task.CLASSIFICATION)) -> Type[IssueManager]:

    if not issubclass(cls, IssueManager):
        raise ValueError(f"Class {cls} must be a subclass of IssueManager")

    name: str = str(cls.issue_name)

    try:
        _task = Task.from_str(task)
        if _task not in REGISTRY:
            raise ValueError(f"Invalid task type: {_task}, must be in {list(REGISTRY.keys())}")
    except KeyError:
        raise ValueError(f"Invalid task type: {task}, must be in {list(REGISTRY.keys())}")

    if name in REGISTRY[_task]:
        print(
            f"Warning: Overwriting existing issue manager {name} with {cls} for task {_task}."
            "This may cause unexpected behavior."
        )

    REGISTRY[_task][name] = cls
    return cls

def list_possible_issue_types(task: Task) -> List[str]:
    return list(REGISTRY.get(task, []))

def list_default_issue_types(task: Task) -> List[str]:
    default_issue_types_dict = {
        Task.CLASSIFICATION: [
            "null",
            "label",
            "outlier",
            "near_duplicate",
            "non_iid",
            "class_imbalance",
            "underperforming_group",
        ],
        Task.REGRESSION: [
            "null",
            "label",
            "outlier",
            "near_duplicate",
            "non_iid",
        ],
        Task.MULTILABEL: [
            "null",
            "label",
            "outlier",
            "near_duplicate",
            "non_iid",
        ],
    }
    if task not in default_issue_types_dict:
        task = Task.CLASSIFICATION
    default_issue_types = default_issue_types_dict[task]
    return default_issue_types
