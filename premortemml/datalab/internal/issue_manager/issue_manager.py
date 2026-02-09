from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from itertools import chain
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Set, Tuple, Type, TypeVar
import json

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from premortemml.datalab.datalab import Datalab

T = TypeVar("T", bound="IssueManager")
TM = TypeVar("TM", bound="IssueManagerMeta")

class IssueManagerMeta(ABCMeta):

    issue_name: ClassVar[str]
    issue_score_key: ClassVar[str]
    verbosity_levels: ClassVar[Dict[int, List[str]]] = {
        0: [],
        1: [],
        2: [],
        3: [],
    }

    def __new__(
        meta: Type[TM],
        name: str,
        bases: Tuple[Type[Any], ...],
        class_dict: Dict[str, Any],
    ) -> TM:  # Classes that inherit from ABC don't need to be modified
        if ABC in bases:
            return super().__new__(meta, name, bases, class_dict)

        # Ensure that the verbosity levels don't have keys other than those in ["issue", "info"]
        verbosity_levels = class_dict.get("verbosity_levels", meta.verbosity_levels)
        for level, level_list in verbosity_levels.items():
            if not isinstance(level_list, list):
                raise ValueError(
                    f"Verbosity levels must be lists. "
                    f"Got {level_list} in {name}.verbosity_levels"
                )
            prohibited_keys = [key for key in level_list if not isinstance(key, str)]
            if prohibited_keys:
                raise ValueError(
                    f"Verbosity levels must be lists of strings. "
                    f"Got {prohibited_keys} in {name}.verbosity_levels[{level}]"
                )

        # Concrete classes need to have an issue_name attribute
        if "issue_name" not in class_dict:
            raise TypeError("IssueManagers need an issue_name class variable")

        # Add issue_score_key to class
        class_dict["issue_score_key"] = f"{class_dict['issue_name']}_score"
        return super().__new__(meta, name, bases, class_dict)

class IssueManager(ABC, metaclass=IssueManagerMeta):

    description: ClassVar[str] = ""
    
    issue_name: ClassVar[str]
    
    issue_score_key: ClassVar[str]
    
    verbosity_levels: ClassVar[Dict[int, List[str]]] = {
        0: [],
        1: [],
        2: [],
        3: [],
    }
    

    def __init__(self, datalab: Datalab, **_):
        self.datalab = datalab
        self.info: Dict[str, Any] = {}
        self.issues: pd.DataFrame = pd.DataFrame()
        self.summary: pd.DataFrame = pd.DataFrame()

    def __repr__(self):
        class_name = self.__class__.__name__
        return class_name

    @classmethod
    def __init_subclass__(cls):
        required_class_variables = [
            "issue_name",
        ]
        for var in required_class_variables:
            if not hasattr(cls, var):
                raise NotImplementedError(f"Class {cls.__name__} must define class variable {var}")

    @abstractmethod
    def find_issues(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def collect_info(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    @classmethod
    def make_summary(cls, score: float) -> pd.DataFrame:
        if not 0 <= score <= 1:
            raise ValueError(f"Score must be between 0 and 1. Got {score}.")

        return pd.DataFrame(
            {
                "issue_type": [cls.issue_name],
                "score": [score],
            },
        )

    @classmethod
    def report(
        cls,
        issues: pd.DataFrame,
        summary: pd.DataFrame,
        info: Dict[str, Any],
        num_examples: int = 5,
        verbosity: int = 0,
        include_description: bool = False,
        info_to_omit: Optional[List[str]] = None,
    ) -> str:
        

        max_verbosity = max(cls.verbosity_levels.keys())
        top_level = max_verbosity + 1
        if verbosity not in list(cls.verbosity_levels.keys()) + [top_level]:
            raise ValueError(
                f"Verbosity level {verbosity} not supported. "
                f"Supported levels: {cls.verbosity_levels.keys()}"
                f"Use verbosity={top_level} to print all info."
            )
        if issues.empty:
            print(f"No issues found")

        topk_ids = issues.sort_values(by=cls.issue_score_key, ascending=True).index[:num_examples]

        score = summary["score"].loc[0]
        report_str = f"{' ' + cls.issue_name + ' issues ':-^60}\n\n"

        if include_description and cls.description:
            description = cls.description
            if verbosity == 0:
                description = description.split("\n\n", maxsplit=1)[0]
            report_str += "About this issue:\n\t" + description + "\n\n"
        report_str += (
            f"Number of examples with this issue: {issues[f'is_{cls.issue_name}_issue'].sum()}\n"
            f"Overall dataset quality in terms of this issue: {score:.4f}\n\n"
        )

        info_to_print: Set[str] = set()
        _info_to_omit = set(issues.columns).union(info_to_omit or [])
        verbosity_levels_values = chain.from_iterable(
            list(cls.verbosity_levels.values())[: verbosity + 1]
        )
        info_to_print.update(set(verbosity_levels_values) - _info_to_omit)
        if verbosity == top_level:
            info_to_print.update(set(info.keys()) - _info_to_omit)

        report_str += "Examples representing most severe instances of this issue:\n"
        report_str += issues.loc[topk_ids].to_string()

        def truncate(s, max_len=4) -> str:
            if hasattr(s, "shape") or hasattr(s, "ndim"):
                s = np.array(s)
                if s.ndim > 1:
                    description = f"array of shape {s.shape}\n"
                    with np.printoptions(threshold=max_len):
                        if s.ndim == 2:
                            description += f"{s}"
                        if s.ndim > 2:
                            description += f"{s}"
                    return description
                s = s.tolist()

            if isinstance(s, list):
                if all([isinstance(s_, list) for s_ in s]):
                    return truncate(np.array(s, dtype=object), max_len=max_len)
                if len(s) > max_len:
                    s = s[:max_len] + ["..."]
            return str(s)

        if info_to_print:
            info_to_print_dict = {key: info[key] for key in info_to_print}
            # Print the info dict, truncating arrays to 4 elements,
            report_str += f"\n\nAdditional Information: "
            for key, value in info_to_print_dict.items():
                if key == "statistics":
                    continue
                if isinstance(value, dict):
                    report_str += f"\n{key}:\n{json.dumps(value, indent=4)}"
                elif isinstance(value, pd.DataFrame):
                    max_rows = 5
                    df_str = value.head(max_rows).to_string()
                    if len(value) > max_rows:
                        df_str += f"\n... (total {len(value)} rows)"
                    report_str += f"\n{key}:\n{df_str}"
                else:
                    report_str += f"\n{key}: {truncate(value)}"
        return report_str
