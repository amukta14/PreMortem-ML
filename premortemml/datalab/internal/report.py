from typing import TYPE_CHECKING, List

import pandas as pd

from premortemml.datalab.internal.adapter.constants import DEFAULT_CLEANVISION_ISSUES
from premortemml.datalab.internal.issue_manager_factory import _IssueManagerFactory
from premortemml.datalab.internal.task import Task

if TYPE_CHECKING:  # pragma: no cover
    from premortemml.datalab.internal.data_issues import DataIssues

class Reporter:

    def __init__(
        self,
        data_issues: "DataIssues",
        task: Task,
        verbosity: int = 1,
        include_description: bool = True,
        show_summary_score: bool = False,
        show_all_issues: bool = False,
        **kwargs,
    ):
        self.data_issues = data_issues
        self.task = task
        self.verbosity = verbosity
        self.include_description = include_description
        self.show_summary_score = show_summary_score
        self.show_all_issues = show_all_issues

    def _get_empty_report(self) -> str:
        report_str = "No issues found in the data. Good job!"
        if not self.show_summary_score:
            recommendation_msg = (
                "Try re-running Datalab.report() with "
                "`show_summary_score = True` and `show_all_issues = True`."
            )
            report_str += f"\n\n{recommendation_msg}"
        return report_str

    def report(self, num_examples: int) -> None:
        print(self.get_report(num_examples=num_examples))

    def get_report(self, num_examples: int) -> str:
        report_str = ""
        issue_summary = self.data_issues.issue_summary
        should_return_empty_report = not (
            self.show_all_issues or issue_summary.empty or issue_summary["num_issues"].sum() > 0
        )

        if should_return_empty_report:
            return self._get_empty_report()
        issue_summary_sorted = issue_summary.sort_values(by="num_issues", ascending=False)
        report_str += self._write_summary(summary=issue_summary_sorted)

        issue_types = self._get_issue_types(issue_summary_sorted)

        def add_issue_to_report(issue_name: str) -> bool:
            if self.show_all_issues:
                return True
            summary = self.data_issues.get_issue_summary(issue_name=issue_name)
            has_issues = summary["num_issues"][0] > 0
            return has_issues

        issue_reports = [
            _IssueManagerFactory.from_str(issue_type=key, task=self.task).report(
                issues=self.data_issues.get_issues(issue_name=key),
                summary=self.data_issues.get_issue_summary(issue_name=key),
                info=self.data_issues.get_info(issue_name=key),
                num_examples=num_examples,
                verbosity=self.verbosity,
                include_description=self.include_description,
            )
            for key in issue_types
        ]

        report_str += "\n\n\n".join(issue_reports)
        return report_str

    def _write_summary(self, summary: pd.DataFrame) -> str:
        statistics = self.data_issues.get_info("statistics")
        num_examples = statistics["num_examples"]
        num_classes = statistics.get(
            "num_classes"
        )  # This may not be required for all types of datasets  in the future (e.g. unlabeled/regression)

        dataset_information = f"Dataset Information: num_examples: {num_examples}"
        if num_classes is not None:
            dataset_information += f", num_classes: {num_classes}"

        if not self.show_all_issues:
            # Drop any items in the issue_summary that have no issues (any issue detected in data needs to have num_issues > 0)
            summary = summary.query("num_issues > 0")

        report_header = (
            f"{dataset_information}\n\n"
            + "Here is a summary of various issues found in your data:\n\n"
        )
        report_footer = (
            "\n\n"
            + "See which examples in your dataset exhibit each issue via: `datalab.get_issues(<ISSUE_NAME>)`\n\n"
            + "Data indices corresponding to top examples of each issue are shown below.\n\n\n"
        )

        if self.show_summary_score:
            return (
                report_header
                + summary.to_string(index=False)
                + "\n\n"
                + "(Note: A lower score indicates a more severe issue across all examples in the dataset.)"
                + report_footer
            )

        return (
            report_header + summary.drop(columns=["score"]).to_string(index=False) + report_footer
        )

    def _get_issue_types(self, issue_summary: pd.DataFrame) -> List[str]:
        issue_types = [
            issue_type
            for issue_type, num_issues in zip(
                issue_summary["issue_type"].tolist(), issue_summary["num_issues"].tolist()
            )
            if issue_type not in DEFAULT_CLEANVISION_ISSUES
            and (self.show_all_issues or num_issues > 0)
        ]
        return issue_types
