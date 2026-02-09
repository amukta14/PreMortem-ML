import pandas as pd
import numpy as np
from typing import Optional, cast, Dict, Any  # noqa: F401
from premortemml.multilabel_classification.filter import (
    find_multilabel_issues_per_class,
    find_label_issues,
)
from premortemml.internal.multilabel_utils import get_onehot_num_classes
from collections import defaultdict

def common_multilabel_issues(
    labels=list,
    pred_probs=None,
    *,
    class_names=None,
    confident_joint=None,
) -> pd.DataFrame:
    

    num_examples = _get_num_examples_multilabel(labels=labels, confident_joint=confident_joint)
    summary_issue_counts = defaultdict(list)
    y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
    label_issues_list, labels_list, pred_probs_list = find_multilabel_issues_per_class(
        labels=labels,
        pred_probs=pred_probs,
        confident_joint=confident_joint,
        return_indices_ranked_by="self_confidence",
    )

    for class_num, (label, issues_for_class) in enumerate(zip(y_one.T, label_issues_list)):
        binary_label_issues = np.zeros(len(label)).astype(bool)
        binary_label_issues[issues_for_class] = True
        true_but_false_count = sum(np.logical_and(label == 1, binary_label_issues))
        false_but_true_count = sum(np.logical_and(label == 0, binary_label_issues))

        if class_names is not None:
            summary_issue_counts["Class Name"].append(class_names[class_num])
        summary_issue_counts["Class Index"].append(class_num)
        summary_issue_counts["In Given Label"].append(True)
        summary_issue_counts["In Suggested Label"].append(False)
        summary_issue_counts["Num Examples"].append(true_but_false_count)
        summary_issue_counts["Issue Probability"].append(true_but_false_count / num_examples)

        if class_names is not None:
            summary_issue_counts["Class Name"].append(class_names[class_num])
        summary_issue_counts["Class Index"].append(class_num)
        summary_issue_counts["In Given Label"].append(False)
        summary_issue_counts["In Suggested Label"].append(True)
        summary_issue_counts["Num Examples"].append(false_but_true_count)
        summary_issue_counts["Issue Probability"].append(false_but_true_count / num_examples)
    return (
        pd.DataFrame.from_dict(summary_issue_counts)
        .sort_values(by=["Issue Probability"], ascending=False)
        .reset_index(drop=True)
    )

def rank_classes_by_multilabel_quality(
    labels=None,
    pred_probs=None,
    *,
    class_names=None,
    joint=None,
    confident_joint=None,
) -> pd.DataFrame:
    

    issues_df = common_multilabel_issues(
        labels=labels, pred_probs=pred_probs, class_names=class_names, confident_joint=joint
    )
    issues_dict = defaultdict(defaultdict)  # type: Dict[str, Any]
    num_examples = _get_num_examples_multilabel(labels=labels, confident_joint=confident_joint)
    return_columns = [
        "Class Name",
        "Class Index",
        "Label Issues",
        "Inverse Label Issues",
        "Label Noise",
        "Inverse Label Noise",
        "Label Quality Score",
    ]
    if class_names is None:
        return_columns = return_columns[1:]
    for class_num, row in issues_df.iterrows():
        if row["In Given Label"]:
            if class_names is not None:
                issues_dict[row["Class Index"]]["Class Name"] = row["Class Name"]
            issues_dict[row["Class Index"]]["Label Issues"] = int(
                row["Issue Probability"] * num_examples
            )
            issues_dict[row["Class Index"]]["Label Noise"] = row["Issue Probability"]
            issues_dict[row["Class Index"]]["Label Quality Score"] = (
                1 - issues_dict[row["Class Index"]]["Label Noise"]
            )
        else:
            if class_names is not None:
                issues_dict[row["Class Index"]]["Class Name"] = row["Class Name"]
            issues_dict[row["Class Index"]]["Inverse Label Issues"] = int(
                row["Issue Probability"] * num_examples
            )
            issues_dict[row["Class Index"]]["Inverse Label Noise"] = row["Issue Probability"]

    issues_df_dict = defaultdict(list)
    for i in issues_dict:
        issues_df_dict["Class Index"].append(i)
        for j in issues_dict[i]:
            issues_df_dict[j].append(issues_dict[i][j])
    return (
        pd.DataFrame.from_dict(issues_df_dict)
        .sort_values(by="Label Quality Score", ascending=True)
        .reset_index(drop=True)
    )[return_columns]

def _get_num_examples_multilabel(labels=None, confident_joint: Optional[np.ndarray] = None) -> int:

    if labels is None and confident_joint is None:
        raise ValueError(
            "Error: num_examples is None. You must either provide confident_joint, "
            "or provide both num_example and joint as input parameters."
        )
    _confident_joint = cast(np.ndarray, confident_joint)
    num_examples = len(labels) if labels is not None else cast(int, np.sum(_confident_joint[0]))
    return num_examples

def overall_multilabel_health_score(
    labels=None,
    pred_probs=None,
    *,
    confident_joint=None,
) -> float:
    
    num_examples = _get_num_examples_multilabel(labels=labels)
    issues = find_label_issues(
        labels=labels, pred_probs=pred_probs, confident_joint=confident_joint
    )
    return 1.0 - sum(issues) / num_examples

def multilabel_health_summary(
    labels=None,
    pred_probs=None,
    *,
    class_names=None,
    num_examples=None,
    confident_joint=None,
    verbose=True,
) -> Dict:
    
    from premortemml.internal.util import smart_display_dataframe

    if num_examples is None:
        num_examples = _get_num_examples_multilabel(labels=labels)

    if verbose:
        longest_line = f"|   for your dataset with {num_examples:,} examples "
        print(
            "-" * (len(longest_line) - 1)
            + "\n"
            + f"|  Generating a PreMortemML Dataset Health Summary{' ' * (len(longest_line) - 49)}|\n"
            + longest_line
            + f"|  Note, PreMortemML is not a medical doctor... yet.{' ' * (len(longest_line) - 51)}|\n"
            + "-" * (len(longest_line) - 1)
            + "\n",
        )

    df_class_label_quality = rank_classes_by_multilabel_quality(
        labels=labels,
        pred_probs=pred_probs,
        class_names=class_names,
        confident_joint=confident_joint,
    )
    if verbose:
        print("Overall Class Quality and Noise across your dataset (below)")
        print("-" * 60, "\n", flush=True)
        smart_display_dataframe(df_class_label_quality)

    df_common_issues = common_multilabel_issues(
        labels=labels,
        pred_probs=pred_probs,
        class_names=class_names,
        confident_joint=confident_joint,
    )
    if verbose:
        print(
            "\nCommon multilabel issues are" + "\n" + "-" * 83 + "\n",
            flush=True,
        )
        smart_display_dataframe(df_common_issues)
        print()

    health_score = overall_multilabel_health_score(
        labels=labels,
        pred_probs=pred_probs,
        confident_joint=confident_joint,
    )
    if verbose:
        print("\nGenerated with <3 from PreMortemML.\n")
    return {
        "overall_multilabel_health_score": health_score,
        "classes_by_multilabel_quality": df_class_label_quality,
        "common_multilabel_issues": df_common_issues,
    }
