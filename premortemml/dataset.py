from typing import Optional, cast
import numpy as np
import pandas as pd

from premortemml.count import estimate_joint, num_label_issues
from premortemml.internal.constants import EPSILON

def rank_classes_by_label_quality(
    labels=None,
    pred_probs=None,
    *,
    class_names=None,
    num_examples=None,
    joint=None,
    confident_joint=None,
    multi_label=False,
) -> pd.DataFrame:
    
    if multi_label:
        raise ValueError(
            "For multilabel data, please instead call:  multilabel_classification.dataset.overall_multilabel_health_score()"
        )

    if joint is None:
        joint = estimate_joint(
            labels=labels,
            pred_probs=pred_probs,
            confident_joint=confident_joint,
        )
    if num_examples is None:
        num_examples = _get_num_examples(labels=labels)
    given_label_noise = joint.sum(axis=1) - joint.diagonal()  # p(s=k) - p(s=k,y=k) = p(y!=k, s=k)
    true_label_noise = joint.sum(axis=0) - joint.diagonal()  # p(y=k) - p(s=k,y=k) = p(s!=k,y=k)
    given_conditional_noise = given_label_noise / np.clip(
        joint.sum(axis=1), a_min=EPSILON, a_max=None
    )  # p(y!=k, s=k) / p(s=k) , avoiding division by 0
    true_conditional_noise = true_label_noise / np.clip(
        joint.sum(axis=0), a_min=EPSILON, a_max=None
    )  # p(s!=k, y=k) / p(y=k) , avoiding division by 0
    df = pd.DataFrame(
        {
            "Class Index": np.arange(len(joint)),
            "Label Issues": (given_label_noise * num_examples).round().astype(int),
            "Inverse Label Issues": (true_label_noise * num_examples).round().astype(int),
            "Label Noise": given_conditional_noise,  # p(y!=k | s=k)
            "Inverse Label Noise": true_conditional_noise,  # p(s!=k | y=k)
            # Below could equivalently be computed as: joint.diagonal() / joint.sum(axis=1)
            "Label Quality Score": 1 - given_conditional_noise,  # p(y=k | s=k)
        }
    )
    if class_names is not None:
        df.insert(loc=0, column="Class Name", value=class_names)
    return df.sort_values(by="Label Quality Score", ascending=True).reset_index(drop=True)

def find_overlapping_classes(
    labels=None,
    pred_probs=None,
    *,
    asymmetric=False,
    class_names=None,
    num_examples=None,
    joint=None,
    confident_joint=None,
    multi_label=False,
) -> pd.DataFrame:
    

    def _2d_matrix_to_row_column_value_list(matrix):

        return [(*i, v) for i, v in np.ndenumerate(matrix)]

    if multi_label:
        raise ValueError(
            "For multilabel data, please instead call: multilabel_classification.dataset.common_multilabel_issues()"
        )

    if joint is None:
        joint = estimate_joint(
            labels=labels,
            pred_probs=pred_probs,
            confident_joint=confident_joint,
        )
    if num_examples is None:
        num_examples = _get_num_examples(labels=labels, confident_joint=confident_joint)
    if asymmetric:
        rcv_list = _2d_matrix_to_row_column_value_list(joint)
        # Remove diagonal elements
        rcv_list = [tup for tup in rcv_list if tup[0] != tup[1]]
    else:  # symmetric
        # Sum the upper and lower triangles and remove the lower triangle and the diagonal
        sym_joint = np.triu(joint) + np.tril(joint).T
        rcv_list = _2d_matrix_to_row_column_value_list(sym_joint)
        # Provide values only in (the upper triangle) of the matrix.
        rcv_list = [tup for tup in rcv_list if tup[0] < tup[1]]
    df = pd.DataFrame(rcv_list, columns=["Class Index A", "Class Index B", "Joint Probability"])
    num_overlapping = (df["Joint Probability"] * num_examples).round().astype(int)
    df.insert(loc=2, column="Num Overlapping Examples", value=num_overlapping)
    if class_names is not None:
        df.insert(
            loc=0, column="Class Name A", value=df["Class Index A"].apply(lambda x: class_names[x])
        )
        df.insert(
            loc=1, column="Class Name B", value=df["Class Index B"].apply(lambda x: class_names[x])
        )
    return df.sort_values(by="Joint Probability", ascending=False).reset_index(drop=True)

def overall_label_health_score(
    labels=None,
    pred_probs=None,
    *,
    num_examples=None,
    confident_joint=None,
    joint=None,
    multi_label=False,
    verbose=True,
) -> float:
    
    if multi_label:
        raise ValueError(
            "For multilabel data, please instead call: multilabel_classification.dataset.overall_multilabel_health_score()"
        )
    if num_examples is None:
        num_examples = _get_num_examples(labels=labels, confident_joint=confident_joint)

    if pred_probs is None or labels is None:
        if joint is None:
            joint = estimate_joint(
                labels=labels,
                pred_probs=pred_probs,
                confident_joint=confident_joint,
            )
        joint_trace = joint.trace()
        num_issues = (num_examples * (1 - joint_trace)).round().astype(int)
        health_score = joint_trace
    else:
        num_issues = num_label_issues(
            labels=labels, pred_probs=pred_probs, confident_joint=confident_joint
        )
        health_score = 1 - num_issues / num_examples

    if verbose:
        print(
            f" * Overall, about {(1 - health_score):.0%} ({num_issues:,} of the {num_examples:,}) "
            f"labels in your dataset have potential issues.\n"
            f" ** The overall label health score for this dataset is: {health_score:.2f}."
        )
    return health_score

def health_summary(
    labels=None,
    pred_probs=None,
    *,
    asymmetric=False,
    class_names=None,
    num_examples=None,
    joint=None,
    confident_joint=None,
    multi_label=False,
    verbose=True,
) -> dict:
    
    from premortemml.internal.util import smart_display_dataframe

    if multi_label:
        raise ValueError(
            "For multilabel data, please call multilabel_classification.dataset.health_summary"
        )
    if joint is None:
        joint = estimate_joint(
            labels=labels,
            pred_probs=pred_probs,
            confident_joint=confident_joint,
        )
    if num_examples is None:
        num_examples = _get_num_examples(labels=labels)

    if verbose:
        longest_line = (
            f"|   for your dataset with {num_examples:,} examples "
            f"and {len(joint):,} classes.  |\n"
        )
        print(
            "-" * (len(longest_line) - 1)
            + "\n"
            + f"|  Generating a PreMortemML Dataset Health Summary{' ' * (len(longest_line) - 49)}|\n"
            + longest_line
            + f"|  Note, PreMortemML is not a medical doctor... yet.{' ' * (len(longest_line) - 51)}|\n"
            + "-" * (len(longest_line) - 1)
            + "\n",
        )

    df_class_label_quality = rank_classes_by_label_quality(
        labels=labels,
        pred_probs=pred_probs,
        class_names=class_names,
        num_examples=num_examples,
        joint=joint,
        confident_joint=confident_joint,
    )
    if verbose:
        print("Overall Class Quality and Noise across your dataset (below)")
        print("-" * 60, "\n", flush=True)
        smart_display_dataframe(df_class_label_quality)

    df_overlapping_classes = find_overlapping_classes(
        labels=labels,
        pred_probs=pred_probs,
        asymmetric=asymmetric,
        class_names=class_names,
        num_examples=num_examples,
        joint=joint,
        confident_joint=confident_joint,
    )
    if verbose:
        print(
            "\nClass Overlap. In some cases, you may want to merge classes in the top rows (below)"
            + "\n"
            + "-" * 83
            + "\n",
            flush=True,
        )
        smart_display_dataframe(df_overlapping_classes)
        print()

    health_score = overall_label_health_score(
        labels=labels,
        pred_probs=pred_probs,
        num_examples=num_examples,
        confident_joint=confident_joint,
        verbose=verbose,
    )
    if verbose:
        print("\nGenerated with <3 from PreMortemML.\n")
    return {
        "overall_label_health_score": health_score,
        "joint": joint,
        "classes_by_label_quality": df_class_label_quality,
        "overlapping_classes": df_overlapping_classes,
    }

def _get_num_examples(labels=None, confident_joint: Optional[np.ndarray] = None) -> int:

    if labels is None and confident_joint is None:
        raise ValueError(
            "Error: num_examples is None. You must either provide confident_joint, "
            "or provide both num_example and joint as input parameters."
        )
    _confident_joint = cast(np.ndarray, confident_joint)
    num_examples = len(labels) if labels is not None else cast(int, np.sum(_confident_joint))
    return num_examples
