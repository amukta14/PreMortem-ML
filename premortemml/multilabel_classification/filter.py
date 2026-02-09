import warnings
import inspect
from typing import Optional, Union, Tuple, List, Any
import numpy as np

def find_label_issues(
    labels: list,
    pred_probs: np.ndarray,
    return_indices_ranked_by: Optional[str] = None,
    rank_by_kwargs={},
    filter_by: str = "prune_by_noise_rate",
    frac_noise: float = 1.0,
    num_to_remove_per_class: Optional[List[int]] = None,
    min_examples_per_class=1,
    confident_joint: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    low_memory: bool = False,
) -> np.ndarray:
    
    from premortemml.filter import _find_label_issues_multilabel

    if low_memory:
        if rank_by_kwargs:
            warnings.warn(f"`rank_by_kwargs` is not used when `low_memory=True`.")

        func_signature = inspect.signature(find_label_issues)
        default_args = {
            k: v.default
            for k, v in func_signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        arg_values = {
            "filter_by": filter_by,
            "num_to_remove_per_class": num_to_remove_per_class,
            "confident_joint": confident_joint,
            "n_jobs": n_jobs,
            "num_to_remove_per_class": num_to_remove_per_class,
            "frac_noise": frac_noise,
            "min_examples_per_class": min_examples_per_class,
        }
        for arg_name, arg_val in arg_values.items():
            if arg_val != default_args[arg_name]:
                warnings.warn(f"`{arg_name}` is not used when `low_memory=True`.")

    return _find_label_issues_multilabel(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by=return_indices_ranked_by,
        rank_by_kwargs=rank_by_kwargs,
        filter_by=filter_by,
        frac_noise=frac_noise,
        num_to_remove_per_class=num_to_remove_per_class,
        min_examples_per_class=min_examples_per_class,
        confident_joint=confident_joint,
        n_jobs=n_jobs,
        verbose=verbose,
        low_memory=low_memory,
    )

def find_multilabel_issues_per_class(
    labels: list,
    pred_probs: np.ndarray,
    return_indices_ranked_by: Optional[str] = None,
    rank_by_kwargs={},
    filter_by: str = "prune_by_noise_rate",
    frac_noise: float = 1.0,
    num_to_remove_per_class: Optional[List[int]] = None,
    min_examples_per_class=1,
    confident_joint: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    low_memory: bool = False,
) -> Union[np.ndarray, Tuple[List[np.ndarray], List[Any], List[np.ndarray]]]:
    
    from premortemml.internal.multilabel_utils import get_onehot_num_classes, stack_complement
    from premortemml.experimental.label_issues_batched import find_label_issues_batched
    from premortemml.filter import _find_label_issues_multilabel

    y_one, num_classes = get_onehot_num_classes(labels, pred_probs)
    if return_indices_ranked_by is None:
        bissues = np.zeros(y_one.shape).astype(bool)
    else:
        label_issues_list = []
    labels_list = []
    pred_probs_list = []
    if confident_joint is not None and not low_memory:
        confident_joint_shape = confident_joint.shape
        if confident_joint_shape == (num_classes, num_classes):
            warnings.warn(
                f"The new recommended format for `confident_joint` in multi_label settings is (num_classes,2,2) as output by compute_confident_joint(...,multi_label=True). Your K x K confident_joint in the old format is being ignored."
            )
            confident_joint = None
        elif confident_joint_shape != (num_classes, 2, 2):
            raise ValueError("confident_joint should be of shape (num_classes, 2, 2)")
    for class_num, (label, pred_prob_for_class) in enumerate(zip(y_one.T, pred_probs.T)):
        pred_probs_binary = stack_complement(pred_prob_for_class)
        if low_memory:
            quality_score_kwargs = (
                {"method": return_indices_ranked_by} if return_indices_ranked_by else None
            )
            binary_label_issues = find_label_issues_batched(
                labels=label,
                pred_probs=pred_probs_binary,
                verbose=verbose,
                quality_score_kwargs=quality_score_kwargs,
                return_mask=return_indices_ranked_by is None,
            )
        else:
            if confident_joint is None:
                conf = None
            else:
                conf = confident_joint[class_num]
            if num_to_remove_per_class is not None:
                ml_num_to_remove_per_class = [num_to_remove_per_class[class_num], 0]
            else:
                ml_num_to_remove_per_class = None
            binary_label_issues = _find_label_issues_multilabel(
                labels=label,
                pred_probs=pred_probs_binary,
                return_indices_ranked_by=return_indices_ranked_by,
                frac_noise=frac_noise,
                rank_by_kwargs=rank_by_kwargs,
                filter_by=filter_by,
                num_to_remove_per_class=ml_num_to_remove_per_class,
                min_examples_per_class=min_examples_per_class,
                confident_joint=conf,
                n_jobs=n_jobs,
                verbose=verbose,
                low_memory=False,
            )

        if return_indices_ranked_by is None:
            bissues[:, class_num] = binary_label_issues
        else:
            label_issues_list.append(binary_label_issues)
            labels_list.append(label)
            pred_probs_list.append(pred_probs_binary)
    if return_indices_ranked_by is None:
        return bissues
    else:
        return label_issues_list, labels_list, pred_probs_list
