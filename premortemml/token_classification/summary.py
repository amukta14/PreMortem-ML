from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from premortemml.internal.token_classification_utils import color_sentence, get_sentence

def display_issues(
    issues: list,
    tokens: List[List[str]],
    *,
    labels: Optional[list] = None,
    pred_probs: Optional[list] = None,
    exclude: List[Tuple[int, int]] = [],
    class_names: Optional[List[str]] = None,
    top: int = 20,
) -> None:
    
    if not class_names and (labels or pred_probs):
        print(
            "Classes will be printed in terms of their integer index since `class_names` was not provided.\n"
            "Specify this argument to see the string names of each class.\n"
        )

    top = min(top, len(issues))
    shown = 0
    is_tuple = isinstance(issues[0], tuple)

    for issue in issues:
        if is_tuple:
            i, j = issue
            sentence = get_sentence(tokens[i])
            word = tokens[i][j]

            if pred_probs:
                prediction = pred_probs[i][j].argmax()
            if labels:
                given = labels[i][j]
            if pred_probs and labels:
                if (given, prediction) in exclude:
                    continue

            if pred_probs and class_names:
                prediction = class_names[prediction]
            if labels and class_names:
                given = class_names[given]

            shown += 1
            print(f"Sentence index: {i}, Token index: {j}")
            print(f"Token: {word}")
            if labels and not pred_probs:
                print(f"Given label: {given}")
            elif not labels and pred_probs:
                print(f"Predicted label according to provided pred_probs: {prediction}")
            elif labels and pred_probs:
                print(
                    f"Given label: {given}, predicted label according to provided pred_probs: {prediction}"
                )
            print("----")
            print(color_sentence(sentence, word))
        else:
            shown += 1
            sentence = get_sentence(tokens[issue])
            print(f"Sentence issue: {sentence}")
        if shown == top:
            break
        print("\n")

def common_label_issues(
    issues: List[Tuple[int, int]],
    tokens: List[List[str]],
    *,
    labels: Optional[list] = None,
    pred_probs: Optional[list] = None,
    class_names: Optional[List[str]] = None,
    top: int = 10,
    exclude: List[Tuple[int, int]] = [],
    verbose: bool = True,
) -> pd.DataFrame:
    
    count: Dict[str, Any] = {}
    if not labels or not pred_probs:
        for issue in issues:
            i, j = issue
            word = tokens[i][j]
            if word not in count:
                count[word] = 0
            count[word] += 1

        words = [word for word in count.keys()]
        freq = [count[word] for word in words]
        rank = np.argsort(freq)[::-1][:top]

        for r in rank:
            print(
                f"Token '{words[r]}' is potentially mislabeled {freq[r]} times throughout the dataset\n"
            )

        info = [[word, f] for word, f in zip(words, freq)]
        info = sorted(info, key=lambda x: x[1], reverse=True)
        return pd.DataFrame(info, columns=["token", "num_label_issues"])

    if not class_names:
        print(
            "Classes will be printed in terms of their integer index since `class_names` was not provided. "
        )
        print("Specify this argument to see the string names of each class. \n")

    n = pred_probs[0].shape[1]
    for issue in issues:
        i, j = issue
        word = tokens[i][j]
        label = labels[i][j]
        pred = pred_probs[i][j].argmax()
        if word not in count:
            count[word] = np.zeros([n, n], dtype=int)
        if (label, pred) not in exclude:
            count[word][label][pred] += 1
    words = [word for word in count.keys()]
    freq = [np.sum(count[word]) for word in words]
    rank = np.argsort(freq)[::-1][:top]

    for r in rank:
        matrix = count[words[r]]
        most_frequent = np.argsort(count[words[r]].flatten())[::-1]
        print(
            f"Token '{words[r]}' is potentially mislabeled {freq[r]} times throughout the dataset"
        )
        if verbose:
            print(
                "---------------------------------------------------------------------------------------"
            )
            for f in most_frequent:
                i, j = f // n, f % n
                if matrix[i][j] == 0:
                    break
                if class_names:
                    print(
                        f"labeled as class `{class_names[i]}` but predicted to actually be class `{class_names[j]}` {matrix[i][j]} times"
                    )
                else:
                    print(
                        f"labeled as class {i} but predicted to actually be class {j} {matrix[i][j]} times"
                    )
        print()
    info = []
    for word in words:
        for i in range(n):
            for j in range(n):
                num = count[word][i][j]
                if num > 0:
                    if not class_names:
                        info.append([word, i, j, num])
                    else:
                        info.append([word, class_names[i], class_names[j], num])
    info = sorted(info, key=lambda x: x[3], reverse=True)
    return pd.DataFrame(
        info, columns=["token", "given_label", "predicted_label", "num_label_issues"]
    )

def filter_by_token(
    token: str, issues: List[Tuple[int, int]], tokens: List[List[str]]
) -> List[Tuple[int, int]]:
    
    returned_issues = []
    for issue in issues:
        i, j = issue
        if token.lower() == tokens[i][j].lower():
            returned_issues.append(issue)
    return returned_issues
