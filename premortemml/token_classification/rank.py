import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple

from premortemml.rank import get_label_quality_scores as main_get_label_quality_scores
from premortemml.internal.numerics import softmax

def get_label_quality_scores(
    labels: list,
    pred_probs: list,
    *,
    tokens: Optional[list] = None,
    token_score_method: str = "self_confidence",
    sentence_score_method: str = "min",
    sentence_score_kwargs: dict = {},
) -> Tuple[np.ndarray, list]:
    
    methods = ["min", "softmin"]
    assert sentence_score_method in methods, "Select from the following methods:\n%s" % "\n".join(
        methods
    )

    labels_flatten = np.array([l for label in labels for l in label])
    pred_probs_flatten = np.array([p for pred_prob in pred_probs for p in pred_prob])

    sentence_length = [len(label) for label in labels]

    def nested_list(x, sentence_length):
        i = iter(x)
        return [[next(i) for _ in range(length)] for length in sentence_length]

    token_scores = main_get_label_quality_scores(
        labels=labels_flatten, pred_probs=pred_probs_flatten, method=token_score_method
    )
    scores_nl = nested_list(token_scores, sentence_length)

    if sentence_score_method == "min":
        sentence_scores = np.array(list(map(np.min, scores_nl)))
    else:
        assert sentence_score_method == "softmin"
        temperature = sentence_score_kwargs.get("temperature", 0.05)
        sentence_scores = _softmin_sentence_score(scores_nl, temperature=temperature)

    if tokens:
        token_info = [pd.Series(scores, index=token) for scores, token in zip(scores_nl, tokens)]
    else:
        token_info = [pd.Series(scores) for scores in scores_nl]
    return sentence_scores, token_info

def issues_from_scores(
    sentence_scores: np.ndarray, *, token_scores: Optional[list] = None, threshold: float = 0.1
) -> Union[list, np.ndarray]:
    
    if token_scores:
        issues_with_scores = []
        for sentence_index, scores in enumerate(token_scores):
            for token_index, score in enumerate(scores):
                if score < threshold:
                    issues_with_scores.append((sentence_index, token_index, score))

        issues_with_scores = sorted(issues_with_scores, key=lambda x: x[2])
        issues = [(i, j) for i, j, _ in issues_with_scores]
        return issues

    else:
        ranking = np.argsort(sentence_scores)
        cutoff = 0
        while sentence_scores[ranking[cutoff]] < threshold and cutoff < len(ranking):
            cutoff += 1
        return ranking[:cutoff]

def _softmin_sentence_score(
    token_scores: List[np.ndarray], *, temperature: float = 0.05
) -> np.ndarray:
    
    if temperature == 0:
        return np.array([np.min(scores) for scores in token_scores])

    if temperature == np.inf:
        return np.array([np.mean(scores) for scores in token_scores])

    def fun(scores: np.ndarray) -> float:
        return np.dot(
            scores, softmax(x=1 - np.array(scores), temperature=temperature, axis=0, shift=True)
        )

    sentence_scores = list(map(fun, token_scores))
    return np.array(sentence_scores)
