from __future__ import annotations

import re
import string
import numpy as np
from termcolor import colored
from typing import List, Optional, Callable, Tuple, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    T = TypeVar("T", bound=npt.NBitBase)

def get_sentence(words: List[str]) -> str:
    sentence = ""
    for word in words:
        if word not in string.punctuation or word in ["-", "("]:
            word = " " + word
        sentence += word
    sentence = sentence.replace(" '", "'").replace("( ", "(").strip()
    return sentence

def filter_sentence(
    sentences: List[str],
    condition: Optional[Callable[[str], bool]] = None,
) -> Tuple[List[str], List[bool]]:
    
    if not condition:
        condition = lambda sentence: len(sentence) > 1 and "#" not in sentence
    mask = list(map(condition, sentences))
    sentences = [sentence for m, sentence in zip(mask, sentences) if m]
    return sentences, mask

def process_token(token: str, replace: List[Tuple[str, str]] = [("#", "")]) -> str:
    replace_dict = {re.escape(k): v for (k, v) in replace}
    pattern = "|".join(replace_dict.keys())
    compiled_pattern = re.compile(pattern)
    replacement = lambda match: replace_dict[re.escape(match.group(0))]
    processed_token = compiled_pattern.sub(replacement, token)
    return processed_token

def mapping(entities: List[int], maps: List[int]) -> List[int]:
    f = lambda x: maps[x]
    return list(map(f, entities))

def merge_probs(
    probs: npt.NDArray["np.floating[T]"], maps: List[int]
) -> npt.NDArray["np.floating[T]"]:
    
    old_classes = probs.shape[1]
    map_size = np.max(maps) + 1
    probs_merged = np.zeros([len(probs), map_size], dtype=probs.dtype.type)

    for i in range(old_classes):
        if maps[i] >= 0:
            probs_merged[:, maps[i]] += probs[:, i]
    if -1 in maps:
        row_sums = probs_merged.sum(axis=1)
        probs_merged /= row_sums[:, np.newaxis]
    return probs_merged

def color_sentence(sentence: str, word: str) -> str:
    colored_word = colored(word, "red", force_color=True)
    return _replace_sentence(sentence=sentence, word=word, new_word=colored_word)

def _replace_sentence(sentence: str, word: str, new_word: str) -> str:

    new_sentence, number_of_substitions = re.subn(
        r"\b{}\b".format(re.escape(word)), new_word, sentence
    )
    if number_of_substitions == 0:
        # Use basic string manipulation if regex fails
        new_sentence = sentence.replace(word, new_word)
    return new_sentence
