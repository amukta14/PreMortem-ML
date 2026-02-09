from typing import Optional
import numpy as np

from premortemml.internal.constants import EPSILON

def softmax(
    x: np.ndarray, temperature: float = 1.0, axis: Optional[int] = None, shift: bool = False
) -> np.ndarray:
    
    x = x / max(temperature, EPSILON)
    if shift:
        x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
