from typing import Any, Callable, Union
import numpy as np
import pandas as pd

LabelLike = Union[list, np.ndarray, pd.Series, pd.DataFrame]

DatasetLike = Any

###########################################################
###########################################################

FeatureArray = np.ndarray

Metric = Union[str, Callable]

