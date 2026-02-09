from scipy.spatial.distance import euclidean

from premortemml.typing import FeatureArray, Metric

HIGH_DIMENSION_CUTOFF: int = 3

ROW_COUNT_CUTOFF: int = 100

# Metric decision functions
def _euclidean_large_dataset() -> str:
    return "euclidean"

def _euclidean_small_dataset() -> Metric:
    return euclidean

def _cosine_metric() -> str:
    return "cosine"

def decide_euclidean_metric(features: FeatureArray) -> Metric:
    num_rows = features.shape[0]
    if num_rows > ROW_COUNT_CUTOFF:
        return _euclidean_large_dataset()
    else:
        return _euclidean_small_dataset()

# Main function to decide the metric
def decide_default_metric(features: FeatureArray) -> Metric:
    if features.shape[1] > HIGH_DIMENSION_CUTOFF:
        return _cosine_metric()
    return decide_euclidean_metric(features)
