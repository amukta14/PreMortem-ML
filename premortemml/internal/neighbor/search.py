from __future__ import annotations
from typing import TYPE_CHECKING

from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:

    from premortemml.typing import Metric

def construct_knn(n_neighbors: int, metric: Metric, **knn_kwargs) -> NearestNeighbors:
    sklearn_knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, **knn_kwargs)

    return sklearn_knn
