from typing import Callable, Optional, Union

import numpy as np
from scipy.sparse import csr_matrix

from premortemml.internal.neighbor.knn_graph import create_knn_graph_and_index

def _knn_shapley_score(neighbor_indices: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    N = y.shape[0]
    scores = np.zeros((N, N))

    for y_alpha, s_alpha, idx in zip(y, scores, neighbor_indices):
        y_neighbors = y[idx]
        ans_matches = (y_neighbors == y_alpha).flatten()
        for j in range(k - 2, -1, -1):
            s_alpha[idx[j]] = s_alpha[idx[j + 1]] + float(
                int(ans_matches[j]) - int(ans_matches[j + 1])
            )
    return np.mean(scores / k, axis=0)

def data_shapley_knn(
    labels: np.ndarray,
    *,
    features: Optional[np.ndarray] = None,
    knn_graph: Optional[csr_matrix] = None,
    metric: Optional[Union[str, Callable]] = None,
    k: int = 10,
) -> np.ndarray:
    
    if knn_graph is None and features is None:
        raise ValueError("Either knn_graph or features must be provided.")

    # Use provided knn_graph or compute it from features
    if knn_graph is None:
        knn_graph, _ = create_knn_graph_and_index(features, n_neighbors=k, metric=metric)

    num_examples = labels.shape[0]
    distances = knn_graph.indices.reshape(num_examples, -1)
    scores = _knn_shapley_score(neighbor_indices=distances, y=labels, k=k)
    return 0.5 * (scores + 1)
