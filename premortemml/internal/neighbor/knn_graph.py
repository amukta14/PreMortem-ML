from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import circulant
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from premortemml.typing import FeatureArray, Metric

from premortemml.internal.neighbor.metric import decide_default_metric
from premortemml.internal.neighbor.search import construct_knn

DEFAULT_K = 10

def features_to_knn(
    features: Optional[FeatureArray],
    *,
    n_neighbors: Optional[int] = None,
    metric: Optional[Metric] = None,
    **sklearn_knn_kwargs,
) -> NearestNeighbors:
    
    if features is None:
        raise ValueError("Both knn and features arguments cannot be None at the same time.")
    # Use provided metric if available, otherwise decide based on the features.
    metric = metric or decide_default_metric(features)

    # Decide the number of neighbors to use in the KNN search.
    n_neighbors = _configure_num_neighbors(features, n_neighbors)

    knn = construct_knn(n_neighbors, metric, **sklearn_knn_kwargs)
    return knn.fit(features)

def construct_knn_graph_from_index(
    knn: NearestNeighbors,
    correction_features: Optional[FeatureArray] = None,
) -> csr_matrix:
    

    # Perform self-querying to get the distances and indices of the nearest neighbors
    distances, indices = knn.kneighbors(X=None, return_distance=True)

    # Correct the distances and indices if the correction_features array is provided
    if correction_features is not None:
        distances, indices = correct_knn_distances_and_indices(
            features=correction_features, distances=distances, indices=indices
        )

    N, K = distances.shape

    # Pointers to the row elements distances[indptr[i]:indptr[i+1]],
    # and their corresponding column indices indices[indptr[i]:indptr[i+1]].
    indptr = np.arange(0, N * K + 1, K)

    return csr_matrix((distances.reshape(-1), indices.reshape(-1), indptr), shape=(N, N))

def create_knn_graph_and_index(
    features: Optional[FeatureArray],
    *,
    n_neighbors: Optional[int] = None,
    metric: Optional[Metric] = None,
    correct_exact_duplicates: bool = True,
    **sklearn_knn_kwargs,
) -> Tuple[csr_matrix, NearestNeighbors]:
    
    # Construct NearestNeighbors object
    knn = features_to_knn(features, n_neighbors=n_neighbors, metric=metric, **sklearn_knn_kwargs)
    # Build graph from NearestNeighbors object
    knn_graph = construct_knn_graph_from_index(knn)

    # Ensure that exact duplicates found with np.unique aren't accidentally missed in the KNN graph
    if correct_exact_duplicates:
        assert features is not None
        knn_graph = correct_knn_graph(features, knn_graph)
    return knn_graph, knn

def correct_knn_graph(features: FeatureArray, knn_graph: csr_matrix) -> csr_matrix:
    N = features.shape[0]
    distances, indices = knn_graph.data.reshape(N, -1), knn_graph.indices.reshape(N, -1)

    corrected_distances, corrected_indices = correct_knn_distances_and_indices(
        features, distances, indices
    )
    N = features.shape[0]
    return csr_matrix(
        (corrected_distances.reshape(-1), corrected_indices.reshape(-1), knn_graph.indptr),
        shape=(N, N),
    )

def _compute_exact_duplicate_sets(features: FeatureArray) -> List[np.ndarray]:
    # Use np.unique to catch inverse indices of all unique feature sets
    _, unique_inverse, unique_counts = np.unique(
        features, return_inverse=True, return_counts=True, axis=0
    )

    # Collect different sets of exact duplicates in the dataset
    exact_duplicate_sets = [
        np.where(unique_inverse == u)[0] for u in set(unique_inverse) if unique_counts[u] > 1
    ]

    return exact_duplicate_sets

def correct_knn_distances_and_indices_with_exact_duplicate_sets_inplace(
    distances: np.ndarray,
    indices: np.ndarray,
    exact_duplicate_sets: List[np.ndarray],
) -> None:
    

    # Number of neighbors
    k = distances.shape[1]

    for duplicate_inds in exact_duplicate_sets:
        # Determine the number of same points to include, respecting the limit of k
        num_same = len(duplicate_inds)
        num_same_included = min(num_same - 1, k)  # ensure we do not exceed k neighbors

        sorted_first_k_duplicate_inds = _prepare_neighborhood_of_first_k_duplicates(
            duplicate_inds, num_same_included
        )

        if num_same >= k + 1:
            # All nearest neighbors are exact duplicates

            # We only pass in the ciruclant matrix of nearest neighbors
            indices[duplicate_inds[: k + 1]] = sorted_first_k_duplicate_inds
            # But the rest will just take the k first duplicate ids
            indices[duplicate_inds[k + 1 :]] = duplicate_inds[:k]

            # Finally, set the distances between exact duplicates to zero
            distances[duplicate_inds] = 0
        else:
            # Some of the nearest neighbors aren't exact duplicates, move those to the back

            # Get indices and distances from knn that are not the same as i
            different_point_mask = np.isin(indices[duplicate_inds], duplicate_inds, invert=True)

            # Get the indices of the first m True values in each row of the mask
            true_indices = np.argsort(~different_point_mask, axis=1)[:, :-num_same_included]

            # Copy the values to the last m columns in dists
            distances[duplicate_inds, -(k - num_same_included) :] = distances[
                duplicate_inds, true_indices.T
            ].T
            indices[duplicate_inds, -(k - num_same_included) :] = indices[
                duplicate_inds, true_indices.T
            ].T

            # We can pass the circulant matrix to a slice
            indices[duplicate_inds, :num_same_included] = sorted_first_k_duplicate_inds

            # Finally, set the distances between exact duplicates to zero
            distances[duplicate_inds, :num_same_included] = 0

    return None

def _prepare_neighborhood_of_first_k_duplicates(duplicate_inds, num_same_included):
    circulant_base = duplicate_inds[: num_same_included + 1]
    circulant_matrix = circulant(circulant_base)
    sliced_circulant_matrix = circulant_matrix[:, 1:]
    sorted_first_k_duplicate_inds = np.sort(sliced_circulant_matrix, axis=1)
    return sorted_first_k_duplicate_inds

def correct_knn_distances_and_indices(
    features: FeatureArray,
    distances: np.ndarray,
    indices: np.ndarray,
    exact_duplicate_sets: Optional[List[np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    

    if exact_duplicate_sets is None:
        exact_duplicate_sets = _compute_exact_duplicate_sets(features)

    # Prepare the output arrays
    corrected_distances = np.copy(distances)
    corrected_indices = np.copy(indices)

    correct_knn_distances_and_indices_with_exact_duplicate_sets_inplace(
        distances=corrected_distances,
        indices=corrected_indices,
        exact_duplicate_sets=exact_duplicate_sets,
    )

    return corrected_distances, corrected_indices

def _configure_num_neighbors(features: FeatureArray, k: Optional[int]):
    # Error if the provided value is greater or equal to the number of examples.
    N = features.shape[0]
    k_larger_than_dataset = k is not None and k >= N
    if k_larger_than_dataset:
        raise ValueError(
            f"Number of nearest neighbors k={k} cannot exceed the number of examples N={len(features)} passed into the estimator (knn)."
        )

    # Either use the provided value or select a default value based on the feature array size.
    k = k or min(DEFAULT_K, N - 1)
    return k
