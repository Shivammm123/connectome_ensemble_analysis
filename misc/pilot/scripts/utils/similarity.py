"""
Connectivity-based similarity utilities.

"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def build_connectivity_vectors(
    connections: pd.DataFrame,
    neuron_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build output connectivity matrix for a set of neurons.

    Each row is a neuron; each column is a possible postsynaptic target.
    Values are synapse counts (weight).

    Parameters
    ----------
    connections : pd.DataFrame
        Connectivity table with columns: source, target, weight
    neuron_ids : list of int
        Neurons to build vectors for (rows)

    Returns
    -------
    matrix : np.ndarray, shape (n_neurons, n_targets)
    target_ids : np.ndarray
        Ordered target neuron IDs (column labels)
    """
    target_ids = np.array(sorted(connections['target'].unique()))
    target_index = {t: i for i, t in enumerate(target_ids)}

    matrix = np.zeros((len(neuron_ids), len(target_ids)), dtype=np.float32)

    for row, nid in enumerate(neuron_ids):
        rows = connections[connections['source'] == nid]
        for _, r in rows.iterrows():
            col = target_index.get(r['target'])
            if col is not None:
                matrix[row, col] = r['weight']

    return matrix, target_ids


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity for rows of a matrix.

    Parameters
    ----------
    matrix : np.ndarray, shape (n, d)

    Returns
    -------
    np.ndarray, shape (n, n)
        Values in [-1, 1]; 1 = identical direction.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    safe_norms = np.where(norms == 0, 1.0, norms)
    normed = matrix / safe_norms
    sim = normed @ normed.T
    # Clip to handle floating-point drift
    return np.clip(sim, -1.0, 1.0)


def correlation_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Pearson correlation for rows of a matrix.

    Parameters
    ----------
    matrix : np.ndarray, shape (n, d)

    Returns
    -------
    np.ndarray, shape (n, n)
    """
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    return cosine_similarity_matrix(centered)


def find_top_similar(
    query_idx: int,
    sim_matrix: np.ndarray,
    neuron_ids: List[int],
    top_n: int = 20,
    threshold: float = 0.0,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """
    Return the top-N most similar neurons to a query neuron.

    Parameters
    ----------
    query_idx : int
        Row index of the query neuron in sim_matrix
    sim_matrix : np.ndarray
        Pairwise similarity matrix
    neuron_ids : list of int
        Neuron IDs corresponding to matrix rows
    top_n : int
        Maximum number of results
    threshold : float
        Minimum similarity score to include
    exclude_self : bool
        Exclude the query neuron itself

    Returns
    -------
    pd.DataFrame
        Columns: neuron_id, similarity — sorted descending
    """
    scores = sim_matrix[query_idx].copy()

    if exclude_self:
        scores[query_idx] = -np.inf

    above_threshold = np.where(scores >= threshold)[0]
    top_idx = above_threshold[np.argsort(scores[above_threshold])[::-1][:top_n]]

    return pd.DataFrame({
        'neuron_id': [neuron_ids[i] for i in top_idx],
        'similarity': scores[top_idx],
    })


def normalize_connectivity(matrix: np.ndarray, method: str = 'none') -> np.ndarray:
    """
    Optionally normalize the connectivity matrix before similarity.

    Parameters
    ----------
    matrix : np.ndarray
    method : str
        'none'   — raw synapse counts (default, per paper)
        'total'  — divide each row by its total synapse count
        'sqrt'   — square-root transform

    Returns
    -------
    np.ndarray
    """
    if method == 'none':
        return matrix.astype(np.float32)
    elif method == 'total':
        row_sums = matrix.sum(axis=1, keepdims=True)
        safe_sums = np.where(row_sums == 0, 1.0, row_sums)
        return (matrix / safe_sums).astype(np.float32)
    elif method == 'sqrt':
        return np.sqrt(matrix).astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization method: '{method}'. "
                         f"Choose from 'none', 'total', 'sqrt'.")


def compute_similarity(
    connections: pd.DataFrame,
    neuron_ids: List[int],
    method: str = 'cosine',
    normalization: str = 'none',
) -> Tuple[np.ndarray, List[int]]:
    """
    End-to-end similarity computation for a set of neurons.

    Parameters
    ----------
    connections : pd.DataFrame
        Filtered connectivity (source, target, weight)
    neuron_ids : list of int
        Neurons to compare
    method : str
        'cosine' or 'correlation'
    normalization : str
        Passed to normalize_connectivity

    Returns
    -------
    sim_matrix : np.ndarray, shape (n, n)
    neuron_ids : list of int
        Same order as matrix rows/columns
    """
    matrix, _ = build_connectivity_vectors(connections, neuron_ids)
    matrix = normalize_connectivity(matrix, normalization)

    if method == 'cosine':
        sim_matrix = cosine_similarity_matrix(matrix)
    elif method == 'correlation':
        sim_matrix = correlation_similarity_matrix(matrix)
    else:
        raise ValueError(f"Unknown similarity method: '{method}'. "
                         f"Choose from 'cosine', 'correlation'.")

    return sim_matrix, neuron_ids
