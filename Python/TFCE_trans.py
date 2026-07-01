"""Simplified standalone TFCE transformation code.

Dependencies:
    pip install numpy scipy

This keeps only the parts needed to compute TFCE scores from a statistic map.
It supports:
    1. adjacency=None   -> regular grid/lattice clustering using scipy.ndimage
    2. adjacency=False  -> no neighbors; each significant point is its own cluster
    3. sparse adjacency -> graph clustering; x must be 1D and adjacency shape must match x.size
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from warnings import warn


def _check_tail(tail: int) -> None:
    if tail not in (-1, 0, 1):
        raise ValueError("tail must be one of -1, 0, or 1")


def tfce_transform(
    x,
    threshold: dict,
    tail: int = 1,
    adjacency=None,
    include=None,
    h_power: float | None = None,
    e_power: float | None = None,
):
    """Compute TFCE scores for a statistic array.

    Parameters
    ----------
    x : array-like
        Statistic values. Can be 1D, 2D, or ND if adjacency=None.
        If using sparse adjacency, x must be 1D.
    threshold : dict
        Example: {"start": 0.0, "step": 0.2, "h_power": 2, "e_power": 0.5}
    tail : -1 | 0 | 1
        1: positive clusters only; -1: negative clusters only; 0: both tails.
    adjacency : None | False | scipy sparse array/matrix
        None: use array-grid adjacency.
        False: no adjacency; each significant point is isolated.
        sparse: use graph adjacency; x must be 1D.
    include : bool array | None
        Optional mask of points allowed to be clustered. Must match x.shape.
    h_power, e_power : float | None
        Optional override for threshold["h_power"] and threshold["e_power"].

    Returns
    -------
    scores : ndarray
        TFCE scores with the same shape as x.
    """
    _check_tail(tail)
    x = np.asanyarray(x)

    if not isinstance(threshold, dict):
        raise TypeError("threshold must be a dict for TFCE, e.g. {'start': 0, 'step': 0.2}")
    if "start" not in threshold or "step" not in threshold:
        raise KeyError('threshold must contain "start" and "step"')

    if include is None:
        include = np.ones(x.shape, dtype=bool)
    else:
        include = np.asarray(include, dtype=bool)
        if include.shape != x.shape:
            raise ValueError("include must have the same shape as x")

    finite_x = x[np.isfinite(x)]
    if finite_x.size == 0:
        raise RuntimeError("No finite values found in x")

    start = float(threshold["start"])
    step = float(threshold["step"])
    h_power = threshold.get("h_power", 2.0) if h_power is None else h_power
    e_power = threshold.get("e_power", 0.5) if e_power is None else e_power

    if tail == -1:
        if start > 0:
            raise ValueError('threshold["start"] must be <= 0 for tail=-1')
        if step >= 0:
            raise ValueError('threshold["step"] must be < 0 for tail=-1')
        stop = np.min(finite_x)
    elif tail == 1:
        if step <= 0:
            raise ValueError('threshold["step"] must be > 0 for tail=1')
        stop = np.max(finite_x)
    else:  # two-tailed
        if start < 0:
            raise ValueError('threshold["start"] should be >= 0 for tail=0')
        if step <= 0:
            raise ValueError('threshold["step"] must be > 0 for tail=0')
        stop = max(np.max(finite_x), -np.min(finite_x))

    thresholds = np.arange(start, stop, step, dtype=float)
    if thresholds.size == 0:
        warn("No thresholds were generated; returning zeros")
        return np.zeros_like(x, dtype=float)

    scores = np.zeros(x.size, dtype=float)

    for ti, thresh in enumerate(thresholds):
        if tail == 0:
            masks = [
                np.logical_and(x > thresh, include),
                np.logical_and(x < -thresh, include),
            ]
        elif tail == 1:
            masks = [np.logical_and(x > thresh, include)]
        else:
            masks = [np.logical_and(x < thresh, include)]

        # Height increment between current and previous threshold.
        if ti == 0:
            h = abs(thresh)
        else:
            h = abs(thresh - thresholds[ti - 1])
        h = h ** h_power

        for mask in masks:
            if not np.any(mask):
                continue
            clusters = find_clusters_one_direction(x, mask, adjacency)
            for cluster in clusters:
                cluster_size = _cluster_size(cluster)
                scores[cluster] += h * (cluster_size ** e_power)

    return scores.reshape(x.shape)


def find_clusters_one_direction(x, mask, adjacency=None):
    """Find connected clusters from a boolean mask."""
    x = np.asanyarray(x)
    mask = np.asarray(mask, dtype=bool)

    if adjacency is None:
        return _clusters_from_grid(mask)

    # Sparse graph or no-adjacency mode uses flattened indexing.
    if x.ndim > 1:
        raise ValueError("When using sparse adjacency or adjacency=False, pass x.ravel()")

    mask = mask.ravel()

    if adjacency is False:
        return [np.array([idx], dtype=int) for idx in np.where(mask)[0]]

    if isinstance(adjacency, sparse.spmatrix):
        adjacency = sparse.coo_array(adjacency)
    if not sparse.issparse(adjacency):
        raise TypeError("adjacency must be None, False, or a scipy sparse array/matrix")
    if adjacency.shape != (x.size, x.size):
        raise ValueError("adjacency shape must be (x.size, x.size)")

    return _clusters_from_sparse_adjacency(mask, adjacency)


def _clusters_from_grid(mask):
    """Clusters for regular array geometry, e.g. 2D channel × time grid."""
    labels, n_labels = ndimage.label(mask)
    clusters = []
    for label in range(1, n_labels + 1):
        clusters.append((labels == label).ravel())
    return clusters


def _clusters_from_sparse_adjacency(mask, adjacency):
    """Clusters for graph adjacency using only active/significant nodes."""
    adjacency = sparse.coo_array(adjacency)

    # Keep only edges where both endpoints are significant.
    active_edges = np.logical_and(mask[adjacency.row], mask[adjacency.col])
    row = adjacency.row[active_edges]
    col = adjacency.col[active_edges]
    data = adjacency.data[active_edges]

    # Add self-loops for active nodes, so isolated active nodes become clusters.
    active_nodes = np.where(mask)[0]
    row = np.concatenate([row, active_nodes])
    col = np.concatenate([col, active_nodes])
    data = np.concatenate([data, np.ones(active_nodes.size, dtype=adjacency.data.dtype)])

    active_graph = sparse.coo_array((data, (row, col)), shape=adjacency.shape)
    _, components = connected_components(active_graph, directed=False)

    clusters = []
    for comp in np.unique(components[active_nodes]):
        cluster = np.where((components == comp) & mask)[0]
        if cluster.size:
            clusters.append(cluster)
    return clusters


def _cluster_size(cluster):
    """Return number of points in a cluster, handling different index types."""
    if isinstance(cluster, slice):
        return cluster.stop - cluster.start
    if isinstance(cluster, tuple):
        return len(cluster[0])
    cluster = np.asarray(cluster)
    if cluster.dtype == bool:
        return int(cluster.sum())
    return len(cluster)


if __name__ == "__main__":
    # Example 1: 2D channel × time map, regular grid adjacency.
    x_2d = np.array([
        [0.2, 0.4, 2.1, 2.5, 0.3, 0.1],
        [0.3, 2.2, 2.6, 2.8, 0.2, 0.1],
        [0.1, 0.5, 2.0, 2.3, 0.4, 0.2],
        [0.2, 0.1, 0.3, 0.4, 0.2, 0.1],
    ])

    threshold = {"start": 0.0, "step": 0.5, "h_power": 2, "e_power": 0.5}
    scores_2d = tfce_transform(x_2d, threshold, tail=1, adjacency=None)

    print("2D input:")
    print(x_2d)
    print("\n2D TFCE scores:")
    print(np.round(scores_2d, 3))

    # Example 2: 1D data with sparse chain adjacency.
    x_1d = np.array([0.2, 2.1, 2.5, 2.3, 0.4, 1.8, 0.1])
    row = np.array([0, 1, 2, 3, 4, 5])
    col = np.array([1, 2, 3, 4, 5, 6])
    data = np.ones(row.size)
    adjacency = sparse.coo_array((data, (row, col)), shape=(x_1d.size, x_1d.size))
    adjacency = adjacency + adjacency.T

    scores_1d = tfce_transform(x_1d, threshold, tail=1, adjacency=adjacency)
    print("\n1D input:")
    print(x_1d)
    print("\n1D TFCE scores with sparse adjacency:")
    print(np.round(scores_1d, 3))


import numpy as np
import mne

# ----------------------------
# Create MNE channel adjacency
# ----------------------------
def ch_neg(n_ch=32, montage="standard_1020", sfreq=512):
    if n_ch == 32:
        ch_names = [
            'Fp1','Fp2',
            'F7','F3','Fz','F4','F8',
            'FC5','FC1','FC2','FC6',
            'T7','C3','Cz','C4','T8',
            'CP5','CP1','CP2','CP6',
            'P7','P3','Pz','P4','P8',
            'PO9','O1','Oz','O2','PO10',
            'TP9','TP10'
        ]
    else:
        raise ValueError("This test example uses 32 channels only.")

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types="eeg",
    )
    info.set_montage(montage)

    ch_adj, ch_names = mne.channels.find_ch_adjacency(
        info,
        ch_type="eeg",
    )

    return info, ch_adj, ch_names


# ----------------------------
# Convert sparse adjacency to neighbor list
# ----------------------------
def sparse_to_neighbor_list(ch_adj):
    ch_adj = ch_adj.tocsr()
    neighbors = []

    for ch_idx in range(ch_adj.shape[0]):
        neigh = ch_adj[ch_idx].indices
        neigh = neigh[neigh != ch_idx]  # remove self-neighbor if present
        neighbors.append(neigh)

    return neighbors


# ----------------------------
# Make test data: time × channels
# ----------------------------
info, ch_adj, ch_names = ch_neg(n_ch=32)

neighbors = sparse_to_neighbor_list(ch_adj)

n_times = 20
n_channels = len(ch_names)

rng = np.random.default_rng(42)
x = rng.normal(0, 0.2, size=(n_times, n_channels))

# Inject a fake spatio-temporal cluster
cluster_channels = ["C3", "Cz", "C4"]
cluster_ch_idx = [ch_names.index(ch) for ch in cluster_channels]

x[8:13, cluster_ch_idx] += 4.0

threshold = {
    "start": 0.0,
    "step": 0.5,
    "h_power": 2,
    "e_power": 0.5,
}

# ----------------------------
# Run your simplified TFCE
# ----------------------------
scores_flat = _find_clusters(
    x=x.ravel(),              # important: flattened time × channels
    threshold=threshold,
    tail=1,
    adjacency=neighbors,      # channel adjacency list
    max_step=1,               # adjacent time points connected
    include=None,
    t_power=1,
    show_info=True,
)

scores = scores_flat.reshape(x.shape)

print("Input data shape:", x.shape)
print("TFCE scores shape:", scores.shape)

print("\nCluster channels:")
print(cluster_channels)

print("\nTFCE scores around fake cluster:")
for t in range(7, 14):
    vals = scores[t, cluster_ch_idx]
    print(f"time {t}: {vals}")