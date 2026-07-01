from __future__ import annotations

from warnings import warn

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components


# ============================================================
# Minimal TFCE implementation
# ============================================================

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
        If using sparse adjacency or adjacency=False, pass x.ravel().
    threshold : dict
        Example: {"start": 0.0, "step": 0.2, "h_power": 2, "e_power": 0.5}
    tail : -1 | 0 | 1
        1: positive clusters only; -1: negative clusters only; 0: both tails.
    adjacency : None | False | scipy sparse array/matrix
        None: use regular grid adjacency based on x.shape.
        False: no adjacency; each significant point is isolated.
        sparse: use graph adjacency; x must be 1D and adjacency must be x.size by x.size.
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
        raise TypeError("threshold must be a dict, e.g. {'start': 0, 'step': 0.2}")
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
    else:
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

        # TFCE height increment between current and previous threshold.
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
    """Find connected clusters from a boolean threshold mask."""
    x = np.asanyarray(x)
    mask = np.asarray(mask, dtype=bool)

    if adjacency is None:
        return _clusters_from_grid(mask)

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
        raise ValueError(f"adjacency shape must be {(x.size, x.size)}, got {adjacency.shape}")

    return _clusters_from_sparse_adjacency(mask, adjacency)


def _clusters_from_grid(mask):
    """Clusters for regular array geometry, e.g. a 2D channel-by-time grid."""
    labels, n_labels = ndimage.label(mask)
    return [(labels == label).ravel() for label in range(1, n_labels + 1)]


def _clusters_from_sparse_adjacency(mask, adjacency):
    """Clusters for graph adjacency, using only active/significant nodes."""
    adjacency = sparse.coo_array(adjacency)

    active_edges = np.logical_and(mask[adjacency.row], mask[adjacency.col])
    row = adjacency.row[active_edges]
    col = adjacency.col[active_edges]
    data = adjacency.data[active_edges]

    # Self-loops make isolated active nodes become valid 1-node clusters.
    active_nodes = np.where(mask)[0]
    row = np.concatenate([row, active_nodes])
    col = np.concatenate([col, active_nodes])
    data = np.concatenate([data, np.ones(active_nodes.size, dtype=float)])

    active_graph = sparse.coo_array((data, (row, col)), shape=adjacency.shape)
    _, components = connected_components(active_graph, directed=False)

    clusters = []
    for comp in np.unique(components[active_nodes]):
        cluster = np.where((components == comp) & mask)[0]
        if cluster.size:
            clusters.append(cluster)
    return clusters


def _cluster_size(cluster):
    cluster = np.asarray(cluster)
    if cluster.dtype == bool:
        return int(cluster.sum())
    return len(cluster)


# ============================================================
# MNE channel adjacency + time adjacency test data
# ============================================================

def make_mne_channel_adjacency(n_ch=32, montage="standard_1020", sfreq=512):
    """Create an EEG channel adjacency matrix using MNE channel locations."""
    import mne

    if n_ch == 32:
        ch_names = [
            "Fp1", "Fp2",
            "F7", "F3", "Fz", "F4", "F8",
            "FC5", "FC1", "FC2", "FC6",
            "T7", "C3", "Cz", "C4", "T8",
            "CP5", "CP1", "CP2", "CP6",
            "P7", "P3", "Pz", "P4", "P8",
            "PO9", "O1", "Oz", "O2", "PO10",
            "TP9", "TP10",
        ]
    elif n_ch == 64:
        ch_names = [
            "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7",
            "FT7", "FC5", "FC3", "FC1", "C1", "C3", "C5", "T7",
            "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
            "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz",
            "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8",
            "FT8", "FC6", "FC4", "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8",
            "TP8", "CP6", "CP4", "CP2", "P2", "P4", "P6", "P8", "P10",
            "PO8", "PO4", "O2",
        ]
    else:
        raise ValueError("Only 32- and 64-channel layouts are supported.")

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(montage)
    ch_adj, ch_names = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    ch_adj = sparse.coo_array(ch_adj)
    ch_adj.setdiag(0)
    ch_adj.eliminate_zeros()
    return info, ch_adj, list(ch_names)


def make_spatiotemporal_adjacency(ch_adj, n_times: int):
    """Combine channel adjacency with immediate-neighbor time adjacency.

    Data must be flattened from a time-by-channel array using x.ravel().
    Flat index is: flat_index = time_index * n_channels + channel_index.
    """
    ch_adj = sparse.coo_array(ch_adj)
    n_channels = ch_adj.shape[0]

    # Time adjacency: t is connected to t-1 and t+1.
    time_rows = np.arange(n_times - 1)
    time_cols = np.arange(1, n_times)
    time_adj = sparse.coo_array(
        (np.ones(n_times - 1), (time_rows, time_cols)),
        shape=(n_times, n_times),
    )
    time_adj = time_adj + time_adj.T

    # Same time, neighboring channels: I_time ⊗ ch_adj
    spatial_part = sparse.kron(sparse.eye(n_times, format="coo"), ch_adj, format="coo")

    # Same channel, neighboring times: time_adj ⊗ I_channels
    temporal_part = sparse.kron(time_adj, sparse.eye(n_channels, format="coo"), format="coo")

    adjacency = spatial_part + temporal_part
    adjacency = sparse.coo_array(adjacency)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


def make_test_data(ch_names, n_times=20, seed=42):
    """Create time-by-channel synthetic data with a clear C3/Cz/C4 cluster."""
    rng = np.random.default_rng(seed)
    n_channels = len(ch_names)
    x = rng.normal(0, 0.2, size=(n_times, n_channels))

    cluster_channels = ["C3", "Cz", "C4"]
    cluster_ch_idx = [ch_names.index(ch) for ch in cluster_channels]
    cluster_times = np.arange(8, 13)

    # Inject a strong positive spatio-temporal cluster.
    x[np.ix_(cluster_times, cluster_ch_idx)] += 4.0
    return x, cluster_channels, cluster_ch_idx, cluster_times


def run_mne_spatiotemporal_tfce_example(plot=False):
    info, ch_adj, ch_names = make_mne_channel_adjacency(n_ch=32)
    x, cluster_channels, cluster_ch_idx, cluster_times = make_test_data(ch_names, n_times=20)

    adjacency = make_spatiotemporal_adjacency(ch_adj, n_times=x.shape[0])
    threshold = {"start": 0.0, "step": 0.5, "h_power": 2, "e_power": 0.5}

    scores_flat = tfce_transform(
        x=x.ravel(),
        threshold=threshold,
        tail=1,
        adjacency=adjacency,
    )
    scores = scores_flat.reshape(x.shape)

    print("Input data shape, time × channels:", x.shape)
    print("Spatio-temporal adjacency shape:", adjacency.shape)
    print("TFCE scores shape:", scores.shape)
    print("Injected cluster channels:", cluster_channels)
    print("Injected cluster times:", cluster_times.tolist())

    print("\nTFCE scores around fake cluster:")
    for t in range(7, 14):
        vals = scores[t, cluster_ch_idx]
        print(f"time {t:02d}: {np.round(vals, 3)}")

    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        im0 = axes[0].imshow(x.T, aspect="auto", origin="lower")
        axes[0].set_title("Input statistic map")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Channel index")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(scores.T, aspect="auto", origin="lower")
        axes[1].set_title("TFCE scores")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Channel index")
        fig.colorbar(im1, ax=axes[1])
        plt.show()

    return x, scores, adjacency, ch_names, info


if __name__ == "__main__":
    run_mne_spatiotemporal_tfce_example(plot=True)
