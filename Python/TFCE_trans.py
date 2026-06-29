import mne
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, sparse

from ..fixes import jit
from ..utils import (
    ProgressBar,
    _check_option,
    _pl,
    _validate_type,
    check_random_state,
    logger,
    split_list,
    verbose,
    warn,
)

def ch_neg(n_ch=32, montage="standard_1020", sfreq=512, return_sparse=False):
    """
    Generate the EEG channel adjacency matrix.

    Parameters
    ----------
    n_ch : int
        Number of EEG channels (32 or 64).
    montage : str
        Name of the MNE montage. Default is "standard_1020".
    sfreq : float
        Sampling frequency (only required for creating the Info object).

    Returns
    -------
    ch_adjacency : scipy.sparse.csr_matrix
        Channel adjacency matrix.
    ch_names : list of str
        Channel names corresponding to the adjacency matrix.
    """

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

    elif n_ch == 64:
        ch_names = [
            'Fp1','AF7','AF3','F1','F3','F5','F7',
            'FT7','FC5','FC3','FC1','C1','C3','C5','T7',
            'TP7','CP5','CP3','CP1','P1','P3','P5','P7',
            'P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz',
            'Fpz','Fp2','AF8','AF4','AFz','Fz','F2','F4','F6','F8',
            'FT8','FC6','FC4','FC2','FCz','Cz','C2','C4','C6','T8',
            'TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10',
            'PO8','PO4','O2'
        ]

    else:
        raise ValueError("Only 32- and 64-channel layouts are currently supported.")

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



def tfce2d(ch_adj, n_times = 10):
    adj = mne.stats.combine_adjacency(n_times, ch_adj)

def _get_components(x_in, adjacency, return_list=True):
    """Get connected components from a mask and a adjacency matrix."""
    if adjacency is False:
        components = np.arange(len(x_in))
    else:
        mask = np.logical_and(x_in[adjacency.row], x_in[adjacency.col])
        data = adjacency.data[mask]
        row = adjacency.row[mask]
        col = adjacency.col[mask]
        shape = adjacency.shape
        idx = np.where(x_in)[0]
        row = np.concatenate((row, idx))
        col = np.concatenate((col, idx))
        data = np.concatenate((data, np.ones(len(idx), dtype=data.dtype)))
        adjacency = sparse.coo_array((data, (row, col)), shape=shape)
        _, components = connected_components(adjacency)
    if return_list:
        start = np.min(components)
        stop = np.max(components)
        comp_list = [list() for i in range(start, stop + 1, 1)]
        mask = np.zeros(len(comp_list), dtype=bool)
        for ii, comp in enumerate(components):
            comp_list[comp].append(ii)
            mask[comp] += x_in[ii]
        clusters = [np.array(k) for k, m in zip(comp_list, mask) if m]
        return clusters
    else:
        return components

@jit()
def _masked_sum(x, c):
    return np.sum(x[c])

@jit()
def _masked_sum_power(x, c, t_power):
    return np.sum(np.sign(x[c]) * np.abs(x[c]) ** t_power)

def _find_clusters_1dir(x, x_in, adjacency, max_step, t_power, ndimage):
    """Actually call the clustering algorithm."""
    if x.ndim > 1:
        raise Exception(
            "Data should be 1D when using a adjacency to define clusters."
        )
    if isinstance(adjacency, sparse.spmatrix):
        adjacency = sparse.coo_array(adjacency)

    if sparse.issparse(adjacency) or adjacency is False:
        clusters = _get_components(x_in, adjacency)
    # elif isinstance(adjacency, list):  # use temporal adjacency
    #     clusters = _get_clusters_st(x_in, adjacency, max_step)
    else:
        raise TypeError(
            f"adjacency must be a sparse array or list, got {type(adjacency)}"
        )
    if t_power == 1:
        sums = [_masked_sum(x, c) for c in clusters]
    else:
        sums = [_masked_sum_power(x, c, t_power) for c in clusters]

    return clusters, np.atleast_1d(sums)



def _find_clusters(
    x,
    threshold,
    tail=0,
    adjacency=None,
    max_step=1,
    include=None,
    partitions=None,
    t_power=1,
    show_info=False,
):
    """Find all clusters which are above/below a certain threshold.

    When doing a two-tailed test (tail == 0), only points with the same
    sign will be clustered together.

    Parameters
    ----------
    x : 1D array
        Data
    threshold : float | dict
        Where to threshold the statistic. Should be negative for tail == -1,
        and positive for tail == 0 or 1. Can also be an dict for
        threshold-free cluster enhancement.
    tail : -1 | 0 | 1
        Type of comparison
    adjacency : scipy.sparse.coo_array, None, or list
        Defines adjacency between features. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        If adjacency is a list, it is assumed that each entry stores the
        indices of the spatial neighbors in a spatio-temporal dataset x.
        Default is None, i.e, a regular lattice adjacency.
        False means no adjacency.
    max_step : int
        If adjacency is a list, this defines the maximal number of steps
        between vertices along the second dimension (typically time) to be
        considered adjacent.
    include : 1D bool array or None
        Mask to apply to the data of points to cluster. If None, all points
        are used.
    partitions : array of int or None
        An array (same size as X) of integers indicating which points belong
        to each partition.
    t_power : float
        Power to raise the statistical values (usually t-values) by before
        summing (sign will be retained). Note that t_power == 0 will give a
        count of nodes in each cluster, t_power == 1 will weight each node by
        its statistical score.
    show_info : bool
        If True, display information about thresholds used (for TFCE). Should
        only be done for the standard permutation.

    Returns
    -------
    clusters : list of slices or list of arrays (boolean masks)
        We use slices for 1D signals and mask to multidimensional
        arrays. None is returned if threshold is a dict (TFCE)
    sums : array
        Sum of x values in clusters.
    """
    _check_option("tail", tail, [-1, 0, 1])

    x = np.asanyarray(x)

    # if not np.isscalar(threshold):
    if not isinstance(threshold, dict):
        raise TypeError(
            "threshold must be a number, or a dict for "
            "threshold-free cluster enhancement"
        )
    if not all(key in threshold for key in ["start", "step"]):
        raise KeyError('threshold, if dict, must have at least "start" and "step"')
        
    # tfce = True
    use_x = x[np.isfinite(x)]
    if use_x.size == 0:
        raise RuntimeError(
            "No finite values found in the observed statistic values"
        )
    if tail == -1:
        if threshold["start"] > 0:
            raise ValueError('threshold["start"] must be <= 0 for tail == -1')
        if threshold["step"] >= 0:
            raise ValueError('threshold["step"] must be < 0 for tail == -1')
        stop = np.min(use_x)
    elif tail == 1:
        stop = np.max(use_x)
    else:  # tail == 0
        stop = max(np.max(use_x), -np.min(use_x))
    del use_x
    thresholds = np.arange(threshold["start"], stop, threshold["step"], float)
    h_power = threshold.get("h_power", 2)
    e_power = threshold.get("e_power", 0.5)
    if show_info is True:
        if len(thresholds) == 0:
            warn(
                f'threshold["start"] ({threshold["start"]}) is more extreme '
                f"than data statistics with most extreme value {stop}"
            )
        else:
            logger.info(
                "Using %d thresholds from %0.2f to %0.2f for TFCE "
                "computation (h_power=%0.2f, e_power=%0.2f)",
                len(thresholds),
                thresholds[0],
                thresholds[-1],
                h_power,
                e_power,
            )
    scores = np.zeros(x.size)

    # include all points by default
    if include is None:
        include = np.ones(x.shape, dtype=bool)

    if tail in [0, 1] and not np.all(np.diff(thresholds) > 0):
        raise ValueError("Thresholds must be monotonically increasing")
    if tail == -1 and not np.all(np.diff(thresholds) < 0):
        raise ValueError("Thresholds must be monotonically decreasing")

    # set these here just in case thresholds == []
    clusters = list()
    sums = list()
    for ti, thresh in enumerate(thresholds):
        # these need to be reset on each run
        clusters = list()
        if tail == 0:
            x_ins = [
                np.logical_and(x > thresh, include),
                np.logical_and(x < -thresh, include),
            ]
        elif tail == -1:
            x_ins = [np.logical_and(x < thresh, include)]
        else:  # tail == 1
            x_ins = [np.logical_and(x > thresh, include)]
        # loop over tails
        for x_in in x_ins:
            if np.any(x_in):
                out = _find_clusters_1dir(
                    x, x_in, adjacency, max_step, t_power, ndimage
                )
                clusters += out[0]
                sums.append(out[1])

        if ti == 0:
            h = abs(thresh)
        else:
            h = abs(thresh - thresholds[ti - 1])
        h = h**h_power
        for c in clusters:
            # triage based on cluster storage type
            if isinstance(c, slice):
                len_c = c.stop - c.start
            elif isinstance(c, tuple):
                len_c = len(c)
            elif c.dtype == np.dtype(bool):
                len_c = np.sum(c)
            else:
                len_c = len(c)
            scores[c] += h * (len_c**e_power)

    sums = scores
    clusters = None  # clusters construction is made in _permutation_cluster_test

    return clusters, sums


def main():
    n_ch = 32

    info, ch_adj, ch_names = ch_neg(n_ch=n_ch, return_sparse=True)
    print(ch_adj, ch_names)
    print(type(ch_adj))

    # fig = mne.viz.plot_ch_adjacency(
    #     info,
    #     ch_adj,
    #     ch_names,
    #     show = False)
    # plt.show()

    mne.viz.plot_sensors(
        info,
        kind="topomap",
        show_names=True
        )
    plt.show()

    print(f"Number of channels: {len(ch_names)}")
    print(f"First 10 channels: {ch_names[:10]}")


if __name__ == "__main__":
    main()
# def tfce_2d(stat_map, dh = )
# epochs = mne.Epochs(
#     raw,
#     events,
#     event_id,
#     tmin,
#     tmax,
#     picks=picks,
#     decim=2,  # just for speed!
#     baseline=None,
#     reject=reject,
#     preload=True,
# )

# # Obtain the data as a 3D matrix and transpose it such that
# # the dimensions are as expected for the cluster permutation test:
# # n_epochs × n_times × n_channels
# X = [epochs[event_name].get_data(copy=False) for event_name in event_id]
# X = [np.transpose(x, (0, 2, 1)) for x in X]

# adjacency, ch_names = find_ch_adjacency(epochs.info, ch_type="mag")


# # We are running an F test, so we look at the upper tail
# # see also: https://stats.stackexchange.com/a/73993
# tail = 1

# # We want to set a critical test statistic (here: F), to determine when
# # clusters are being formed. Using Scipy's percent point function of the F
# # distribution, we can conveniently select a threshold that corresponds to
# # some alpha level that we arbitrarily pick.
# alpha_cluster_forming = 0.001

# # For an F test we need the degrees of freedom for the numerator
# # (number of conditions - 1) and the denominator (number of observations
# # - number of conditions):
# n_conditions = len(event_id)
# n_observations = len(X[0])
# dfn = n_conditions - 1
# dfd = n_observations - n_conditions

# # Note: we calculate 1 - alpha_cluster_forming to get the critical value
# # on the right tail
# f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)

# # run the cluster based permutation analysis
# cluster_stats = spatio_temporal_cluster_test(
#     X,
#     n_permutations=1000,
#     threshold=f_thresh,
#     tail=tail,
#     n_jobs=None,
#     buffer_size=None,
#     adjacency=adjacency,
# )
# F_obs, clusters, p_values, _ = cluster_stats