#!/usr/bin/env python

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import f as fstat
from scipy.stats import t as tstat

from mne.fixes import _reshape_view, has_numba, jit
from mne.utils import (
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
# from ..parallel import parallel_func
# from ..source_estimate import MixedSourceEstimate, SourceEstimate, VolSourceEstimate
# from ..source_space import SourceSpaces
# from ..utils import (
#     ProgressBar,
#     _check_option,
#     _pl,
#     _validate_type,
#     check_random_state,
#     logger,
#     split_list,
#     verbose,
#     warn,
# )


def _find_clusters(
    x,
    threshold,
    tail=0,
    adjacency=None,
    max_step=1,
    include=None,
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
    sums : array
        Sum of x values in clusters.
    """
    _check_option("tail", tail, [-1, 0, 1])

    x = np.asanyarray(x)

    if not isinstance(threshold, dict):
        raise TypeError(
            "threshold must be a number, or a dict for "
            "threshold-free cluster enhancement"
        )
    if not all(key in threshold for key in ["start", "step"]):
        raise KeyError('threshold, if dict, must have at least "start" and "step"')
    tfce = True
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
                ) #partitions = None
                clusters += out[0]
                sums.append(out[1])
        # the score of each point is the sum of the h^H * e^E for each
        # supporting section "rectangle" h x e.
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
    # turn sums into array
    sums = scores

    return sums


def _find_clusters_1dir(x, x_in, adjacency, max_step, t_power, ndimage):
    """Actually call the clustering algorithm."""
    if adjacency is None:
        labels, n_labels = ndimage.label(x_in)

        if x.ndim == 1:
            # slices
            clusters = ndimage.find_objects(labels, n_labels)
            # equivalent to if len(clusters) == 0 but faster
            if not clusters:
                sums = list()
            else:
                index = list(range(1, n_labels + 1))
                if t_power == 1:
                    sums = ndimage.sum(x, labels, index=index)
                else:
                    sums = ndimage.sum(
                        np.sign(x) * np.abs(x) ** t_power, labels, index=index
                    )
        else:
            # boolean masks (raveled)
            clusters = list()
            sums = np.empty(n_labels)
            for label in range(n_labels):
                c = labels == label + 1
                clusters.append(c.ravel())
                if t_power == 1:
                    sums[label] = np.sum(x[c])
                else:
                    sums[label] = np.sum(np.sign(x[c]) * np.abs(x[c]) ** t_power)
    else:
        if x.ndim > 1:
            raise Exception(
                "Data should be 1D when using a adjacency to define clusters."
            )
        if isinstance(adjacency, sparse.spmatrix):
            adjacency = sparse.coo_array(adjacency)
        if sparse.issparse(adjacency) or adjacency is False:
            clusters = _get_components(x_in, adjacency)
        elif isinstance(adjacency, list):  # use temporal adjacency
            clusters = _get_clusters_st(x_in, adjacency, max_step)
        else:
            raise TypeError(
                f"adjacency must be a sparse array or list, got {type(adjacency)}"
            )
        if t_power == 1:
            sums = [_masked_sum(x, c) for c in clusters]
        else:
            sums = [_masked_sum_power(x, c, t_power) for c in clusters]

    return clusters, np.atleast_1d(sums)

@jit()
def _masked_sum(x, c):
    return np.sum(x[c])


@jit()
def _masked_sum_power(x, c, t_power):
    return np.sum(np.sign(x[c]) * np.abs(x[c]) ** t_power)


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


def _get_clusters_st(x_in, neighbors, max_step=1):
    """Choose the most efficient version."""
    n_src = len(neighbors)
    n_times = x_in.size // n_src
    cl_goods = np.where(x_in)[0]
    if len(cl_goods) > 0:
        keepers = [np.array([], dtype=int)] * n_times
        row, col = np.unravel_index(cl_goods, (n_times, n_src))
        lims = [0]
        if isinstance(row, int):
            row = [row]
            col = [col]
        else:
            order = np.argsort(row)
            row = row[order]
            col = col[order]
            lims += (np.where(np.diff(row) > 0)[0] + 1).tolist()
            lims.append(len(row))

        for start, end in zip(lims[:-1], lims[1:]):
            keepers[row[start]] = np.sort(col[start:end])
        if max_step == 1:
            return _get_clusters_st_1step(keepers, neighbors)
        else:
            return _get_clusters_st_multistep(keepers, neighbors, max_step)
    else:
        return []
    

def _get_clusters_st_1step(keepers, neighbors):
    """Directly calculate clusters.

    This uses knowledge that time points are
    only adjacent to immediate neighbors for data organized as time x space.

    This algorithm time increases linearly with the number of time points,
    compared to with the square for the standard (graph) algorithm.

    This algorithm creates clusters for each time point using a method more
    efficient than the standard graph method (but otherwise equivalent), then
    combines these clusters across time points in a reasonable way.
    """
    n_src = len(neighbors)
    n_times = len(keepers)
    # start cluster numbering at 1 for diffing convenience
    enum_offset = 1
    check = np.zeros((n_times, n_src), dtype=int)
    clusters = list()
    for ii, k in enumerate(keepers):
        c = _get_clusters_spatial(k, neighbors)
        for ci, cl in enumerate(c):
            check[ii, cl] = ci + enum_offset
        enum_offset += len(c)
        # give them the correct offsets
        c = [cl + ii * n_src for cl in c]
        clusters += c

    # now that each cluster has been assigned a unique number, combine them
    # by going through each time point
    for check1, check2, k in zip(check[:-1], check[1:], keepers[:-1]):
        # go through each one that needs reassignment
        inds = k[check2[k] - check1[k] > 0]
        check1_d = check1[inds]
        n = check2[inds]
        nexts = np.unique(n)
        for num in nexts:
            prevs = check1_d[n == num]
            base = np.min(prevs)
            for pr in np.unique(prevs[prevs != base]):
                _reassign(check1, clusters, base, pr)
            # reassign values
            _reassign(check2, clusters, base, num)
    # clean up clusters
    clusters = [cl for cl in clusters if len(cl) > 0]
    return clusters

def _get_clusters_spatial(s, neighbors):
    """Form spatial clusters using neighbor lists.

    This is equivalent to _get_components with n_times = 1, with a properly
    reconfigured adjacency matrix (formed as "neighbors" list)
    """
    # s is a vector of spatial indices that are significant, like:
    #     s = np.where(x_in)[0]
    # for x_in representing a single time-instant
    r = np.ones(s.shape, bool)
    clusters = list()
    next_ind = 0 if s.size > 0 else -1
    while next_ind >= 0:
        # put first point in a cluster, adjust remaining
        t_inds = [next_ind]
        r[next_ind] = 0
        icount = 1  # count of nodes in the current cluster
        while icount <= len(t_inds):
            ind = t_inds[icount - 1]
            # look across other vertices
            buddies = _get_buddies(r, s, neighbors[s[ind]])
            t_inds.extend(buddies)
            icount += 1
        next_ind = _where_first(r)
        clusters.append(s[t_inds])
    return clusters


def _get_clusters_st_multistep(keepers, neighbors, max_step=1):
    """Directly calculate clusters.

    This uses knowledge that time points are
    only adjacent to immediate neighbors for data organized as time x space.

    This algorithm time increases linearly with the number of time points,
    compared to with the square for the standard (graph) algorithm.
    """
    n_src = len(neighbors)
    n_times = len(keepers)
    t_border = list()
    t_border.append(0)
    for ki, k in enumerate(keepers):
        keepers[ki] = k + ki * n_src
        t_border.append(t_border[ki] + len(k))
    t_border = np.array(t_border)
    keepers = np.concatenate(keepers)
    v = keepers
    t, s = divmod(v, n_src)

    r = np.ones(t.shape, dtype=bool)
    clusters = list()
    inds = np.arange(t_border[0], t_border[n_times])
    next_ind = 0 if s.size > 0 else -1
    while next_ind >= 0:
        # put first point in a cluster, adjust remaining
        t_inds = [next_ind]
        r[next_ind] = False
        icount = 1  # count of nodes in the current cluster
        # look for significant values at the next time point,
        # same sensor, not placed yet, and add those
        while icount <= len(t_inds):
            ind = t_inds[icount - 1]
            selves = _get_selves(r, s, ind, inds, t, t_border, max_step)

            # look at current time point across other vertices
            these_inds = inds[t_border[t[ind]] : t_border[t[ind] + 1]]
            buddies = _get_buddies(r, s, neighbors[s[ind]], these_inds)

            t_inds += buddies + selves
            icount += 1
        next_ind = _where_first(r)
        clusters.append(v[t_inds])

    return clusters


def _reassign(check, clusters, base, num):
    """Reassign cluster numbers."""
    # reconfigure check matrix
    check[check == num] = base
    # concatenate new values into clusters array
    clusters[base - 1] = np.concatenate((clusters[base - 1], clusters[num - 1]))
    clusters[num - 1] = np.array([], dtype=int)



import numpy as np

# Small channels × time test data
x = np.array([
    [0.2, 0.4, 2.1, 2.5, 0.3, 0.1],
    [0.3, 2.2, 2.6, 2.8, 0.2, 0.1],
    [0.1, 0.5, 2.0, 2.3, 0.4, 0.2],
    [0.2, 0.1, 0.3, 0.4, 0.2, 0.1],
])

threshold = {
    "start": 0.0,
    "step": 0.5,
    "h_power": 2,
    "e_power": 0.5,
}

scores = _find_clusters(
    x=x,
    threshold=threshold,
    tail=1,
    adjacency=None,
    max_step=1,
    include=None,
    t_power=1,
    show_info=True,
)

print("Input x shape:", x.shape)
print("Output scores shape:", scores.shape)
print("TFCE scores flattened:")
print(scores)

print("TFCE scores reshaped to channels × time:")
print(scores.reshape(x.shape))