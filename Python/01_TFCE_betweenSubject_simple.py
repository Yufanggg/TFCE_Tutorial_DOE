import numpy as np
import scipy.io as sio
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# Load MATLAB data
# ==========================================================

mat = sio.loadmat("../data/01_simulated_between_subject_EEG.mat")

data = mat["data"]          # subjects x channels x time
group = mat["group"].ravel()  # -1 = Control, 1 = Treatment
times = mat["times"].ravel()

n_sub, n_chan, n_time = data.shape

# ==========================================================
# Function: compute regression t-map
# Model: EEG ~ group
# ==========================================================

def compute_tmap_lm(data, predictor):
    n_sub, n_chan, n_time = data.shape

    predictor = np.asarray(predictor).ravel()
    X = sm.add_constant(predictor)

    t_map = np.zeros((n_chan, n_time))

    for ch in range(n_chan):
        for tp in range(n_time):
            y = data[:, ch, tp]

            model = sm.OLS(y, X).fit()

            # coefficient 1 = group effect
            t_map[ch, tp] = model.tvalues[1]

    return t_mapimport numpy as np
from collections import deque

def tfce_2d(stat_map, chan_neighbors, dh=0.1, e_power=2/3, h_power=2):
    """
    TFCE transformation for EEG channel x time statistical maps.

    Parameters
    ----------
    stat_map : array, shape (n_channels, n_times)
        Observed statistical map, e.g. t-values.

    chan_neighbors : array, shape (n_channels, n_channels)
        Channel adjacency matrix.
        chan_neighbors[i, j] = 1 if channels i and j are neighbors.

    dh : float
        Step size for numerical integration over thresholds.

    e_power : float
        Extent exponent E. Default = 2/3.

    h_power : float
        Height exponent H. Default = 2.

    Returns
    -------
    tfce_map : array, shape (n_channels, n_times)
        TFCE-enhanced statistical map.
    """

    n_chan, n_time = stat_map.shape

    abs_map = np.abs(stat_map)
    sign_map = np.sign(stat_map)

    max_h = np.nanmax(abs_map)
    thresholds = np.arange(0, max_h + dh, dh)

    scores = np.zeros((n_chan, n_time))

    for ti, thresh in enumerate(thresholds):

        if ti == 0:
            h = abs(thresh)
        else:
            h = abs(thresh - thresholds[ti - 1])

        h = h ** h_power

        supra = abs_map >= thresh
        visited = np.zeros((n_chan, n_time), dtype=bool)

        for ch in range(n_chan):
            for tp in range(n_time):

                if not supra[ch, tp] or visited[ch, tp]:
                    continue

                cluster = []
                queue = deque([(ch, tp)])
                visited[ch, tp] = True

                while queue:
                    c_ch, c_tp = queue.popleft()
                    cluster.append((c_ch, c_tp))

                    neighbors = []

                    # spatial neighbors
                    chan_neigh = np.where(chan_neighbors[c_ch, :] == 1)[0]
                    for nch in chan_neigh:
                        neighbors.append((nch, c_tp))

                    # temporal neighbors
                    if c_tp > 0:
                        neighbors.append((c_ch, c_tp - 1))
                    if c_tp < n_time - 1:
                        neighbors.append((c_ch, c_tp + 1))

                    for nch, ntp in neighbors:
                        if supra[nch, ntp] and not visited[nch, ntp]:
                            visited[nch, ntp] = True
                            queue.append((nch, ntp))

                extent = len(cluster)

                for c_ch, c_tp in cluster:
                    scores[c_ch, c_tp] += h * (extent ** e_power)

    tfce_map = scores * sign_map

    return tfce_map



# ==========================================================
# Step 1: observed regression t-map
# ==========================================================

t_obs = compute_tmap_lm(data, group)

# ==========================================================
# Step 2: permutation test using max-t correction
# ==========================================================

n_perm = 999
perm_max_t = np.zeros(n_perm)

rng = np.random.default_rng(123)

for p in range(n_perm):
    if (p + 1) % 50 == 0:
        print(f"Permutation {p + 1} / {n_perm}")

    perm_group = rng.permutation(group)

    perm_t = compute_tmap_lm(data, perm_group)

    perm_max_t[p] = np.max(np.abs(perm_t))

# ==========================================================
# Step 3: corrected p-values
# ==========================================================

p_values = np.zeros((n_chan, n_time))

for ch in range(n_chan):
    for tp in range(n_time):
        p_values[ch, tp] = (
            np.sum(perm_max_t >= abs(t_obs[ch, tp])) + 1
        ) / (n_perm + 1)

alpha = 0.05
mask = p_values < alpha

# ==========================================================
# Step 4: plot significant t-values
# ==========================================================

sig_t = t_obs.copy()
sig_t[~mask] = 0

plt.figure(figsize=(10, 6))
plt.imshow(
    sig_t,
    aspect="auto",
    origin="lower",
    extent=[times[0], times[-1], 1, n_chan],
    cmap="RdBu_r"
)
plt.colorbar(label="t value")
plt.xlabel("Time (ms)")
plt.ylabel("Channel")
plt.title("Regression-based max-t corrected effects")
plt.xlim([-200, 800])
plt.show()

# ==========================================================
# Step 5: save results
# ==========================================================

Path("../results").mkdir(exist_ok=True)

sio.savemat(
    "../results/01_between_subject_regression_permutation_Python.mat",
    {
        "t_obs": t_obs,
        "perm_max_t": perm_max_t,
        "p_values": p_values,
        "mask": mask,
        "times": times,
        "group": group,
        "alpha": alpha,
        "n_perm": n_perm,
    }
)

print("Regression-based permutation analysis completed.")