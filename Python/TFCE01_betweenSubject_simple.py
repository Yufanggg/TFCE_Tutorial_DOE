import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.stats import linregress
from scipy.sparse.csgraph import connected_components


# ==========================================================
# Settings
# ==========================================================

eeg_csv = "../data/EEGdata_long.csv"
design_csv = "../data/designTable.csv"

n_perm = 999
alpha = 0.05
rng = np.random.default_rng(123)

times_expected = np.arange(-200, 801, 4)

channels = [
    'Fp1','Fp2',
    'F7','F3','Fz','F4','F8',
    'FC5','FC1','FC2','FC6',
    'T7','C3','Cz','C4','T8',
    'CP5','CP1','CP2','CP6',
    'P7','P3','Pz','P4','P8',
    'PO9','O1','Oz','O2','PO10',
    'TP9','TP10'
]


# ==========================================================
# Load CSV data
# ==========================================================

df = pd.read_csv(eeg_csv)
design = pd.read_csv(design_csv)

subjects = np.sort(df["subject"].unique())
times = np.sort(df["time"].unique())

n_sub = len(subjects)
n_chan = len(channels)
n_time = len(times)

EEGdata = np.empty((n_sub, n_chan, n_time))

for i, sub in enumerate(subjects):
    tmp = df[df["subject"] == sub]
    mat = tmp.pivot(index="channel", columns="time", values="Amplitude")
    EEGdata[i] = mat.loc[channels, times].to_numpy()

design = design.sort_values("Subject")
group = design["GroupCode"].to_numpy()

print("EEGdata shape:", EEGdata.shape)


# ==========================================================
# Step 1: observed t-statistic map
# MATLAB equivalent: fitlm(X, EEG_local), coefficient 2 tStat
# ==========================================================

def regression_t_map(EEGdata, group):
    n_sub, n_chan, n_time = EEGdata.shape
    t_map = np.zeros((n_chan, n_time))

    for ch in range(n_chan):
        for t in range(n_time):
            y = EEGdata[:, ch, t]
            res = linregress(group, y)
            t_map[ch, t] = res.slope / res.stderr

    return t_map


print("Step 1: Computing observed t-statistic map...")
tObs = regression_t_map(EEGdata, group)


# ==========================================================
# Step 2: channel × time adjacency
# ==========================================================

info = mne.create_info(channels, sfreq=512, ch_types="eeg")
info.set_montage("standard_1020")

ch_adj, _ = mne.channels.find_ch_adjacency(info, ch_type="eeg")
ch_adj = ch_adj.tocsr()


def make_spatiotemporal_adjacency(ch_adj, n_time):
    n_chan = ch_adj.shape[0]

    spatial = sparse.kron(
        ch_adj,
        sparse.eye(n_time, format="csr"),
        format="csr"
    )

    row = np.arange(n_time - 1)
    col = np.arange(1, n_time)

    time_adj = sparse.coo_array(
        (np.ones(n_time - 1), (row, col)),
        shape=(n_time, n_time)
    )
    time_adj = time_adj + time_adj.T

    temporal = sparse.kron(
        sparse.eye(n_chan, format="csr"),
        time_adj,
        format="csr"
    )

    return (spatial + temporal).tocoo()


adjacency = make_spatiotemporal_adjacency(ch_adj, n_time)


# ==========================================================
# Step 3: TFCE transform
# Similar to ept_mex_TFCE2D(tObs, ChN, E_H)
# E_H = [0.66, 2] means E=0.66, H=2
# ==========================================================

def tfce_transform(x, adjacency, dh=0.1, E=0.66, H=2.0, tail=0):
    x = np.asarray(x)
    original_shape = x.shape
    x_flat = x.ravel()

    scores = np.zeros_like(x_flat, dtype=float)

    max_h = np.nanmax(np.abs(x_flat))
    thresholds = np.arange(0, max_h, dh)

    for h in thresholds:
        if tail == 1:
            masks = [x_flat > h]
            signs = [1]
        elif tail == -1:
            masks = [x_flat < -h]
            signs = [-1]
        else:
            masks = [x_flat > h, x_flat < -h]
            signs = [1, -1]

        height_weight = dh * (h ** H)

        for mask, sign in zip(masks, signs):
            if not np.any(mask):
                continue

            clusters = find_clusters_sparse(mask, adjacency)

            for cluster in clusters:
                extent = len(cluster)
                scores[cluster] += sign * height_weight * (extent ** E)

    return scores.reshape(original_shape)


def find_clusters_sparse(mask, adjacency):
    adjacency = sparse.coo_array(adjacency)

    active_edges = mask[adjacency.row] & mask[adjacency.col]

    row = adjacency.row[active_edges]
    col = adjacency.col[active_edges]
    data = adjacency.data[active_edges]

    active_nodes = np.where(mask)[0]

    row = np.concatenate([row, active_nodes])
    col = np.concatenate([col, active_nodes])
    data = np.concatenate([data, np.ones(active_nodes.size)])

    graph = sparse.coo_array((data, (row, col)), shape=adjacency.shape)

    _, components = connected_components(graph, directed=False)

    clusters = []
    for comp in np.unique(components[active_nodes]):
        cluster = np.where((components == comp) & mask)[0]
        clusters.append(cluster)

    return clusters


print("Step 2: Computing observed TFCE map...")
TFCE_Obs = tfce_transform(
    tObs,
    adjacency=adjacency,
    dh=0.1,
    E=0.66,
    H=2.0,
    tail=0
)


# ==========================================================
# Step 4: permutation testing
# ==========================================================

print(f"Step 3: Starting permutation testing: {n_perm} permutations...")

TFCE_permMax = np.zeros(n_perm)

for p in range(n_perm):
    perm_group = rng.permutation(group)

    permT = regression_t_map(EEGdata, perm_group)

    TFCE_perm = tfce_transform(
        permT,
        adjacency=adjacency,
        dh=0.1,
        E=0.66,
        H=2.0,
        tail=0
    )

    TFCE_permMax[p] = np.max(np.abs(TFCE_perm))

    if (p + 1) % 50 == 0:
        print(f"Finished permutation {p + 1}/{n_perm}")


# ==========================================================
# Step 5: corrected significance
# ==========================================================

print("Step 4: Computing TFCE-corrected significance...")

maxTFCE = np.sort(np.r_[TFCE_permMax, np.max(np.abs(TFCE_Obs))])
crit_index = int(np.round(n_perm * (1 - alpha)))
crit_index = min(crit_index, len(maxTFCE) - 1)

maxTFCEcrit = maxTFCE[crit_index]

Mask = np.abs(TFCE_Obs) >= maxTFCEcrit

P_Values = np.zeros_like(TFCE_Obs)

for ch in range(n_chan):
    for t in range(n_time):
        P_Values[ch, t] = (
            np.sum(TFCE_permMax >= np.abs(TFCE_Obs[ch, t])) + 1
        ) / (n_perm + 1)

print("Critical TFCE value:", maxTFCEcrit)


# ==========================================================
# Step 6: plots
# ==========================================================

sigT = tObs.copy()
sigT[~Mask] = 0

plt.figure()
plt.imshow(
    sigT,
    aspect="auto",
    origin="lower",
    extent=[times[0], times[-1], 0, n_chan - 1],
)
plt.yticks(np.arange(n_chan), channels)
plt.xlabel("Time (ms)")
plt.ylabel("Channel")
plt.title("TFCE-corrected Significant Effects")
plt.colorbar()
plt.tight_layout()

plt.figure()
plt.imshow(
    TFCE_Obs,
    aspect="auto",
    origin="lower",
    extent=[times[0], times[-1], 0, n_chan - 1],
)
plt.yticks(np.arange(n_chan), channels)
plt.xlabel("Time (ms)")
plt.ylabel("Channel")
plt.title("Observed TFCE Map")
plt.colorbar()
plt.tight_layout()

plt.show()


# ==========================================================
# Step 7: save results
# ==========================================================

out = pd.DataFrame({
    "channel": np.repeat(channels, n_time),
    "time": np.tile(times, n_chan),
    "tObs": tObs.ravel(),
    "TFCE_Obs": TFCE_Obs.ravel(),
    "P_Value": P_Values.ravel(),
    "Mask": Mask.ravel(),
})

out.to_csv("../results/01_TFCE_between_subject_results.csv", index=False)

np.savez(
    "../results/01_TFCE_between_subject_results.npz",
    tObs=tObs,
    TFCE_Obs=TFCE_Obs,
    TFCE_permMax=TFCE_permMax,
    maxTFCEcrit=maxTFCEcrit,
    P_Values=P_Values,
    Mask=Mask,
    times=times,
    channels=np.array(channels),
    alpha=alpha,
    n_perm=n_perm,
)

print("TFCE analysis completed and results saved.")