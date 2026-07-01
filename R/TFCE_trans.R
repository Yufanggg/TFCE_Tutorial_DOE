library(igraph)

# ==========================================================
# TFCE transformation for channels ¡Á time t-map
# ==========================================================

tfce_transform <- function(tmap, adjacency, dh = 0.1, E = 0.66, H = 2, tail = 0) {
  
  n_chan <- nrow(tmap)
  n_time <- ncol(tmap)
  n_node <- n_chan * n_time
  
  tvec <- as.vector(t(tmap))
  tfce <- rep(0, n_node)
  
  max_h <- max(abs(tvec), na.rm = TRUE)
  thresholds <- seq(0, max_h, by = dh)
  
  for (h in thresholds) {
    
    if (tail == 1) {
      masks <- list(tvec > h)
      signs <- c(1)
    } else if (tail == -1) {
      masks <- list(tvec < -h)
      signs <- c(-1)
    } else {
      masks <- list(tvec > h, tvec < -h)
      signs <- c(1, -1)
    }
    
    for (m in seq_along(masks)) {
      
      mask <- masks[[m]]
      sign_val <- signs[m]
      active <- which(mask)
      
      if (length(active) == 0) next
      
      sub_adj <- adjacency[active, active]
      g <- graph_from_adjacency_matrix(sub_adj, mode = "undirected")
      comps <- components(g)$membership
      
      for (comp_id in unique(comps)) {
        cluster_nodes <- active[which(comps == comp_id)]
        extent <- length(cluster_nodes)
        
        tfce[cluster_nodes] <- tfce[cluster_nodes] +
          sign_val * dh * (h ^ H) * (extent ^ E)
      }
    }
  }
  
  tfce_mat <- matrix(tfce, nrow = n_time, ncol = n_chan, byrow = TRUE)
  tfce_mat <- t(tfce_mat)
  
  return(tfce_mat)
}


library(reticulate)
library(Matrix)
library(igraph)

# ==========================================================
# 1. Channel names
# ==========================================================

channels <- c(
  'Fp1','Fp2',
  'F7','F3','Fz','F4','F8',
  'FC5','FC1','FC2','FC6',
  'T7','C3','Cz','C4','T8',
  'CP5','CP1','CP2','CP6',
  'P7','P3','Pz','P4','P8',
  'PO9','O1','Oz','O2','PO10',
  'TP9','TP10'
)

# ==========================================================
# 2. Get MNE channel adjacency through reticulate
# ==========================================================

make_mne_channel_adjacency <- function(channels, sfreq = 512) {
  
  mne <- import("mne")
  
  info <- mne$create_info(
    ch_names = channels,
    sfreq = sfreq,
    ch_types = "eeg"
  )
  
  info$set_montage("standard_1020")
  
  result <- mne$channels$find_ch_adjacency(
    info,
    ch_type = "eeg"
  )
  
  ch_adj_py <- result[[1]]
  ch_names <- result[[2]]
  
  ch_adj <- py_to_r(ch_adj_py$toarray())
  ch_adj <- Matrix(ch_adj, sparse = TRUE)
  
  return(list(
    ch_adj = ch_adj,
    ch_names = ch_names
  ))
}

# ==========================================================
# 3. Make test data
# time ¡Á channels
# ==========================================================

make_test_data <- function(ch_names, n_times = 20, seed = 42) {
  
  set.seed(seed)
  
  n_channels <- length(ch_names)
  
  x <- matrix(
    rnorm(n_times * n_channels, mean = 0, sd = 0.2),
    nrow = n_times,
    ncol = n_channels
  )
  
  cluster_channels <- c("C3", "Cz", "C4")
  cluster_ch_idx <- match(cluster_channels, ch_names)
  cluster_times <- 9:13   # R is 1-based; Python 8:12 becomes R 9:13
  
  x[cluster_times, cluster_ch_idx] <- 
    x[cluster_times, cluster_ch_idx] + 4.0
  
  return(list(
    x = x,
    cluster_channels = cluster_channels,
    cluster_ch_idx = cluster_ch_idx,
    cluster_times = cluster_times
  ))
}

# ==========================================================
# 4. Build spatio-temporal adjacency
# Flattening convention:
# time ¡Á channels matrix
# as.vector(t(x)) gives time-major ordering
# node = (time - 1) * n_channels + channel
# ==========================================================

make_spatiotemporal_adjacency <- function(ch_adj, n_times) {
  
  n_channels <- nrow(ch_adj)
  
  spatial_adj <- kronecker(
    Diagonal(n_times),
    ch_adj
  )
  
  time_adj <- bandSparse(
    n = n_times,
    k = c(-1, 1),
    diagonals = list(rep(1, n_times - 1), rep(1, n_times - 1))
  )
  
  temporal_adj <- kronecker(
    time_adj,
    Diagonal(n_channels)
  )
  
  adjacency <- spatial_adj + temporal_adj
  adjacency[adjacency > 0] <- 1
  
  return(adjacency)
}

# ==========================================================
# 5. TFCE transform
# ==========================================================

tfce_transform <- function(x, adjacency, threshold, tail = 1) {
  
  original_dim <- dim(x)
  
  # time-major flattening
  x_vec <- as.vector(t(x))
  
  start <- threshold$start
  step <- threshold$step
  H <- threshold$h_power
  E <- threshold$e_power
  
  if (tail == 1) {
    stop <- max(x_vec, na.rm = TRUE)
  } else if (tail == -1) {
    stop <- min(x_vec, na.rm = TRUE)
  } else {
    stop <- max(abs(x_vec), na.rm = TRUE)
  }
  
  thresholds <- seq(start, stop, by = step)
  
  scores <- rep(0, length(x_vec))
  
  for (ti in seq_along(thresholds)) {
    
    thresh <- thresholds[ti]
    
    if (tail == 1) {
      masks <- list(x_vec > thresh)
      signs <- c(1)
    } else if (tail == -1) {
      masks <- list(x_vec < thresh)
      signs <- c(-1)
    } else {
      masks <- list(x_vec > thresh, x_vec < -thresh)
      signs <- c(1, -1)
    }
    
    if (ti == 1) {
      h <- abs(thresh)
    } else {
      h <- abs(thresh - thresholds[ti - 1])
    }
    
    h <- h ^ H
    
    for (mi in seq_along(masks)) {
      
      mask <- masks[[mi]]
      sign_val <- signs[mi]
      active <- which(mask)
      
      if (length(active) == 0) next
      
      sub_adj <- adjacency[active, active]
      graph <- graph_from_adjacency_matrix(
        as.matrix(sub_adj),
        mode = "undirected"
      )
      
      comps <- components(graph)$membership
      
      for (comp_id in unique(comps)) {
        cluster_nodes <- active[which(comps == comp_id)]
        extent <- length(cluster_nodes)
        
        scores[cluster_nodes] <- scores[cluster_nodes] +
          sign_val * h * (extent ^ E)
      }
    }
  }
  
  scores_mat <- matrix(scores, nrow = original_dim[2], ncol = original_dim[1])
  scores_mat <- t(scores_mat)
  
  return(scores_mat)
}

# ==========================================================
# 6. Run example
# ==========================================================

run_mne_spatiotemporal_tfce_example <- function(plot = FALSE) {
  
  adj_info <- make_mne_channel_adjacency(channels)
  
  ch_adj <- adj_info$ch_adj
  ch_names <- adj_info$ch_names
  
  test <- make_test_data(ch_names, n_times = 20)
  
  x <- test$x
  
  adjacency <- make_spatiotemporal_adjacency(
    ch_adj = ch_adj,
    n_times = nrow(x)
  )
  
  threshold <- list(
    start = 0.0,
    step = 0.5,
    h_power = 2,
    e_power = 0.5
  )
  
  scores <- tfce_transform(
    x = x,
    adjacency = adjacency,
    threshold = threshold,
    tail = 1
  )
  
  cat("Input data shape, time ¡Á channels:", dim(x), "\n")
  cat("Spatio-temporal adjacency shape:", dim(adjacency), "\n")
  cat("TFCE scores shape:", dim(scores), "\n")
  cat("Injected cluster channels:", test$cluster_channels, "\n")
  cat("Injected cluster times:", test$cluster_times, "\n")
  
  if (plot) {
    par(mfrow = c(1, 2))
    
    image(
      t(x[nrow(x):1, ]),
      main = "Input data",
      xlab = "Channel",
      ylab = "Time",
      axes = FALSE
    )
    
    image(
      t(scores[nrow(scores):1, ]),
      main = "TFCE scores",
      xlab = "Channel",
      ylab = "Time",
      axes = FALSE
    )
  }
  
  return(list(
    x = x,
    scores = scores,
    adjacency = adjacency,
    ch_names = ch_names,
    cluster_channels = test$cluster_channels,
    cluster_times = test$cluster_times
  ))
}

# Run
res <- run_mne_spatiotemporal_tfce_example(plot = TRUE)

# ==========================================================
# Create channel ¡Á time adjacency
# Simple grid adjacency:
# neighboring channels + neighboring time points
# ==========================================================

make_adjacency <- function(n_chan, n_time) {
  
  n_node <- n_chan * n_time
  adj <- matrix(0, n_node, n_node)
  
  node_id <- function(ch, time) {
    return((ch - 1) * n_time + time)
  }
  
  for (ch in 1:n_chan) {
    for (time in 1:n_time) {
      
      id <- node_id(ch, time)
      
      if (ch > 1) {
        adj[id, node_id(ch - 1, time)] <- 1
      }
      if (ch < n_chan) {
        adj[id, node_id(ch + 1, time)] <- 1
      }
      if (time > 1) {
        adj[id, node_id(ch, time - 1)] <- 1
      }
      if (time < n_time) {
        adj[id, node_id(ch, time + 1)] <- 1
      }
    }
  }
  
  adj <- adj + t(adj)
  adj[adj > 0] <- 1
  
  return(adj)
}


# ==========================================================
# Test data: channels ¡Á time t-map
# ==========================================================

channels <- c(
  'Fp1','Fp2',
  'F7','F3','Fz','F4','F8',
  'FC5','FC1','FC2','FC6',
  'T7','C3','Cz','C4','T8',
  'CP5','CP1','CP2','CP6',
  'P7','P3','Pz','P4','P8',
  'PO9','O1','Oz','O2','PO10',
  'TP9','TP10'
)

library(reticulate)

mne <- import("mne")

info <- mne$create_info(
  ch_names = channels,
  sfreq = 512,
  ch_types = "eeg"
)

info$set_montage("standard_1020")

result <- mne$channels$find_ch_adjacency(
  info,
  ch_type = "eeg"
)

ch_adj <- result[[1]]
ch_names <- result[[2]]


# ==========================================================
# Run TFCE
# ==========================================================

adjacency <- make_adjacency(n_chan, n_time)

tfce_scores <- tfce_transform(
  tmap = tmap,
  adjacency = ch_adj,
  dh = 0.1,
  E = 0.66,
  H = 2,
  tail = 0
)

cat("\nInput t-map:\n")
print(round(tmap, 3))

cat("\nTFCE scores:\n")
print(round(tfce_scores, 3))


# ==========================================================
# Optional plots
# ==========================================================

par(mfrow = c(1, 2))

image(
  t(tmap[nrow(tmap):1, ]),
  main = "Input t-map",
  xlab = "Time",
  ylab = "Channel",
  axes = FALSE
)
axis(1, at = seq(0, 1, length.out = n_time), labels = 1:n_time)
axis(2, at = seq(0, 1, length.out = n_chan), labels = n_chan:1)

image(
  t(tfce_scores[nrow(tfce_scores):1, ]),
  main = "TFCE scores",
  xlab = "Time",
  ylab = "Channel",
  axes = FALSE
)
axis(1, at = seq(0, 1, length.out = n_time), labels = 1:n_time)
axis(2, at = seq(0, 1, length.out = n_chan), labels = n_chan:1)
