library(reticulate)
library(Matrix)
library(igraph)

py_install(
  packages = c("mne", "numpy", "scipy", "matplotlib"),
  pip = TRUE
)

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
  
  ch_adj <- py_to_r(result[[1]]$toarray())
  ch_adj <- Matrix(ch_adj, sparse = TRUE)
  
  ch_names <- result[[2]]
  
  list(
    ch_adj = ch_adj,
    ch_names = ch_names
  )
}

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
  cluster_times <- 9:13
  
  x[cluster_times, cluster_ch_idx] <-
    x[cluster_times, cluster_ch_idx] + 4.0
  
  list(
    x = x,
    cluster_channels = cluster_channels,
    cluster_ch_idx = cluster_ch_idx,
    cluster_times = cluster_times
  )
}

make_spatiotemporal_adjacency <- function(ch_adj, n_times) {
  n_channels <- nrow(ch_adj)
  
  spatial_adj <- kronecker(
    Diagonal(n_times),
    ch_adj
  )
  
  time_adj <- bandSparse(
    n = n_times,
    k = c(-1, 1),
    diagonals = list(
      rep(1, n_times - 1),
      rep(1, n_times - 1)
    )
  )
  
  temporal_adj <- kronecker(
    time_adj,
    Diagonal(n_channels)
  )
  
  adjacency <- spatial_adj + temporal_adj
  adjacency[adjacency > 0] <- 1
  
  adjacency
}

tfce_transform <- function(x, adjacency, threshold, tail = 1) {
  original_dim <- dim(x)
  
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
  
  scores_mat <- matrix(
    scores,
    nrow = original_dim[2],
    ncol = original_dim[1]
  )
  
  t(scores_mat)
}

run_mne_spatiotemporal_tfce_example <- function(plot = TRUE) {
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
  
  list(
    x = x,
    scores = scores,
    adjacency = adjacency,
    ch_names = ch_names,
    cluster_channels = test$cluster_channels,
    cluster_times = test$cluster_times
  )
}

res <- run_mne_spatiotemporal_tfce_example(plot = TRUE)
