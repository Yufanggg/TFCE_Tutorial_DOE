library(permuco)

## Test data
ch_names <- c(
  "Fp1", "Fp2",
  "F7", "F3", "Fz", "F4", "F8",
  "FC5", "FC1", "FC2", "FC6",
  "T7", "C3", "Cz", "C4", "T8",
  "CP5", "CP1", "CP2", "CP6",
  "P7", "P3", "Pz", "P4", "P8",
  "O1", "Oz", "O2"
)

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

test <- make_test_data(ch_names)
x <- test$x

## Flatten map into one row
distribution <- matrix(as.vector(x), nrow = 1)

## TFCE parameters
E <- 0.5
H <- 2
dh <- 0.1

## Important fix:
## do NOT use seq(0, max(abs(distribution)), by = dh)
n_h <- floor(max(abs(distribution)) / dh)
dhi <- seq_len(n_h)

## Call internal function
tfce_out <- permuco:::tfce_distribution(
  distribution,
  E,
  H,
  dh,
  dhi
)

## Reshape back to time ¡Á channel
tfce_map <- matrix(
  tfce_out[1, ],
  nrow = nrow(x),
  ncol = ncol(x)
)

colnames(tfce_map) <- ch_names
rownames(tfce_map) <- paste0("t", seq_len(nrow(x)))

## Inspect injected cluster
tfce_map[test$cluster_times, test$cluster_ch_idx]

## 8. Plot original map and TFCE map side by side

op <- par(mfrow = c(1, 2), mar = c(5, 7, 4, 2))

## Original t-map
image(
  x = seq_len(nrow(x)),
  y = seq_len(ncol(x)),
  z = x,
  xlab = "Time",
  ylab = "Channel",
  main = "Original t-map",
  axes = FALSE
)

axis(1, at = seq_len(nrow(x)))
axis(2, at = seq_len(ncol(x)), labels = ch_names, las = 2)
box()

## TFCE map
image(
  x = seq_len(nrow(tfce_map)),
  y = seq_len(ncol(tfce_map)),
  z = tfce_map,
  xlab = "Time",
  ylab = "Channel",
  main = "TFCE map",
  axes = FALSE
)

axis(1, at = seq_len(nrow(tfce_map)))
axis(2, at = seq_len(ncol(tfce_map)), labels = ch_names, las = 2)
box()

par(op)