TFCE_adjusted_p <- function(observed_map, perm_max, perm_min) {
  
  
  ## Include observed extrema in the reference distributions
  perm_max <- c(perm_max, max(observed_map, na.rm = TRUE))
  perm_min <- c(perm_min, min(observed_map, na.rm = TRUE))
  
  B <- length(perm_max)
  
  stopifnot(length(perm_min) == B)
  
  ## Positive-tail adjusted p-values
  p_pos <- matrix(
    vapply(as.vector(observed_map), function(x) {
      (sum(perm_max >= x) + 1) / (B + 1)
    }, numeric(1)),
    nrow = nrow(observed_map),
    ncol = ncol(observed_map),
    dimnames = dimnames(observed_map)
  )
  

  ## Negative-tail adjusted p-values
  p_neg <- matrix(
    vapply(as.vector(observed_map), function(x) {
      (sum(perm_min <= x) + 1) / (B + 1)
    }, numeric(1)),
    nrow = nrow(observed_map),
    ncol = ncol(observed_map),
    dimnames = dimnames(observed_map)
  )

  
  print(p_neg)
  
  ## Two-sided adjustment
  p_adj <- pmin(2 * pmin(p_pos, p_neg), 1)
    
  
  ## Restore electrode-time dimensions
  matrix(
    p_adj,
    nrow = nrow(observed_map),
    ncol = ncol(observed_map),
    dimnames = dimnames(observed_map)
  )
}
