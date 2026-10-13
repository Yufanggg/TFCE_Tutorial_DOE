tfce_transform <- function(x, E = 0.5, H = 2, dh = 0.1) {
  
  if (!requireNamespace("permuco", quietly = TRUE)) {
    stop("Package 'permuco' is required.")
  }
  
  
  if (!is.numeric(x) || !is.matrix(x) || length(x) == 0L || any(!is.finite(x))) {
    stop("x must be a nonempty numeric matrix with finite values.")
  }
  
  if (length(E) != 1L || !is.finite(E) || E < 0 ||
      length(H) != 1L || !is.finite(H) || H < 0 ||
      length(dh) != 1L || !is.finite(dh) || dh <= 0) {
    stop("E and H must be nonnegative scalars; dh must be positive.")
  }
  
  distribution <- matrix(as.vector(x), nrow = 1)
  
  max_h <- max(abs(distribution), na.rm = TRUE)
  
  if (max_h == 0) {
    return(matrix(
      0,
      nrow = nrow(x),
      ncol = ncol(x),
      dimnames = dimnames(x)
    ))
  }
  
  n_h <- floor(max_h / dh)
  dhi <- seq_len(n_h)
  
  
  tfce_out <- permuco:::tfce_distribution(pmax(distribution, 0), E, H, dh, dhi) -
    permuco:::tfce_distribution(pmax(-distribution, 0), E, H, dh, dhi)
  
  tfce_map <- matrix(
    tfce_out,            # Values from the first row
    nrow = nrow(x),           # Same number of rows as x
    ncol = ncol(x),           # Same number of columns as x
    dimnames = dimnames(x)   # Same row and column names as x
  )
  
  return(tfce_map)
}