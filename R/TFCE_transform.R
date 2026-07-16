tfce_transform <- function(x, E = 0.5, H = 2, dh = 0.1) {
  
  x = abs(x)
  
  if (!requireNamespace("permuco", quietly = TRUE)) {
    stop("Package 'permuco' is required.")
  }
  
  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }
  
  if (!is.numeric(x)) {
    stop("Input x must be numeric.")
  }
  
  distribution <- matrix(
    as.vector(x),
    nrow = 1
  )
  
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
  
  tfce_out <- permuco:::tfce_distribution(
    distribution,
    E,
    H,
    dh,
    dhi
  )
  
  tfce_map <- matrix(
    tfce_out[1, ],
    nrow = nrow(x),
    ncol = ncol(x),
    dimnames = dimnames(x)
  )
  
  return(tfce_map)
}