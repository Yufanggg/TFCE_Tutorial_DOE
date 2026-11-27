TFCE_transform <- function(map,
                           graph,
                           E = 0.66,
                           H = 2,
                           ndh = 500L,
                           alternative = "two.sided") {
  alternative <- match.arg(alternative, c("greater", "less", "two.sided"))
  
  switch(
    alternative,
    greater   = tfce_transform(map, graph, E, H, ndh),
    less      = -tfce_transform(-map, graph, E, H, ndh),
    two.sided = tfce_transform(map, graph, E, H, ndh) -
      tfce_transform(-map, graph, E, H, ndh)
  )
}


get_graph <- function(channels, map){
  # map: electrode \times time
  stopifnot(!is.null(channels), !anyDuplicated(channels))
  
  # eegUtils looks up standard scalp coordinates from electrode names.
  locations <- eegUtils::electrode_locations(data.frame(electrode = channels), drop = FALSE)
  
  # Match the coordinates to the row order of your t-map.
  idx <- match(channels, locations$electrode)
  xyz <- as.matrix(locations[idx, c("cart_x", "cart_y", "cart_z")])
  
  radius <- sqrt(rowSums(xyz^2))
  valid <- is.finite(radius) & radius > 0
  if (any(!valid)) {
    stop("Missing coordinates for: ", paste(electrodes[!valid], collapse = ", "))
  }
  
  # Base R computes the angular distance between each pair of electrodes.
  unit_xyz <- sweep(xyz, 1L, radius, "/")
  cos_angle <- tcrossprod(unit_xyz)
  angle_deg <- acos(pmax(pmin(cos_angle, 1), -1)) * 180 / pi
  
  # Apply the same cutoff to every electrode pair.
  spatial_cutoff_deg <- 35
  spatial_adj <- angle_deg <= spatial_cutoff_deg
  diag(spatial_adj) <- FALSE
  dimnames(spatial_adj) <- list(channels, channels)
  
  
  # 1. igraph: build the electrode-only graph in t-map row order.
  channel_graph <- igraph::graph_from_adjacency_matrix(spatial_adj[channels, channels, drop = FALSE], mode = "undirected", diag = FALSE)
  
  # 2. permuco4brain: repeat the electrode graph at each time point
  #    and connect each electrode to itself at the next time point.
  graph <- permuco4brain:::full_graph(channel_graph, t = ncol(map))
  
  stopifnot(igraph::vcount(graph) == length(map))
  
  return(graph)
}

tfce_transform <- function(map, graph, E, H, ndh) {
  # map: electrode \times time
  stopifnot(ndh >= 2L)
  positive <- map >= 0
  map <- map * positive
  
  max_height <- max(map)
  dhi <- seq(0, max_height, length.out = ndh)
  dh <- dhi[2L] - dhi[1L]
  HH <- dh * dhi^H
  
  # Wrap the observed t-map as 1 map ¡Á time ¡Á electrode.
  distribution <- array(t(map), dim = c(1L, ncol(map), nrow(map)))
  
  # graph is the full electrode¨Ctime graph.
  # dhi is the threshold grid and HH contains dh * dhi^H.
  extent_maps <- permuco4brain:::cluster_extend_array(distribution = distribution,
                                                      graph = graph,
                                                      threshold = dhi)
  
  tfce_contributions <- Map(function(extent, weight)
    extent^E * weight, extent_maps, HH[seq_along(extent_maps)])
  
  # Sum the contributions across thresholds at each electrode¨Ctime point.
  tfce_scores <- Reduce(`+`, tfce_contributions)
  
  
  # Restore the original electrode ¡Á time layout and labels.
  tfce_map <- t(matrix(
    tfce_scores,
    nrow = ncol(map),
    ncol = nrow(map),
    dimnames = rev(dimnames(map))
  ))
  
  return(tfce_map)
  
}


TFCE_adjusted_p <- function(observed_map, perm_max, perm_min) {
  
  
  ## Include observed extrema in the reference distributions
  perm_max <- c(perm_max, max(observed_map, na.rm = TRUE))
  perm_min <- c(perm_min, min(observed_map, na.rm = TRUE))
  
  B <- length(perm_max)
  
  stopifnot(length(perm_min) == B)
  
  ## Positive-tail adjusted p-values
  p_pos <- matrix(apply(observed_map, c(1, 2), function(x) mean(perm_max >= x)),
    nrow = nrow(observed_map),
    ncol = ncol(observed_map),
    dimnames = dimnames(observed_map)
  )
  
  
  ## Negative-tail adjusted p-values
  p_neg <- matrix(apply(observed_map, c(1, 2), function(x) mean(perm_min <= x)),
                  nrow = nrow(observed_map),
                  ncol = ncol(observed_map),
                  dimnames = dimnames(observed_map)
  )
  
  ## Two-sided adjustment
  
  p_adj <- matrix(NA, nrow = nrow(p_pos), ncol = ncol(p_pos))
  
  for (row in 1:nrow(p_pos)){
    for (col in 1:ncol(p_pos)){
      p_adj[row, col] = min(1, 2*min(p_pos[row, col], p_neg[row, col]))
    }
  }
  
  
  ## Restore electrode-time dimensions
  matrix(
    p_adj,
    nrow = nrow(observed_map),
    ncol = ncol(observed_map),
    dimnames = dimnames(observed_map)
  )
}