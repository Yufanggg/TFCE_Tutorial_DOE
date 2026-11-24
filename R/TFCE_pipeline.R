
TFCE_pipeline <- function(
    data,
    formula,
    effect,
    permute_data,
    channels,
    times,
    n_perm = 999,
    alpha = 0.05,
    E = 0.5,
    H = 2,
    dh = 0.1
) {
  
  ## ========================================================
  ## Step 1. Calculate observed t-map
  ## ========================================================
  
  t_obs <- compute_tmap(
    data = data,
    formula = formula,
    effect = effect,
    channels = channels,
    times = times
  )
  
  ## ========================================================
  ## Step 2. Calculate observed TFCE map
  ## ========================================================
  
  TFCE_obs <- tfce_transform(t_obs, E = E, H = H, dh = dh)
  
  ## ========================================================
  ## Step 3. Generate permutation distributions
  ## ========================================================
  
  perm_max <- numeric(n_perm)
  perm_min <- numeric(n_perm)
  
  
  library(future)
  library(future.apply)
  
  ## Set up parallel workers
  n_workers <- max(1, parallel::detectCores() - 1)
  
  plan(multisession, workers = n_workers)
  
  ## Run permutations in parallel
  perm_results <- future_lapply(seq_len(n_perm), function(b) {
    ## Generate transformed data
    perm_dat <- permute_data(data)
    
    ## Calculate transformed t-map
    t_perm <- compute_tmap(
      data = perm_dat,
      formula = formula,
      effect = effect,
      channels = channels,
      times = times
    )
    
    ## Calculate transformed TFCE map
    TFCE_perm <- tfce_transform(t_perm, E = E, H = H, dh = dh)
    
    ## Return positive and negative extrema
    c(max = max(TFCE_perm, na.rm = TRUE),
      min = min(TFCE_perm, na.rm = TRUE))
  }, future.seed = TRUE)
  
  ## Combine permutation results
  perm_results <- do.call(rbind, perm_results)
  
  perm_max <- perm_results[, "max"]
  perm_min <- perm_results[, "min"]
  
  ## Restore sequential execution
  plan(sequential)
  
  
  ## ========================================================
  ## Step 4. Calculate adjusted p-values
  ## ========================================================
  
  ## TFCE_adjusted_p() adds the observed extrema
  ## to the two reference distributions
  P_Values <- TFCE_adjusted_p(
    observed_map = TFCE_obs,
    perm_max = perm_max,
    perm_min = perm_min
  )
  
  ## ========================================================
  ## Step 5. Identify significant points
  ## ========================================================
  
  Mask <- P_Values <= alpha
  
  ## ========================================================
  ## Step 6. Return results
  ## ========================================================
  
  list(
    tObs = t_obs,
    TFCE_Obs = TFCE_obs,
    TFCE_Null_Pos = perm_max,
    TFCE_Null_Neg = perm_min,
    P_Values = P_Values,
    Mask = Mask,
    alpha = alpha,
    nPerm = n_perm,
    formula = formula,
    effect = effect
  )
}
