compute_map <- function(
    data, formula, term,
    channel_col = "Channel", time_col = "Time",
    channels = NULL, times = NULL, metric = c("t", "p")
) {
  stopifnot(
    inherits(formula, "formula"),
    is.character(term),
    length(term) == 1L,
    !is.na(term),
    all(c(channel_col, time_col) %in% names(data)),
    channel_col != time_col,!anyNA(data[c(channel_col, time_col)])
  )
  
  metric <- match.arg(metric)
  stat_col <- switch(metric, t = "t value", p = "Pr(>|t|)")
  
  if (is.null(channels)) channels <- sort(unique(data[[channel_col]]))
  if (is.null(times)) times <- sort(unique(data[[time_col]]))
  
  stopifnot(
    length(channels) > 0L, !anyNA(channels), !anyDuplicated(channels),
    length(times) > 0L, !anyNA(times), !anyDuplicated(times)
  )
  
  # Fit one model per channel¨Ctime pair.
  grouped <- dplyr::group_by(data, .data[[channel_col]], .data[[time_col]])
  
  results <- dplyr::group_modify(grouped, function(group_data, key) {
    fit <- stats::lm(formula, data = group_data)
    coefficients <- coef(summary(fit))
    
    #print(coefficients)
    
    stat_value <- if (term %in% rownames(coefficients)) {
      unname(coefficients[term, stat_col])
    } else {
      NA_real_
    }
    
    data.frame(stat_value = stat_value)
  })
  
  # Place each statistic in its corresponding matrix cell.
  map <- matrix(
    NA_real_,
    nrow = length(channels),
    ncol = length(times),
    dimnames = list(as.character(channels), as.character(times))
  )
  
  row_idx <- match(results[[channel_col]], channels)
  col_idx <- match(results[[time_col]], times)
  keep <- !is.na(row_idx) & !is.na(col_idx)
  
  map[cbind(row_idx[keep], col_idx[keep])] <- results$stat_value[keep]
  
  return(map)
}
# -------------------------------------------------------------------------

# 
# ## ==========================================================
# ## 1. Between-subject design
# ## ==========================================================
# 
# t_between <- compute_tmap(
#   dat,
#   formula = Amplitude ~ CondCode,
#   effect = "CondCode",
#   channels = channels,
#   times = times
# )
# 
# 
# ## ==========================================================
# ## 2. Between-subject design with nuisance covariates
# ## ==========================================================
# 
# t_covariate <- compute_tmap(
#   dat,
#   formula = Amplitude ~ CondCode + Covariate,
#   effect = "CondCode",
#   channels = channels,
#   times = times
# )
# 
# 
# ## ==========================================================
# ## 3. Factorial design (2 x 2)
# ## ==========================================================
# 
# formula_factorial <- Amplitude ~ FactorA * FactorB
# 
# t_A <- compute_tmap(
#   dat, formula_factorial, "FactorA", channels, times
# )
# 
# t_B <- compute_tmap(
#   dat, formula_factorial, "FactorB", channels, times
# )
# 
# t_Int <- compute_tmap(
#   dat, formula_factorial, "FactorA:FactorB", channels, times
# )
# 
# 
# ## ==========================================================
# ## 4. Within-subject design
# ## ==========================================================
# 
# t_within <- compute_tmap(
#   dat,
#   formula = Amplitude ~ CondCode + (1 | Subject),
#   effect = "CondCode",
#   channels = channels,
#   times = times
# )
# 
# 
# ## ==========================================================
# ## 5. Split-plot design
# ## ==========================================================
# 
# formula_split <- Amplitude ~
#   GroupCode * CondCode + (1 | Subject)
# 
# t_Group <- compute_tmap(
#   dat, formula_split, "GroupCode", channels, times
# )
# 
# t_Cond <- compute_tmap(
#   dat, formula_split, "CondCode", channels, times
# )
# 
# t_Int <- compute_tmap(
#   dat, formula_split, "GroupCode:CondCode",
#   channels, times
# )
# 
# 
# ## ==========================================================
# ## 6. Nested design
# ## ==========================================================
# 
# t_nested <- compute_tmap(
#   dat,
#   formula = Amplitude ~ ConditionCode +
#     (1 | Class) + (1 | Class:Student),
#   effect = "ConditionCode",
#   channels = channels,
#   times = times
# )
# 
# 
# ## ==========================================================
# ## 7. Fully crossed design
# ## ==========================================================
# 
# ## Original crossed mixed-effects model
# t_crossed <- compute_tmap(
#   dat,
#   formula = Amplitude ~ ConditionCode +
#     (1 | Subject) + (1 | Item),
#   effect = "ConditionCode",
#   channels = channels,
#   times = times
# )
# 
# ## Alternatively, after lmeEEG marginalization
# t_crossed_marginal <- compute_tmap(
#   marginal_dat,
#   formula = MarginalAmplitude ~ ConditionCode,
#   effect = "ConditionCode",
#   channels = channels,
#   times = times
# )
# 
