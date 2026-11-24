
compute_tmap <- function(data, formula, effect,
                         channels, times) {
  
  nChan <- length(channels)
  nTime <- length(times)
  
  res <- data %>%
    group_by(Channel, Time) %>%
    group_modify(~ {
      
      ## Fit the specified model
      if (lme4::isREML(formula)) {
        stop("Please supply a model formula, not a fitted model.")
      }
      
      
      if (length(lme4::findbars(formula)) > 0) {
        fit <- lmerTest::lmer(formula, data = .x)
      } else {
        fit <- lm(formula, data = .x)
      }
      
      ## Extract the statistic for the effect of interest
      cf <- coef(summary(fit))
      
      tibble(
        t_value = cf[effect, "t value"]
      )
    }) %>%
    ungroup()
  
  tmat <- res %>%
    mutate(
      Channel = factor(Channel, levels = channels),
      Time = factor(Time, levels = times)
    ) %>%
    arrange(Channel, Time) %>%
    pull(t_value)
  
  matrix(
    tmat,
    nrow = nChan,
    ncol = nTime,
    byrow = TRUE,
    dimnames = list(channels, times)
  )
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
