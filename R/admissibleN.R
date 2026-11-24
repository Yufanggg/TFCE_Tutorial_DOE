
admissibleN <- function(
    data,
    design,
    effect = NULL,
    method = "label"
) {
  
  ## Factorial function for numerical stability
  factorialN <- function(n) lgamma(n + 1)
  
  ## Number of distinct allocations
  allocationN <- function(x) {
    counts <- as.numeric(table(x))
    exp(
      factorialN(sum(counts)) -
        sum(factorialN(counts))
    )
  }
  
  switch(
    design,
    
    ## 1. Between-subject design
    between = {
      assignments <- dplyr::distinct(
        data, Subject, CondCode
      )
      allocationN(assignments$CondCode)
    },
    
    ## 2. Within-subject design
    within = {
      n <- dplyr::n_distinct(data$Subject)
      2^n
    },
    
    ## 3. Factorial design
    factorial = {
      assignments <- dplyr::distinct(
        data, Subject, FactorA, FactorB
      )
      
      if (effect == "FactorA") {
        prod(vapply(
          split(assignments$FactorA, assignments$FactorB),
          allocationN, numeric(1)
        ))
      } else if (effect == "FactorB") {
        prod(vapply(
          split(assignments$FactorB, assignments$FactorA),
          allocationN, numeric(1)
        ))
      } else {
        stop("Specify the interaction transformation.")
      }
    },
    
    ## 4. Split-plot design
    split_plot = {
      if (effect == "GroupCode") {
        assignments <- dplyr::distinct(
          data, Subject, GroupCode
        )
        allocationN(assignments$GroupCode)
        
      } else if (effect == "CondCode") {
        n <- dplyr::n_distinct(data$Subject)
        2^n
        
      } else {
        stop("Specify the interaction transformation.")
      }
    },
    
    ## 5. Nested design
    nested = {
      n <- nrow(dplyr::distinct(
        data, Class, Student
      ))
      2^n
    },
    
    ## 6. Crossed design:
    ## Within Subject-Item pair exchanges
    crossed = {
      n <- nrow(dplyr::distinct(
        data, Subject, Item
      ))
      2^n
    },
    
    stop("Unknown experimental design.")
  )
}
