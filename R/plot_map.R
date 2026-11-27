plot_map <- funcation(X){
  # X is your electrode ¡Á time matrix of signed TFCE scores.
  times_ms <- as.numeric(colnames(X))
  electrodes <- rownames(X)
  
  cols <- colorRampPalette(c("#2166AC", "white", "#B2182B"))(101)
  limit <- max(abs(X[is.finite(X)]), 0)
  if (limit == 0) limit <- 1
  
  breaks <- seq(-limit, limit, length.out = length(cols) + 1L)
  midpoints <- (head(breaks, -1L) + tail(breaks, -1L)) / 2
  
  old_par <- par(no.readonly = TRUE)
  layout(matrix(c(1, 2), nrow = 1), widths = c(5, 1))
  
  # Left panel: electrode¨Ctime heatmap.
  par(mar = c(4, 5, 2, 0.5))
  image(
    times_ms, seq_len(nrow(X)), t(X),
    col = cols, breaks = breaks,
    xlab = "Time (ms)", ylab = "Electrode", yaxt = "n"
  )
  axis(2, at = seq_len(nrow(X)), labels = electrodes, las = 1, cex.axis = 0.75)
  box()
  
  # Right panel: color bar.
  par(mar = c(4, 0.5, 2, 3.5))
  image(
    x = c(0, 1), y = breaks,
    z = matrix(midpoints, nrow = 1),
    col = cols, breaks = breaks,
    axes = FALSE, xlab = "", ylab = "",
    xaxs = "i", yaxs = "i"
  )
  axis(4, at = c(-limit, 0, limit), labels = signif(c(-limit, 0, limit), 3),
       las = 1, cex.axis = 0.8)
  mtext("TFCE", side = 3, line = 0.4)
  box()
  
  layout(1)
  par(old_par)
}