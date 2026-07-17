## ==========================================================
## Helper plotting function
## ==========================================================

plot_eeg_map <- function(mat, title, times, nChan, channels) {
  
  fields::image.plot(
    x = times,
    y = seq_len(nChan),
    z = t(mat),
    xlab = "Time (ms)",
    ylab = "Channel",
    main = title,
    axes = FALSE
  )
  
  axis(1, at = seq(-200, 800, by = 200))
  axis(
    2,
    at = seq_len(nChan),
    labels = channels,
    las = 2,
    cex.axis = 0.6
  )
  box()
}