args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
trim_mean <- function(x) { ub <- quantile(x, 0.98); mean(x[x <= ub]) }
metrics <- function(file, gamma_true) {
  e <- new.env(); load(file, envir = e); result <- e$result
  cls <- sapply(result, function(r) class(r)[1])
  ok <- which(sapply(result, function(r) is.list(r) && !is.null(r$table)))
  mat <- do.call(rbind, lapply(result[ok], function(r) r$table[, 1:26]))
  bcols <- paste0("b", 1:14)
  nzt <- bcols[gamma_true != 0]; zt <- bcols[gamma_true == 0]
  SP <- rowSums(mat[, nzt, drop = FALSE] != 0) / length(nzt)
  SE <- rowSums(mat[, zt, drop = FALSE] == 0) / length(zt)
  CR <- (rowSums(mat[, nzt, drop = FALSE] != 0) + rowSums(mat[, zt, drop = FALSE] == 0)) / 14
  lam <- mat[, "lambda"]
  out <- t(sapply(sort(unique(lam)), function(l) {
    i <- lam == l
    c(lambda = l, SE = mean(SE[i]), SP = mean(SP[i]), CR = mean(CR[i]),
      SE_trim = trim_mean(SE[i]), SP_trim = 1 - trim_mean(1 - SP[i]), CR_trim = trim_mean(CR[i]))
  }))
  cat(sprintf("\n== %s: %d/%d replicates usable; classes: %s\n", basename(file), length(ok), length(result),
              paste(names(table(cls)), table(cls), collapse = ", ")))
  print(round(out, 3))
}
g_exp2 <- c(5,3,0,0,0,0,0,0,0,0,0,0,0,0)
g_v2   <- c(5,3,0,5,1,0,0,0,0,0,0,0,0,0)
metrics(file.path(dir, "g1_GLASSO_k100.RData"), g_exp2)
metrics(file.path(dir, "g1_RoCG_k100.RData"), g_exp2)
metrics(file.path(dir, "g2_GLASSO_k100.RData"), g_v2)
metrics(file.path(dir, "g2_RoCG_k100.RData"), g_v2)
