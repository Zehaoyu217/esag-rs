args <- commandArgs(trailingOnly = TRUE); dir <- args[1]
trim_mean <- function(x) { ub <- quantile(x, 0.98); mean(x[x <= ub]) }
metrics <- function(file, gamma_true, label) {
  e <- new.env(); load(file, envir = e); result <- e$result
  ok <- which(sapply(result, function(r) is.list(r) && !is.null(r$table)))
  mat <- do.call(rbind, lapply(result[ok], function(r) r$table[, 1:26]))
  bcols <- paste0("b", 1:14); nzt <- bcols[gamma_true != 0]; zt <- bcols[gamma_true == 0]
  SP <- rowSums(mat[, nzt, drop = FALSE] != 0) / length(nzt)
  SE <- rowSums(mat[, zt, drop = FALSE] == 0) / length(zt)
  CR <- (rowSums(mat[, nzt, drop = FALSE] != 0) + rowSums(mat[, zt, drop = FALSE] == 0)) / 14
  lam <- mat[, "lambda"]
  out <- t(sapply(sort(unique(lam)), function(l) { i <- lam == l
    c(lambda = l, SE = trim_mean(SE[i]), SP = 1 - trim_mean(1 - SP[i]), CR = trim_mean(CR[i])) }))
  cat(sprintf("\n== %s [%s] n_ok=%d; true gamma in data rows: b1..b5 of replicate-1 lambda0 = %s\n", basename(file), label, length(ok),
              paste(round(mat[1, c('b1','b2','b3','b4','b5')], 2), collapse = ",")))
  print(round(t(out), 3))
}
g_exp2 <- c(5,3,0,0,0,0,0,0,0,0,0,0,0,0); g_v2 <- c(5,3,0,5,1,0,0,0,0,0,0,0,0,0)
g_exp1 <- c(5,2,3,0,5,1,0,1,5,0,3,2,0,0)
metrics(file.path(dir, "g2_GLASSO_k50.RData"), g_v2, "v2 truth")
metrics(file.path(dir, "g2_aG_k100.RData"), g_v2, "v2 truth")
metrics(file.path(dir, "g1_GLASSO_k50.RData"), g_exp2, "exp2 truth")
metrics(file.path(dir, "l4_LASSO_k100.RData"), g_exp2, "exp2 truth")
metrics(file.path(dir, "l4_RoC_k100.RData"), g_exp2, "exp2 truth")
metrics(file.path(dir, "g3_GLASSO_k100.RData"), g_exp1, "exp1 truth")
