args <- commandArgs(trailingOnly = TRUE); dir <- args[1]; basic <- args[2]
real_library <- base::library
library <- function(package, ...) { p <- as.character(substitute(package)); if (p == "optimx") return(invisible(NULL)); real_library(p, character.only = TRUE, ...) }
source(basic); rm(library)
trim_mean <- function(x) { ub <- quantile(x, 0.98); mean(x[x <= ub]) }
metrics <- function(file, gamma_metric, gamma_V, label, do_V = TRUE) {
  e <- new.env(); load(file, envir = e); result <- e$result
  mat <- do.call(rbind, lapply(result, function(r) r$table[, 1:26]))
  bcols <- paste0("b", 1:14); nzt <- bcols[gamma_metric != 0]; zt <- bcols[gamma_metric == 0]
  SP <- rowSums(mat[, nzt, drop = FALSE] != 0) / length(nzt)
  SE <- rowSums(mat[, zt, drop = FALSE] == 0) / length(zt)
  CR <- (rowSums(mat[, nzt, drop = FALSE] != 0) + rowSums(mat[, zt, drop = FALSE] == 0)) / 14
  mu <- c(2,-5,3,-4,5,6)
  mub <- sqrt(rowSums(sweep(mat[, paste0("a", 1:6)], 2, mu)^2))
  Verr <- rep(NA, nrow(mat))
  if (do_V) { V <- covariance_matrix_gamma(mu, gamma_V)
    for (i in 1:nrow(mat)) Verr[i] <- sqrt(sum((V - covariance_matrix_gamma(mat[i, paste0("a",1:6)], mat[i, bcols]))^2)) }
  lam <- mat[, "lambda"]
  out <- t(sapply(sort(unique(lam)), function(l) { i <- lam == l
    c(lambda = l, SE = trim_mean(SE[i]), SP = 1 - trim_mean(1 - SP[i]), CR = trim_mean(CR[i]),
      mu_bias = trim_mean(mub[i]), V_err = if (do_V) trim_mean(Verr[i]) else NA) }))
  cat(sprintf("\n== %s [%s]\n", basename(file), label)); print(round(t(out), 3))
}
g_exp2 <- c(5,3,0,0,0,0,0,0,0,0,0,0,0,0); g_v2 <- c(5,3,0,5,1,0,0,0,0,0,0,0,0,0)
metrics(file.path(dir, "g1_GLASSO_k100.RData"), g_v2, g_v2, "group v1 data, scored with v2 truth (metrics & V)")
metrics(file.path(dir, "g1_GLASSO_k100.RData"), g_exp2, g_exp2, "group v1 data, scored with its own truth")
metrics(file.path(dir, "g1_RoCG_k100.RData"), g_v2, g_v2, "group v1 RoC-GLASSO data, scored with v2 truth")
