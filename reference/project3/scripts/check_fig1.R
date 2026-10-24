args <- commandArgs(trailingOnly = TRUE); dir <- args[1]; basic <- args[2]
real_library <- base::library
library <- function(package, ...) { p <- as.character(substitute(package)); if (p == "optimx") return(invisible(NULL)); real_library(p, character.only = TRUE, ...) }
source(basic); rm(library)
trim_mean <- function(x) { ub <- quantile(x, 0.98); mean(x[x <= ub]) }
g_exp1 <- c(5,2,3,0,5,1,0,1,5,0,3,2,0,0); mu <- c(2,-5,3,-4,5,6)
V <- covariance_matrix_gamma(mu, g_exp1)
for (m in c("Regular_LASSO","Weighted_LASSO","RoC_Weighted_LASSO")) {
  e <- new.env(); load(file.path(dir, paste0("l2_", m, "_k50.RData")), envir = e); result <- e$result
  cls <- table(sapply(result, function(r) class(r)[1]))
  ok <- which(sapply(result, function(r) is.list(r) && !is.null(r$table)))
  mat <- do.call(rbind, lapply(result[ok], function(r) r$table[, 1:26]))
  bcols <- paste0("b", 1:14); nzt <- bcols[g_exp1 != 0]; zt <- bcols[g_exp1 == 0]
  SP <- rowSums(mat[, nzt] != 0) / 9; SE <- rowSums(mat[, zt] == 0) / 5
  CR <- (rowSums(mat[, nzt] != 0) + rowSums(mat[, zt] == 0)) / 14
  mub <- sqrt(rowSums(sweep(mat[, paste0("a", 1:6)], 2, mu)^2))
  Verr <- sapply(1:nrow(mat), function(i) sqrt(sum((V - covariance_matrix_gamma(mat[i, paste0("a",1:6)], mat[i, bcols]))^2)))
  rmse <- sapply(1:nrow(mat), function(i) { mu_i <- mat[i, paste0("a",1:6)]; Vi <- solve(covariance_matrix_gamma(mu_i, mat[i, bcols]))
     Y <- result[[mat[i, 1]]]$Y; P <- diag(6) - mu_i %*% t(mu_i) / sum(mu_i^2); R <- Y %*% P; sqrt(mean(rowSums((R %*% Vi) * R))) })
  lam <- mat[, "lambda"]
  out <- t(sapply(sort(unique(lam)), function(l) { i <- lam == l
    c(lambda = l, SE = trim_mean(SE[i]), SP = 1 - trim_mean(1 - SP[i]), CR = trim_mean(CR[i]), mu_bias = trim_mean(mub[i]),
      V_err = trim_mean(Verr[i]), rmse = trim_mean(rmse[i]), n_rep = sum(i)) }))
  cat(sprintf("\n== %s k50 v2 (classes: %s)\n", m, paste(names(cls), cls, collapse=","))); print(round(t(out), 3))
}
