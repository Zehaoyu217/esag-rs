args <- commandArgs(trailingOnly = TRUE); basic <- args[1]; fx <- args[2]
real_library <- base::library
library <- function(package, ...) { p <- as.character(substitute(package)); if (p == "optimx") return(invisible(NULL)); real_library(p, character.only = TRUE, ...) }
source(basic); rm(library)
r <- readRDS(file.path(fx, "sim_RoCLASSO.rds")); Y <- r$Y; up <- r$unpenalized$par
ll <- function(par) { f <- 0; for (i in 1:nrow(Y)) f <- f + as.numeric(log_likelihood(Bx_i = par, Y[i, ])); f }
obj <- function(B, lam) { W <- abs(1 / up); W[1:6] <- 0; roc <- max(sqrt(sum(up[1:6]^2)) / sqrt(sum(B[1:6]^2)), 1); ll(B) - lam * sum(abs(W^roc * B)) }
for (lam in c(16, 128)) {
  f <- r$fits[[paste0("lambda_", lam)]]; p0 <- if (f$objective_sum_scale >= obj(f$stored_paper_run$par_untruncated, lam)) f$par_untruncated else f$stored_paper_run$par_untruncated
  o0 <- obj(p0, lam)
  b <- optim(p0, obj, lam = lam, control = list(fnscale = -1, maxit = 10000, reltol = 1e-7), method = "BFGS")
  nm <- optim(b$par, obj, lam = lam, control = list(fnscale = -1, maxit = 20000, reltol = 1e-12), method = "Nelder-Mead")
  b2 <- optim(nm$par, obj, lam = lam, control = list(fnscale = -1, maxit = 10000, reltol = 1e-10), method = "BFGS")
  cat(sprintf("RoC-LASSO lam=%g: best R point %.4f -> BFGS restart %.4f -> Nelder-Mead %.4f -> BFGS %.4f\n", lam, o0, b$value, nm$value, b2$value))
}
