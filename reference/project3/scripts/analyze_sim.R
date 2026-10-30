args <- commandArgs(trailingOnly = TRUE); basic <- args[1]; fx <- args[2]
real_library <- base::library
library <- function(package, ...) { p <- as.character(substitute(package)); if (p == "optimx") return(invisible(NULL)); real_library(p, character.only = TRUE, ...) }
source(basic); rm(library)
groups_B <- list(7:8, 9:11, 12:15, 16:20)
ll <- function(par, Y) { f <- 0; for (i in 1:nrow(Y)) f <- f + as.numeric(log_likelihood(Bx_i = par, Y[i, ])); f }
pen <- function(m, par, unp) switch(m,
  LASSO = sum(abs(par[7:20])),
  aLASSO = { W <- 1 / unp; W[1:6] <- 0; sum(abs(W * par)) },
  RoCLASSO = { W <- abs(1 / unp); W[1:6] <- 0; roc <- max(sqrt(sum(unp[1:6]^2)) / sqrt(sum(par[1:6]^2)), 1); sum(abs(W^roc * par)) },
  GLASSO = sum(sapply(groups_B, function(g) sqrt(sum(par[g]^2)))),
  aGLASSO = { W <- 1 / sapply(groups_B, function(g) sqrt(sum(unp[g]^2))); sum(W * sapply(groups_B, function(g) sqrt(sum(par[g]^2)))) },
  RoCGLASSO = { W <- 1 / sapply(groups_B, function(g) sqrt(sum(unp[g]^2))); roc <- max(sqrt(sum(unp[1:6]^2)) / sqrt(sum(par[1:6]^2)), 1)
                sum((W^roc) * sapply(groups_B, function(g) sqrt(sum(par[g]^2)))) })
for (m in c("LASSO","aLASSO","RoCLASSO","GLASSO","aGLASSO","RoCGLASSO")) {
  r <- readRDS(file.path(fx, paste0("sim_", m, ".rds"))); Y <- r$Y
  unp <- if (!is.null(r$unpenalized)) r$unpenalized$par else rep(NA, 20)
  for (f in r$fits) {
    lam <- f$lambda; s <- f$stored_paper_run$par_untruncated
    obj_stored <- ll(s, Y) - lam * pen(m, s, unp)
    cat(sprintf("%-9s lam=%-4g  obj(mine)=%.6f  obj(stored pt, my weights)=%.6f  diff(mine-stored)=%+.4f | ll_untrunc=%.3f ll_thresh=%.3f (drop %.3f)\n",
                m, lam, f$objective_sum_scale, obj_stored, f$objective_sum_scale - obj_stored, f$loglik_sum_untruncated, f$loglik_sum_thresholded,
                f$loglik_sum_untruncated - f$loglik_sum_thresholded))
  }
}
r <- readRDS(file.path(fx, "sim_aGLASSO.rds")); f <- r$fits$lambda_16
cat("\naGLASSO lambda=16: untruncated gamma =", paste(signif(f$par_untruncated[7:20], 3), collapse = ", "), "\n")
cat("                    thresholded zeros at gamma indices:", which(f$par_thresholded[7:20] == 0), "\n")
r <- readRDS(file.path(fx, "sim_GLASSO.rds")); f <- r$fits$lambda_128
cat("GLASSO lambda=128:  untruncated gamma =", paste(signif(f$par_untruncated[7:20], 3), collapse = ", "), "\n")
cat("                    ll_untrunc =", f$loglik_sum_untruncated, " ll_thresh =", f$loglik_sum_thresholded, "\n")
