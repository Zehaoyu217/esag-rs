args <- commandArgs(trailingOnly = TRUE); basic <- args[1]; fx <- args[2]
real_library <- base::library
library <- function(package, ...) { p <- as.character(substitute(package)); if (p == "optimx") return(invisible(NULL)); real_library(p, character.only = TRUE, ...) }
source(basic); rm(library); real_library(parallel)
r <- readRDS(file.path(fx, "sim_aLASSO.rds")); Y <- r$Y; X <- matrix(1, nrow(Y), 1)
jll <- function(B, y = Y, x = X) { n <- nrow(y); d <- ncol(y); B_dim <- (d^2+d-2)/2; f <- rep(0, n); for (i in 1:n) { Bx <- t(matrix(B, nrow = B_dim) %*% t(x)); f[i] <- log_likelihood(Bx_i = Bx[i,], y[i,]) }; sum(f) }
oracle <- r$unpenalized
set.seed(7)
starts <- c(list(application_style = c(rep(1, 6), rep(0, 14))), lapply(1:5, function(i) c(rnorm(6, sd = 3), rnorm(14, sd = 2))))
names(starts)[2:6] <- paste0("random_", 1:5)
fits <- mclapply(starts, function(s) optim(par = s, jll, control = list(fnscale = -1, maxit = 10000), method = "BFGS"), mc.cores = 6)
Vor <- covariance_matrix_gamma(oracle$par[1:6], oracle$par[7:20])
cat(sprintf("%-18s loglik=%.4f conv=%d  |mu - mu_or|=%.2e  max|V - V_or|=%.2e  max|gamma - gamma_or|=%.3f  #|gamma|<0.05: %d\n", "oracle (truth+0.1)", oracle$value, oracle$convergence, 0, 0, 0, sum(abs(oracle$par[7:20]) < 0.05)))
for (nm in names(fits)) { f <- fits[[nm]]; Vf <- covariance_matrix_gamma(f$par[1:6], f$par[7:20])
  cat(sprintf("%-18s loglik=%.4f conv=%d  |mu - mu_or|=%.2e  max|V - V_or|=%.2e  max|gamma - gamma_or|=%.3f  #|gamma|<0.05: %d\n", nm, f$value, f$convergence,
      sqrt(sum((f$par[1:6] - oracle$par[1:6])^2)), max(abs(Vf - Vor)), max(abs(f$par[7:20] - oracle$par[7:20])), sum(abs(f$par[7:20]) < 0.05))) }
cat("oracle-start gamma: ", paste(round(oracle$par[7:20], 3), collapse = ", "), "\n")
best <- fits[[which.max(sapply(fits, function(f) f$value))]]
cat("best other gamma:   ", paste(round(best$par[7:20], 3), collapse = ", "), "\n")
