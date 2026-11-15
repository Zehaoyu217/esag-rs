args <- commandArgs(trailingOnly = TRUE); basic <- args[1]
real_library <- base::library
library <- function(package, ...) { p <- as.character(substitute(package)); if (p == "optimx") return(invisible(NULL)); real_library(p, character.only = TRUE, ...) }
source(basic); rm(library)
B <- c(c(2,-5,3,-4,5,6), c(5,2,3,0,5,1,0,1,5,0,3,2,0,0)); set.seed(1); Y <- ESAG(100, B[1:6], B[7:20]); X <- matrix(1, 100, 1)
jll <- function(B, y = Y, x = X) { n <- nrow(y); d <- ncol(y); B_dim <- (d^2+d-2)/2; f <- rep(0, n); for (i in 1:n) { Bx <- t(matrix(B, nrow = B_dim) %*% t(x)); f[i] <- log_likelihood(Bx_i = Bx[i,], y[i,]) }; sum(f) }
t0 <- proc.time(); for (r in 1:20) v <- jll(B + 0.1); el <- (proc.time() - t0)[3] / 20
cat(sprintf("one joint log-lik eval (n=100,d=6): %.4f s -> one numerical gradient (40 evals) ~ %.2f s\n", el, 40 * el))
