# Which group-zero patterns are continuity points of gamma -> V, and are they differentiable?
args <- commandArgs(trailingOnly = TRUE); basic <- args[1]
real_library <- base::library
library <- function(package, ...) { p <- as.character(substitute(package)); if (p == "optimx") return(invisible(NULL)); real_library(p, character.only = TRUE, ...) }
source(basic); rm(library)
V <- function(mu, g) covariance_matrix_gamma(mu, g)
gi <- list(1:2, 3:5, 6:9, 10:14)                      # d = 6 groups
mu6 <- c(2, -5, 3, -4, 5, 6)
base <- c(0.4, -0.7, 1.2, 0.6, -0.3, 0.3, 0.2, 0.5, -0.4, 0.9, -0.6, 0.35, 0.8, -0.45)
pattern_jump <- function(zero_groups, eps) {
  g0 <- base; for (j in zero_groups) g0[gi[[j]]] <- 0
  set.seed(42); Vs <- lapply(1:6, function(s) { g <- g0; for (j in zero_groups) { u <- rnorm(length(gi[[j]])); g[gi[[j]]] <- eps * u / sqrt(sum(u^2)) }; V(mu6, g) })
  max(sapply(2:6, function(k) max(abs(Vs[[k]] - Vs[[1]]))))
}
pats <- list(c(1), c(1,2), c(1,2,3), c(2), c(3), c(4), c(1,3), c(2,3), c(2,3,4), c(1,2,4), c(1,2,3,4))
cat("d=6, base gamma generic; zero groups approached from 6 random directions: max|V(u_s)-V(u_1)|\n")
for (p in pats) cat(sprintf("  zero groups {%s}%s: eps=1e-3 %.2e | eps=1e-6 %.2e | eps=1e-9 %.2e\n", paste(p, collapse=","),
  if (identical(p, seq_len(max(p)))) " [prefix]" else "        ", pattern_jump(p, 1e-3), pattern_jump(p, 1e-6), pattern_jump(p, 1e-9)))
# differentiability at prefix zeros: is u -> D_u V linear?  (D_u V = lim (V(g0 + eps u) - V(g0))/eps)
dlin <- function(zero_groups, j) {   # test linearity in group j's coordinates (first two coords of group j)
  g0 <- base; for (k in zero_groups) g0[gi[[k]]] <- 0; e <- 1e-7; V0 <- V(mu6, g0)
  D <- function(u) { g <- g0; g[gi[[j]]] <- g[gi[[j]]] + e * u; (V(mu6, g) - V0) / e }
  m <- length(gi[[j]]); e1 <- c(1, rep(0, m - 1)); e2 <- c(0, 1, rep(0, m - 2))
  c(nonlin = max(abs(D(e1) + D(e2) - sqrt(2) * D((e1 + e2) / sqrt(2)))), scale = max(abs(D(e1))))
}
cat("\nlinearity defect of the directional derivative (0 = differentiable):\n")
for (case in list(list(z = c(1), j = 1), list(z = c(1,2), j = 2), list(z = c(1,2,3,4), j = 4), list(z = integer(0), j = 2))) {
  r <- dlin(case$z, case$j); cat(sprintf("  zero groups {%s}, perturb group %d: defect %.3f (|D_e1 V| = %.3f)\n", paste(case$z, collapse=","), case$j, r[1], r[2]))
}
