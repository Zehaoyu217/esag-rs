# Discontinuities of the gamma -> V map, checked with the original R functions
# (ONB, parameter, rotation, covariance_matrix, covariance_matrix_gamma from
# reference/r/ESAG_Project2.R). Run from the repository root:
#   Rscript reference/checks/gamma_discontinuity.R
# V is continuous when group 1 or all of gamma shrinks to zero, but jumps by a
# fixed amount (however small eps is) when a whole group j >= 2, or the last
# two entries of a group j >= 2, shrink to zero along different directions.
# Both simulation truths of the Project 3 paper sit at such points.
src <- readLines("reference/r/ESAG_Project2.R")
eval(parse(text = src[10:178]))

mu <- c(1, 2, -1, 0.5, 1.5)
base <- c(0.4, -0.7, 1.2, 0.6, -0.3, 0.3, 0.2, 0.5, -0.4)
cases <- list("group 1" = 1:2, "group 2" = 3:5, "group 3" = 6:9,
              "trailing pair of group 2" = 4:5, "all of gamma" = 1:9)
for (nm in names(cases)) {
  idx <- cases[[nm]]
  for (eps in c(1e-3, 1e-6, 1e-9)) {
    V <- lapply(0:3, function(s) {
      g <- base
      g[idx] <- eps * sin(((s * 7) + (seq_along(idx) - 1) * 3) * 1.3)
      covariance_matrix_gamma(mu, g)
    })
    jump <- max(sapply(2:4, function(k) max(abs(V[[k]] - V[[1]]))))
    cat(sprintf("%-26s -> 0, eps = %.0e: max |V(dir) - V(dir0)| = %.3e\n", nm, eps, jump))
  }
}

mu6 <- c(2, -5, 3, -4, 5, 6)
g_exp2 <- c(5, 3, rep(0, 12))
g_exp1 <- c(5, 2, 3, 0, 5, 1, 0, 1, 5, 0, 3, 2, 0, 0)
set.seed(1)
for (eps in c(1e-3, 1e-6, 1e-9)) {
  V0 <- covariance_matrix_gamma(mu6, g_exp2)
  j2 <- max(sapply(1:4, function(k) {
    g <- g_exp2; g[3:14] <- eps * rnorm(12); max(abs(covariance_matrix_gamma(mu6, g) - V0))
  }))
  V1 <- covariance_matrix_gamma(mu6, g_exp1)
  j1 <- max(sapply(1:4, function(k) {
    g <- g_exp1; g[13:14] <- eps * c(cos(k), sin(k)); max(abs(covariance_matrix_gamma(mu6, g) - V1))
  }))
  cat(sprintf("paper exp-2 truth, groups 2-4 nudged by %.0e: max |dV| = %.3f;  exp-1 truth, trailing pair of group 4 nudged: max |dV| = %.3f\n", eps, j2, j1))
}
