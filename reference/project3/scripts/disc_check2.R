args <- commandArgs(trailingOnly = TRUE); basic <- args[1]
real_library <- base::library
library <- function(package, ...) { p <- as.character(substitute(package)); if (p == "optimx") return(invisible(NULL)); real_library(p, character.only = TRUE, ...) }
source(basic); rm(library)
V <- function(mu, g) covariance_matrix_gamma(mu, g)
sweep_t <- function(label, mu, make, ts = c(0, 1, 2, 3)) {
  cat(sprintf("\n%s\n", label))
  for (e in c(1e-2, 1e-4, 1e-8)) {
    Vs <- lapply(ts, function(t) V(mu, make(e, t)))
    cat(sprintf("  eps=%-6g max_t |V(t)-V(0)| = %.4g\n", e, max(sapply(Vs, function(M) max(abs(M - Vs[[1]]))))))
  }
}
mu5 <- c(1, 2, -1, 0.5, 1.5)
for (g1 in list(c(0.8, -0.6), c(0, 0), c(1.5, 0.4))) {
  sweep_t(sprintf("d=5, group1=(%g,%g): ONLY group 2 trailing pair -> 0: (1.2, e cos t, e sin t); group3=(0.3,-0.5,0.4,0.2)", g1[1], g1[2]), mu5,
          function(e, t) c(g1, 1.2, e*cos(t), e*sin(t), 0.3, -0.5, 0.4, 0.2))
  sweep_t(sprintf("d=5, group1=(%g,%g): ONLY group 3 trailing pair -> 0: (0.3,-0.5, e cos t, e sin t); group2=(1.2,0.3,-0.7)", g1[1], g1[2]), mu5,
          function(e, t) c(g1, 1.2, 0.3, -0.7, 0.3, -0.5, e*cos(t), e*sin(t)))
  sweep_t(sprintf("d=5, group1=(%g,%g): both trailing pairs -> 0 with INDEPENDENT angles (t, 2t+1)", g1[1], g1[2]), mu5,
          function(e, t) c(g1, 1.2, e*cos(t), e*sin(t), 0.3, -0.5, e*cos(2*t+1), e*sin(2*t+1)))
}
# Group 2 trailing pair with all lower groups zero: is the trailing-pair set still a discontinuity?
sweep_t("d=5, group1=0, group2 trailing pair -> 0, group3 = 0 (lower block isotropic)", mu5,
        function(e, t) c(0, 0, 1.2, e*cos(t), e*sin(t), 0, 0, 0, 0))
