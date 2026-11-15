# Numerical check of discontinuities of (mu, gamma) -> V in the user's R parameterization (Basic_Final.R).
args <- commandArgs(trailingOnly = TRUE); basic <- args[1]
real_library <- base::library
library <- function(package, ...) { p <- as.character(substitute(package)); if (p == "optimx") return(invisible(NULL)); real_library(p, character.only = TRUE, ...) }
source(basic); rm(library)
V <- function(mu, g) covariance_matrix_gamma(mu, g)
jump <- function(label, mu0, g0, make, ts = c(0, 1, 2, 3), eps = c(1e-2, 1e-4, 1e-8), mu_make = NULL) {
  V0 <- V(mu0, g0)
  cat(sprintf("\n%s\n", label))
  for (e in eps) {
    Vs <- lapply(ts, function(t) { if (is.null(mu_make)) V(mu0, make(e, t)) else V(mu_make(e, t), g0) })
    d_t  <- max(sapply(Vs, function(M) max(abs(M - Vs[[1]]))))
    d_0  <- max(sapply(Vs, function(M) max(abs(M - V0))))
    cat(sprintf("  eps=%-6g  max_t |V(t)-V(t=0)| = %.4g   max_t |V(t)-V(limit pt)| = %.4g\n", e, d_t, d_0))
  }
}
set.seed(1)
# 1. coordinator's example: d=5, trailing pairs of groups 2 and 3 -> 0 from direction t
mu5 <- c(1, 2, -1, 0.5, 1.5); g1 <- c(0.8, -0.6)
jump("1. d=5, group2=(1.2, e cos t, e sin t), group3=(0.3,-0.5, e cos t, e sin t) [trailing pair -> 0, group nonzero]",
     mu5, c(g1, 1.2, 0, 0, 0.3, -0.5, 0, 0),
     function(e, t) c(g1, 1.2, e*cos(t), e*sin(t), 0.3, -0.5, e*cos(t), e*sin(t)))
# 2. exp-1 truth: group 4 = (0,3,2,0,0) lies on the set; approach (0,3,2, e cos t, e sin t)
mu6 <- c(2, -5, 3, -4, 5, 6); gexp1 <- c(5,2,3,0,5,1,0,1,5,0,3,2,0,0)
jump("2. exp-1 truth, group 4 = (0,3,2, e cos t, e sin t)", mu6, gexp1,
     function(e, t) { g <- gexp1; g[13:14] <- e*c(cos(t), sin(t)); g })
# 3. exp-2 truth: groups 2..4 zero while group 1 = (5,3) nonzero; approach group 2 -> e*u(t) (unit 3-vector)
gexp2 <- c(5,3,0,0,0,0,0,0,0,0,0,0,0,0)
u3 <- function(t) { v <- c(cos(t), sin(t)*cos(2*t+0.3), sin(t)*sin(2*t+0.3)); v / sqrt(sum(v^2)) }
jump("3. exp-2 truth, group 2 = e*u(t) (whole zero group above a NONZERO group 1)", mu6, gexp2,
     function(e, t) { g <- gexp2; g[3:5] <- e*u3(t+0.5); g })
jump("3b. exp-2 truth, group 4 = e*u(t) (whole zero top group)", mu6, gexp2,
     function(e, t) { g <- gexp2; v <- c(cos(t+0.4), sin(t+0.4), 0.5*cos(t), 0.3, -0.2*sin(t)); g[10:14] <- e*v/sqrt(sum(v^2)); g })
# 4. bottom-nested zeros: groups 1..3 -> 0 (all lower groups zero), group 4 nonzero
g4only <- c(0,0, 0,0,0, 0,0,0,0, 1,-2,0.5,1.5,0.7)
jump("4. bottom-nested: groups 1,2,3 = e*u_j(t) (all lower groups zero), group 4 nonzero", mu6, g4only,
     function(e, t) { g <- g4only; g[1:2] <- e*c(cos(t), sin(t)); g[3:5] <- e*u3(t+1); v <- c(cos(2*t), sin(2*t), 0.3, -0.4); g[6:9] <- e*v/sqrt(sum(v^2)); g })
jump("4b. group 1 -> 0 alone (lowest group), groups 2..4 nonzero", mu6, c(0,0, gexp1[3:14]),
     function(e, t) c(e*cos(t), e*sin(t), gexp1[3:14]))
# 5. single zeros not forming a trailing pair (continuity expected)
jump("5a. exp-1 group 2 = (3, e cos t?, 5): single zero in middle coordinate, approach +-e", mu6, gexp1,
     function(e, t) { g <- gexp1; g[4] <- e*cos(t) + 1e-300; g })
jump("5b. exp-1 group 4 first coord -> 0: (e cos t, 3, 2, 0.4, 0.7) vs (0,3,2,0.4,0.7)", mu6, c(gexp1[1:9], 0, 3, 2, 0.4, 0.7),
     function(e, t) { g <- c(gexp1[1:9], e*cos(t), 3, 2, 0.4, 0.7); g })
# 6. frame U(mu) discontinuity: mu = (e cos t, e sin t, 3, -4, 5, 6) with an anisotropic gamma
jump("6. mu -> (0,0,3,-4,5,6) along (e cos t, e sin t, ...), gamma = exp-1", c(0,0,3,-4,5,6), gexp1, NULL,
     mu_make = function(e, t) c(e*cos(t), e*sin(t), 3, -4, 5, 6))
# 7. R conventions at the singular set
cat("\n7. R conventions: parameter(c(5,3, 0,0,0, 0,0,0,0, 0,3,2,0,0)) theta/phi:\n"); p <- parameter(c(5,3,0,0,0,0,0,0,0,0,3,2,0,0)); print(lapply(p, round, 6))
cat("  rotation(gamma) for exp-2 truth == identity block except group 1?\n"); print(round(rotation(gexp2), 6))
