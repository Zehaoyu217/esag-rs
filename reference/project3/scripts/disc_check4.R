args <- commandArgs(trailingOnly = TRUE); basic <- args[1]; inv <- args[2]
real_library <- base::library
library <- function(package, ...) { p <- as.character(substitute(package)); if (p == "optimx") return(invisible(NULL)); real_library(p, character.only = TRUE, ...) }
source(basic); rm(library); source(inv)
V <- function(mu, g) covariance_matrix_gamma(mu, g)
mu6 <- c(2, -5, 3, -4, 5, 6); g1 <- c(5,2,3,0,5,1,0,1,5,0,3,2,0,0); V0 <- V(mu6, g1)
# roundtrip check of the port on the truth itself
set.seed(11); perms <- c(list(c(6,5,4,3,2,1)), lapply(1:7, function(i) sample(6)))
for (perm in perms) {
  P <- diag(6)[perm, ]; mup <- as.vector(P %*% mu6); Vp <- P %*% V0 %*% t(P)
  reps <- lapply(gamma_reps(mup, Vp), function(g) ifelse(abs(g) < 1e-6, 0, g))
  err <- sapply(reps, function(g) max(abs(V(mup, g) - Vp))); ok <- which(err < 1e-8)
  nz <- sapply(reps, function(g) sum(abs(g) < 1e-6))
  if (length(ok) == 0) { cat(sprintf("perm (%s): inverse not exact (permuted truth still on a singular set)\n", paste(perm, collapse = ""))); next }
  best <- reps[[ok[which.max(nz[ok])]]]
  cat(sprintf("perm (%s): %d/%d sign-equivalent reps reproduce V; max #zeros among them = %d; min |gamma_k| over that rep = %.3f\n",
              paste(perm, collapse = ""), length(ok), length(reps), max(nz[ok]), min(abs(best))))
}
# (b) exact representability in d=4: uniaxial V* with eigenvector w in U-coordinates
mu4 <- c(0.5, 1, -1, 2); U <- ONB(mu4); Ut <- U[, 1:3]; mh <- mu4 / sqrt(sum(mu4^2))
l1 <- 0.4; l2 <- 1 / sqrt(l1)
mk <- function(w) { w <- w / sqrt(sum(w^2)); Ut %*% (l2 * diag(3) - (l2 - l1) * w %*% t(w)) %*% t(Ut) + mh %*% t(mh) }
th <- atan2(0.8, 0.6); r <- l2 / l1 - 1
cat(sprintf("\n(b) w in span(u1,u2): ||V(g*) - V*||_max with g* = (r cos th, r sin th, 0,0,0) = %.2e  [exactly representable]\n",
            max(abs(V(mu4, c(r * cos(th), r * sin(th), 0, 0, 0)) - mk(c(0.6, 0.8, 0))))))
Vs <- mk(c(1, 1, 1))
best0 <- min(sapply(seq(0, 2 * pi, length.out = 2001), function(t) max(abs(V(mu4, c(r * cos(t), r * sin(t), 0, 0, 0)) - Vs))))
cat(sprintf("    w=(1,1,1)/sqrt3, group 2 exactly 0: min over theta1 of ||V - V*||_max = %.3f  [not representable with a zero group]\n", best0))
for (e in c(1e-2, 1e-4, 1e-6)) {
  f <- function(p) { g <- c(p[1], p[2], e * c(cos(p[3]), sin(p[3]) * cos(p[4]), sin(p[3]) * sin(p[4]))); sum((V(mu4, g) - Vs)^2) }
  bb <- NULL; set.seed(1)
  for (s in 1:30) { o <- optim(c(rnorm(2, sd = 2), runif(2, 0, pi)), f, method = "BFGS", control = list(maxit = 2000, reltol = 1e-15)); if (is.null(bb) || o$value < bb$value) bb <- o }
  cat(sprintf("    group 2 = %g * u(angles), best ||V - V*||_F = %.2e\n", e, sqrt(bb$value)))
}
