args <- commandArgs(trailingOnly = TRUE); basic <- args[1]
real_library <- base::library
library <- function(package, ...) { p <- as.character(substitute(package)); if (p == "optimx") return(invisible(NULL)); real_library(p, character.only = TRUE, ...) }
source(basic); rm(library)
V <- function(mu, g) covariance_matrix_gamma(mu, g)
# (a) differentiability at gamma = 0 (d = 3): is the directional derivative linear in the direction?
mu3 <- c(1, 2, 2); eps <- 1e-7
D <- function(u) (V(mu3, eps * u) - diag(3)) / eps
lin_err <- max(abs(D(c(1, 0)) + D(c(0, 1)) - sqrt(2) * D(c(1, 1) / sqrt(2))))
cat(sprintf("(a) d=3: max|D_e1 V + D_e2 V - sqrt2 D_(e1+e2)/sqrt2 V| = %.4f  (0 if differentiable)\n", lin_err))
cat(sprintf("    ||D_e1 V||=%.4f  ||D_e2 V||=%.4f  ||D_e1 V + D_e2 V||=%.2e (D_{-u} = D_u up to the 2theta symmetry)\n",
            sqrt(sum(D(c(1,0))^2)), sqrt(sum(D(c(0,1))^2)), sqrt(sum((D(c(1,0)) + D(c(0,1)))^2))))
# (b) representability (d = 4): uniaxial tangent V with a generic eigenvector is NOT in the image of gamma -> V
mu4 <- c(0.5, 1, -1, 2); U <- ONB(mu4); Ut <- U[, 1:3]
w <- c(1, 1, 1) / sqrt(3); l1 <- 0.4; l2 <- 1 / sqrt(l1)          # eigenvalues l1 < l2 = l3, product 1
Vt <- l2 * diag(3) - (l2 - l1) * w %*% t(w)
Vstar <- Ut %*% Vt %*% t(Ut) + (mu4 / sqrt(sum(mu4^2))) %*% t(mu4 / sqrt(sum(mu4^2)))
obj <- function(g) sum((V(mu4, g) - Vstar)^2)
best <- NULL
set.seed(3)
for (s in 1:40) { o <- optim(rnorm(5), obj, method = "BFGS", control = list(maxit = 5000, reltol = 1e-16)); if (is.null(best) || o$value < best$value) best <- o }
g <- best$par
cat(sprintf("(b) d=4 uniaxial target, w=(1,1,1)/sqrt3 in U-frame: best ||V(g)-V*||_F^2 = %.3e at |group1|=%.4f, |group2|=%.3e\n",
            best$value, sqrt(sum(g[1:2]^2)), sqrt(sum(g[3:5]^2))))
w2 <- c(0.6, 0.8, 0); Vt2 <- l2 * diag(3) - (l2 - l1) * w2 %*% t(w2)
Vstar2 <- Ut %*% Vt2 %*% t(Ut) + (mu4 / sqrt(sum(mu4^2))) %*% t(mu4 / sqrt(sum(mu4^2)))
obj2 <- function(g) sum((V(mu4, g) - Vstar2)^2); best2 <- NULL
for (s in 1:20) { o <- optim(rnorm(5), obj2, method = "BFGS", control = list(maxit = 5000, reltol = 1e-16)); if (is.null(best2) || o$value < best2$value) best2 <- o }
cat(sprintf("    same eigenvalues, w=(0.6,0.8,0) in span(u1,u2): best ||V(g)-V*||_F^2 = %.3e, gamma = %s\n", best2$value, paste(round(best2$par, 4), collapse=",")))
# (c) permutation of coordinates: zeros of gamma are not preserved
mu6 <- c(2, -5, 3, -4, 5, 6); g1 <- c(5,2,3,0,5,1,0,1,5,0,3,2,0,0)
V0 <- V(mu6, g1); perm <- c(2, 1, 3, 4, 5, 6); P <- diag(6)[perm, ]
mup <- as.vector(P %*% mu6); Vp <- P %*% V0 %*% t(P)
objp <- function(g) sum((V(mup, g) - Vp)^2); bestp <- NULL
set.seed(5)
for (s in 1:60) { o <- optim(rnorm(14, sd = 2), objp, method = "BFGS", control = list(maxit = 20000, reltol = 1e-16)); if (is.null(bestp) || o$value < bestp$value) bestp <- o }
cat(sprintf("(c) exp-1 truth with coordinates 1,2 swapped: best ||V(g)-V_perm||_F^2 = %.2e\n    gamma_perm = %s\n    min |gamma_perm_k| = %.3f (original truth has 5 exact zeros)\n",
            bestp$value, paste(round(bestp$par, 3), collapse = ", "), min(abs(bestp$par))))
