# Check d ell / d S for ell = log C - (d/2) log q + 0.5 (alpha^2 - |mu|^2) + log M_{d-1}(alpha),
# q = (muhat'y)^2 + z' exp(-S) z, z = Xi' y, alpha = y'mu / sqrt(q).
set.seed(4); d <- 6; k <- d - 1
Mp <- function(a, p) { M <- numeric(p); M[1] <- a*pnorm(a) + dnorm(a); M[2] <- (1 + a^2)*pnorm(a) + a*dnorm(a); if (p >= 3) for (i in 3:p) M[i] <- a*M[i-1] + (i-1)*M[i-2]; M }
logM <- function(a) log(Mp(a, d - 1)[d - 1])
dlogM <- function(a) { M <- Mp(a, d - 1); (d - 1) * M[d - 2] / M[d - 1] }      # (log M_p)' = p M_{p-1}/M_p
mu <- c(2, -5, 3, -4, 5, 6); mh <- mu / sqrt(sum(mu^2))
Xi <- qr.Q(qr(cbind(mh, matrix(rnorm(d * k), d))))[, 2:d]                     # any orthonormal basis of mu-perp
y <- rnorm(d); y <- y / sqrt(sum(y^2)); z <- as.vector(t(Xi) %*% y)
A <- matrix(rnorm(k * k), k); S <- (A + t(A)) / 4; S <- S - diag(sum(diag(S)) / k, k)
expmS <- function(S) { e <- eigen(S, symmetric = TRUE); e$vectors %*% diag(exp(-e$values)) %*% t(e$vectors) }
ell <- function(S) { q <- sum(mh * y)^2 + as.numeric(t(z) %*% expmS(S) %*% z); a <- sum(y * mu) / sqrt(q)
  -(d/2) * log(q) + 0.5 * (a^2 - sum(mu^2)) + logM(a) }
q <- sum(mh * y)^2 + as.numeric(t(z) %*% expmS(S) %*% z); a <- sum(y * mu) / sqrt(q)
dl_dq <- -(1 / (2 * q)) * (d + a^2 + a * dlogM(a))
e <- eigen(S, symmetric = TRUE); Q <- e$vectors; sg <- e$values
F <- outer(sg, sg, function(x, w) ifelse(abs(x - w) > 1e-12, (exp(-x) - exp(-w)) / (x - w), -exp(-x)))
dq_dS <- Q %*% (F * (t(Q) %*% (z %*% t(z)) %*% Q)) %*% t(Q)
G <- dl_dq * dq_dS
# numerical derivative along random symmetric directions H: d/dt ell(S + tH) vs <G, H>
worst <- 0
for (r in 1:10) { H <- matrix(rnorm(k*k), k); H <- (H + t(H)) / 2; h <- 1e-6
  num <- (ell(S + h*H) - ell(S - h*H)) / (2*h); ana <- sum(G * H); worst <- max(worst, abs(num - ana) / max(1, abs(num))) }
cat(sprintf("max relative error of analytic dl/dS (F_ab = (e^-sa - e^-sb)/(sa - sb), F_aa = -e^-sa): %.2e\n", worst))
