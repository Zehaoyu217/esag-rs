set.seed(2)
pava <- function(v) { # nondecreasing isotonic regression, equal weights
  n <- length(v); val <- v; w <- rep(1, n); blk <- as.list(1:n); i <- 1
  vals <- c(); ws <- c(); idx <- list()
  for (j in 1:n) { vals <- c(vals, v[j]); ws <- c(ws, 1); idx[[length(idx) + 1]] <- j
    while (length(vals) > 1 && vals[length(vals) - 1] > vals[length(vals)]) {
      m <- length(vals); nv <- (vals[m-1]*ws[m-1] + vals[m]*ws[m]) / (ws[m-1] + ws[m])
      vals <- c(vals[seq_len(m-2)], nv); ws2 <- ws[m-1] + ws[m]; ws <- c(ws[seq_len(m-2)], ws2)
      idx <- c(idx[seq_len(m-2)], list(c(idx[[m-1]], idx[[m]]))) } }
  out <- numeric(n); for (b in seq_along(idx)) out[idx[[b]]] <- vals[b]; out }
Pgap <- function(X, w) { s <- sort(eigen(X, symmetric = TRUE, only.values = TRUE)$values); sum(w * diff(s)) }
prox_gap <- function(Z, t, w) { e <- eigen(Z, symmetric = TRUE); o <- order(e$values); zeta <- e$values[o]; Q <- e$vectors[, o]
  k <- length(zeta); cc <- numeric(k); cc[1] <- -w[1]; if (k > 2) for (j in 2:(k-1)) cc[j] <- w[j-1] - w[j]; cc[k] <- w[k-1]
  x <- pava(zeta - t * cc); Q %*% diag(x) %*% t(Q) }
sym_from <- function(p, k) { M <- matrix(0, k, k); M[upper.tri(M, diag = TRUE)] <- p; M + t(M) - diag(diag(M)) }
k <- 4; worst <- 0
for (trial in 1:30) {
  A <- matrix(rnorm(k*k), k); Z <- (A + t(A)) / 2; Z <- Z - diag(sum(diag(Z)) / k, k)
  w <- runif(k - 1, 0, 2); if (trial %% 3 == 0) w <- c(3, 0.1, 2)   # deliberately non-concave
  t <- runif(1, 0.1, 1.5)
  X <- prox_gap(Z, t, w); fprox <- 0.5 * sum((X - Z)^2) + t * Pgap(X, w)
  obj <- function(p) { M <- sym_from(p, k); 0.5 * sum((M - Z)^2) + t * Pgap(M, w) }
  best <- Inf
  for (s in 1:12) { p0 <- Z[upper.tri(Z, diag = TRUE)] + rnorm(k*(k+1)/2, sd = 0.5); o <- optim(p0, obj, method = "Nelder-Mead", control = list(maxit = 20000, reltol = 1e-14)); best <- min(best, o$value) }
  worst <- max(worst, fprox - best)
}
cat(sprintf("spectral-gap prox: max over 30 trials of [f(prox) - best brute-force f] = %.2e (<= ~0 means prox is optimal)\n", worst))
cat(sprintf("trace preserved: %.1e\n", abs(sum(diag(prox_gap(Z, 0.7, c(1, 2, 0.5)))) - sum(diag(Z)))))
# nested-prefix group prox: groups H_m = first n_m coords (prefix), composition from smallest to largest
bst <- function(v, idx, tau) { nv <- sqrt(sum(v[idx]^2)); if (nv <= tau) v[idx] <- 0 else v[idx] <- v[idx] * (1 - tau / nv); v }
sizes <- c(2, 5, 9)   # prefixes of gamma groups (2), (2+3), (2+3+4) in a d = 5 toy
worst2 <- 0
for (trial in 1:40) {
  v <- rnorm(9, sd = 1.5); w <- runif(3, 0.1, 1.5); t <- runif(1, 0.2, 1)
  x <- v; for (m in 1:3) x <- bst(x, 1:sizes[m], t * w[m])
  obj2 <- function(z) 0.5 * sum((z - v)^2) + t * sum(sapply(1:3, function(m) w[m] * sqrt(sum(z[1:sizes[m]]^2))))
  best <- Inf; for (s in 1:10) { o <- optim(v + rnorm(9, sd = 0.3), obj2, method = "BFGS", control = list(maxit = 10000, reltol = 1e-15)); best <- min(best, o$value) }
  worst2 <- max(worst2, obj2(x) - best)
}
cat(sprintf("nested-prefix prox (composition small->large): max [f(prox) - best numeric f] = %.2e\n", worst2))
