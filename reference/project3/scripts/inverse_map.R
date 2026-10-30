# R port of esag-rs param::angles_to_gamma / gamma_from_covariance (inverse of the user's parameterization)
angles_to_gamma <- function(lambda, r) {
  k <- length(lambda); d <- k + 1; m <- r; gamma <- rep(0, (d - 2) * (d + 1) / 2)
  plane <- function(mat, p, cc, ss) { x <- mat[, p]; y <- mat[, p + 1]; mat[, p] <- cc * x + ss * y; mat[, p + 1] <- cc * y - ss * x; mat }
  for (gr in (d - 2):1) {
    n <- gr + 1; v <- m[1:n, n]; nc <- max(n - 2, 0); cc <- rep(1, nc); ss <- rep(0, nc)
    if (nc > 0) for (j in 0:(nc - 1)) {
      idx <- gr - j; tl <- sqrt(sum(v[1:(idx + 1)]^2)); sg <- if (j %% 2 == 0) 1 else -1
      cj <- if (tl > 0) max(-1, min(1, sg * v[idx + 1] / tl)) else 1
      cc[j + 1] <- cj; ss[j + 1] <- sqrt(max(0, 1 - cj^2))
    }
    sigma <- if ((gr - 1) %% 2 == 0) 1 else -1
    theta <- if (v[1] == 0 && v[2] == 0) 0 else atan2(-sigma * v[1], sigma * v[2])
    radius <- max(0, lambda[gr + 1] / lambda[gr] - 1); off <- gr * (gr + 1) / 2 - 1; prod_s <- 1
    if (nc > 0) for (j in 0:(nc - 1)) { gamma[off + j + 1] <- radius * prod_s * cc[j + 1]; prod_s <- prod_s * ss[j + 1] }
    gamma[off + n - 1] <- radius * prod_s * cos(theta); gamma[off + n] <- radius * prod_s * sin(theta)
    a <- diag(k); a <- plane(a, 1, cos(theta), sin(theta))
    if (gr >= 2) for (j in 1:(gr - 1)) { jj <- gr - j - 1; a <- plane(a, j + 1, cc[jj + 1], ss[jj + 1]) }
    m <- t(a) %*% m
  }
  gamma
}
# all sign-equivalent gamma representatives of (mu, V): eigenvector column signs with det(R) = +1
gamma_reps <- function(mu, Vm) {
  d <- length(mu); k <- d - 1; U <- ONB(mu); Uk <- U[, 1:k]
  Tm <- t(Uk) %*% Vm %*% Uk; Tm <- (Tm + t(Tm)) / 2; e <- eigen(Tm, symmetric = TRUE)
  o <- order(e$values); lam <- e$values[o]; lam <- exp(log(lam) - mean(log(lam))); Q <- e$vectors[, o]
  reps <- list()
  for (s in 0:(2^k - 1)) {
    sg <- ifelse(bitwAnd(s, 2^(0:(k - 1))) > 0, -1, 1); R <- Q %*% diag(sg)
    if (det(R) < 0) next
    reps[[length(reps) + 1]] <- angles_to_gamma(lam, R)
  }
  reps
}
