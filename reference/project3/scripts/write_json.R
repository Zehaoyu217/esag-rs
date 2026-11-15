# Convert the fixture RDS files (from make_fixtures.R) into JSON for the Rust tests.
args <- commandArgs(trailingOnly = TRUE); basic <- args[1]; fx <- args[2]; outdir <- args[3]
real_library <- base::library
library <- function(package, ...) { p <- as.character(substitute(package)); if (p == "optimx") return(invisible(NULL)); real_library(p, character.only = TRUE, ...) }
source(basic); rm(library)
real_library(jsonlite)
J <- function(x, file) writeLines(toJSON(x, digits = I(17), auto_unbox = TRUE, na = "null", null = "null", matrix = "rowmajor", pretty = 1), file)
ll <- function(par, Y) { f <- 0; for (i in 1:nrow(Y)) f <- f + as.numeric(log_likelihood(Bx_i = par, Y[i, ])); f }
groups_B <- list(7:8, 9:11, 12:15, 16:20)
pen <- function(m, par, unp) switch(m,
  LASSO = sum(abs(par[7:20])),
  aLASSO = { W <- 1 / unp; W[1:6] <- 0; sum(abs(W * par)) },
  RoCLASSO = { W <- abs(1 / unp); W[1:6] <- 0; roc <- max(sqrt(sum(unp[1:6]^2)) / sqrt(sum(par[1:6]^2)), 1); sum(abs(W^roc * par)) },
  GLASSO = sum(sapply(groups_B, function(g) sqrt(sum(par[g]^2)))),
  aGLASSO = { W <- 1 / sapply(groups_B, function(g) sqrt(sum(unp[g]^2))); sum(W * sapply(groups_B, function(g) sqrt(sum(par[g]^2)))) },
  RoCGLASSO = { W <- 1 / sapply(groups_B, function(g) sqrt(sum(unp[g]^2))); roc <- max(sqrt(sum(unp[1:6]^2)) / sqrt(sum(par[1:6]^2)), 1)
                sum((W^roc) * sapply(groups_B, function(g) sqrt(sum(par[g]^2)))) })
pen_doc <- list(
  LASSO = "P(B) = sum_{k=7..20} |B_k|  (all 14 gamma entries, weight 1; mu unpenalized). LASSO_v2.R: LASSO_term = sum(abs(B[-(1:d)]))",
  aLASSO = "P(B) = sum_k |W_k B_k|, W = 1/unp$par with W[1:6] = 0, i.e. sum_{k in gamma} |gamma_k| / |gamma_tilde_k|. Weighted_LASSO_v2.R",
  RoCLASSO = "P(B) = sum_k |W_k^r B_k|, W = |1/unp$par|, W[1:6] = 0, r = max(||mu_tilde|| / ||mu||, 1) with mu = B[1:6] the CURRENT argument and mu_tilde = unp$par[1:6]. RoC_Weighted_LASSO_v2.R",
  GLASSO = "P(B) = sum_{j=1..4} ||B[G_j]||_2, G = (7:8, 9:11, 12:15, 16:20) (no sqrt(group size) factor). Group_LASSO_v3.R",
  aGLASSO = "P(B) = sum_j ||B[G_j]||_2 / ||unp$par[G_j]||_2. Group_Weighted_LASSO_v3.R",
  RoCGLASSO = "P(B) = sum_j (1/||unp$par[G_j]||_2)^r ||B[G_j]||_2, r = max(||mu_tilde||/||mu||, 1). Group_RoC_Weighted_LASSO_v3.R")
methods <- c("LASSO", "aLASSO", "RoCLASSO", "GLASSO", "aGLASSO", "RoCGLASSO")
R <- lapply(methods, function(m) readRDS(file.path(fx, paste0("sim_", m, ".rds")))); names(R) <- methods
Y <- R$LASSO$Y; n <- nrow(Y); truth <- R$LASSO$truth
for (m in methods) stopifnot(identical(R[[m]]$Y, Y))
unp <- R$aLASSO$unpenalized
for (m in c("RoCLASSO", "aGLASSO", "RoCGLASSO")) stopifnot(identical(R[[m]]$unpenalized$par, unp$par))
up <- unp$par
weights <- list(
  LASSO = list(kind = "elementwise", gamma_weights = rep(1, 14)),
  aLASSO = list(kind = "elementwise", gamma_weights = 1 / abs(up[7:20])),
  RoCLASSO = list(kind = "elementwise", base_gamma_weights = 1 / abs(up[7:20]), mu_tilde = up[1:6], mu_tilde_norm = sqrt(sum(up[1:6]^2)),
                  note = "effective weight = base^r with r = max(||mu_tilde||/||mu||, 1) recomputed at every evaluation"),
  GLASSO = list(kind = "group", group_weights = rep(1, 4)),
  aGLASSO = list(kind = "group", group_weights = 1 / sapply(groups_B, function(g) sqrt(sum(up[g]^2)))),
  RoCGLASSO = list(kind = "group", base_group_weights = 1 / sapply(groups_B, function(g) sqrt(sum(up[g]^2))), mu_tilde = up[1:6],
                   mu_tilde_norm = sqrt(sum(up[1:6]^2)), note = "effective weight = base^r, r = max(||mu_tilde||/||mu||, 1)"))
fitj <- function(m, f) {
  sp <- f$stored_paper_run$par_untruncated
  list(lambda = f$lambda, r_call = f$call,
       optim = list(method = f$optim$method, start = f$optim$start, control = f$optim$control, convergence = f$optim$convergence,
                    counts = f$optim$counts, message = f$optim$message, value = f$optim$value, elapsed_sec = f$optim$elapsed_sec),
       par_untruncated = f$par_untruncated, par_thresholded = f$par_thresholded,
       zero_gamma_indices_1based = I(which(f$par_thresholded[7:20] == 0)),
       loglik_sum_untruncated = f$loglik_sum_untruncated, penalty_as_coded_untruncated = f$penalty_as_coded_untruncated,
       roc_at_optimum = f$roc_at_optimum, objective_sum_scale = f$objective_sum_scale,
       objective_check_loglik_minus_lambda_penalty = f$objective_recomputed, objective_mean_scale = f$objective_mean_scale,
       loglik_sum_thresholded = f$loglik_sum_thresholded, BIC = f$BIC, AIC = f$AIC, SBIC = f$SBIC, df_count_nonzero_incl_mu = f$df_BIC,
       stored_paper_run = list(source = "result[[1]]$table row lambda (cluster run, R 4.1.1)", par_untruncated = sp,
                               par_thresholded_rounded3 = f$stored_paper_run$par_thresholded_rounded3, BIC_rounded1 = f$stored_paper_run$BIC_rounded1,
                               objective_sum_scale_with_this_runs_weights = ll(sp, Y) - f$lambda * pen(m, sp, up),
                               max_abs_diff_untruncated = f$max_abs_diff_untruncated_vs_stored,
                               thresholded_zero_pattern_matches = f$thresholded_zero_pattern_matches_stored))
}
out <- list(
  meta = list(
    title = "ESAG Project 3 reference fixture: exp-1 simulation, replicate h = 1, n = 100, six penalized methods at lambda in {1, 16, 128}",
    generated_by = "tmp/p3/scripts/make_fixtures.R + write_json.R (the user's R code is parsed and evaluated verbatim; see SPEC.md section 'Fixtures')",
    R = R.version.string, MASS = as.character(packageVersion("MASS")), RNG = paste(RNGkind(), collapse = "/"),
    number_format = "jsonlite digits=I(17): every double round-trips exactly", matrix_convention = "row-major nested arrays; Y[i][j] = R's Y[i+1, j+1]",
    parameter_vector = "par = c(mu[1..6], gamma[1..14]) (R's B for an intercept-only design). gamma groups (1-based in gamma): G1 = 1:2, G2 = 3:5, G3 = 6:9, G4 = 10:14 (B indices 7:8, 9:11, 12:15, 16:20)",
    data = "Y = result[[1]]$Y of LASSO_Comparisons/v2/Regular_LASSO_k100_v2.RData (bitwise identical in all six method files); regenerating with set.seed(1); ESAG(100, mu, gamma) in R 4.6.1 agrees to 7.2e-16",
    objective = "objective_sum_scale = sum_i log f(y_i; mu, gamma) - lambda * P(par) (the R code maximizes this with optim(fnscale = -1)). Paper scale: Q_n = objective_sum_scale / n with lambda_n = lambda / n. P per method in methods[*].penalty_as_coded.",
    thresholding = "par_thresholded = ifelse(|par_untruncated| < 0.01, 0, par_untruncated) (applied to mu too); R's BIC/AIC/SBIC use the log-likelihood at par_thresholded and df = #nonzero entries of par_thresholded (mu included)",
    start_values = "all penalized fits start at truth + 0.1 (oracle start); the unpenalized MLE (adaptive weights) also starts at truth + 0.1",
    rust_comparison_rule = "R's BFGS (numerical gradients) on this nonsmooth objective stops at suboptimal, platform-dependent points (convergence = 0 is only BFGS's reltol test). Compare objective values: a Rust solver must reach objective_sum_scale >= R's value - 1e-6*|value| (maximization) for the identical P; compare coordinates only loosely. Do NOT compare par_thresholded zero patterns exactly.",
    warning = "Several thresholded estimates sit on discontinuity sets of gamma -> V (e.g. aGLASSO lambda=16 zeroes the trailing pair of group 4: loglik drops 539.7 -> 21.0). See METHODOLOGY.md."),
  truth = list(mu = truth[1:6], gamma = truth[7:20]), n = n, d = 6L, Y = Y,
  unpenalized_mle = list(start = unp$start, optim_method = unp$method, control = unp$control, par = unp$par, loglik_sum = unp$value,
                         convergence = unp$convergence, counts = unp$counts,
                         note = "computed identically in the four adaptive scripts (optim(par = B + 0.1, joint_log_likelihood, fnscale = -1, maxit = 10000, BFGS, default reltol = sqrt(.Machine$double.eps))); not the global MLE: neutral starts reach loglik 544.72 (see SPEC.md)"),
  methods = setNames(lapply(methods, function(m) list(script = R[[m]]$script, penalty_as_coded = pen_doc[[m]], weights = weights[[m]],
                                                     fits = unname(lapply(R[[m]]$fits, function(f) fitj(m, f))))), methods))
dir.create(outdir, showWarnings = FALSE)
J(out, file.path(outdir, "p3_sim_exp1_n100.json"))

# ---------------- application ----------------
A <- readRDS(file.path(fx, "app_coded.rds")); B <- readRDS(file.path(fx, "app_paper_described.rds"))
appj <- function(a) {
  fits <- unname(lapply(a$fits, function(f) list(lambda = f$lambda,
    optim = list(method = f$optim$method, start = f$optim$start, control = f$optim$control, convergence = f$optim$convergence, counts = f$optim$counts, value = f$optim$value),
    par_untruncated = f$par_untruncated, par_thresholded = f$par_thresholded, zero_gamma_indices_1based = I(which(f$par_thresholded[7:20] == 0)),
    loglik_sum_untruncated = f$loglik_sum_untruncated, penalty_as_coded_untruncated = f$penalty_as_coded_untruncated, roc_at_optimum = f$roc_at_optimum,
    objective_sum_scale = f$objective_sum_scale, objective_check = f$objective_recomputed, loglik_sum_thresholded = f$loglik_sum_thresholded,
    BIC = f$BIC, AIC = f$AIC, SBIC = f$SBIC, df_count_nonzero_incl_mu = f$df_BIC)))
  lam <- sapply(a$fits, function(f) f$lambda); bic <- sapply(a$fits, function(f) f$BIC); k <- which.min(bic)
  list(columns = a$columns, raw_counts = a$raw_counts, Y = unname(a$data_spherical),
       unpenalized_mle = list(start = a$unpenalized$start, par = a$unpenalized$par, loglik_sum = a$unpenalized$value, convergence = a$unpenalized$convergence, counts = a$unpenalized$counts),
       fits = fits, bic_path = list(lambda = unname(lam), BIC = unname(bic)),
       bic_selected = list(rule = "which.min(BIC) over the Rmd grid (first minimum)", lambda = unname(lam[k]), lambda_n_paper_scale = unname(lam[k]) / nrow(a$data_spherical),
                           mu = a$fits[[k]]$par_thresholded[1:6], gamma = a$fits[[k]]$par_thresholded[7:20]))
}
app <- list(
  meta = list(title = "ESAG Project 3 microbiome application (Section 6), reproduced with the user's Rmd code",
              R = R.version.string, source = "Applications/real_data_application_final.Rmd chunks 1-2 (lines 13-86), Basic_Final.R in place of the unavailable Basic2.R",
              objective = "objective_sum_scale = sum_i log f(y_i) - lambda * sum_k |W_k^r B_k|, W = |1/unp$par|, W[1:6] = 0, r = max(||mu_tilde||/||mu||, 1); Q_n = objective / 164",
              method = "RoC-LASSO (MLE_RoC_weighted_LASSO in the Rmd); penalized fits start at b_start = (1,1,1,1,1,1, 0 x 14), NOT at the unpenalized MLE; threshold 0.01",
              lambda_grid = "0, 1, 2, 4, 8, 16, 32, 64 (sum scale; lambda_n = lambda/164)",
              rust_comparison_rule = "compare objective values (Rust >= R - 1e-6|R|); the BIC-selected lambda and zero pattern are sensitive to optimizer paths"),
  as_coded = c(list(data_selection = "mgp154[, c(2,5,7,8,9,10)]  -> Alistipes, Blautia, Caloramator, Clostridium, Eubacterium, Faecalibacterium (the code comment and the paper say Bacteroides; column 1 is 'id')",
                    transformation = "Y_i = x_i / ||x_i||_2 on the raw counts x_i of the 6 selected columns (no square root, no proportions); zeros kept (Alistipes, Eubacterium have zero counts)"),
               appj(A)),
  paper_reported = list(mu = c(2.50, 0.76, 0.42, 2.48, 1.65, 1.78), gamma = c(0.00, 1.07, 0.56, 0.14, 0.00, 0.35, -0.58, -0.21, -0.14, 0.46, -0.56, 1.33, -0.22, -0.56),
                        note = "matches as_coded$bic_selected (lambda = 1) up to the equivalence sign flips of groups 1, 3 (first three entries), 4 and ~0.1 optimizer-path differences"),
  rmd_comment_bic = c(-174.5697, -178.9395, -178.9395, -178.9395, -179.3893, -179.3893, -179.3893, -179.3893, -179.3893),
  variant_paper_described = c(list(note = "NOT the user's code: the transformation the paper describes (Bacteroides; sqrt of proportions renormalized over the 6 taxa), fitted with the same Rmd RoC-LASSO code",
                                   data_selection = "Bacteroides, Blautia, Caloramator, Clostridium, Eubacterium, Faecalibacterium",
                                   transformation = "Y_i = sqrt(x_i / sum(x_i))"), appj(B)))
J(app, file.path(outdir, "p3_microbiome.json"))
cat("written\n")
