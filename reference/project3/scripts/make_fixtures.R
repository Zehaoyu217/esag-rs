# Fixture driver for ESAG Project 3 (penalized ESAG).
# Runs the USER'S OWN method code (not rewritten): it parses each simulation script, evaluates the
# statements of the first `test_par(h)` body verbatim (with h = 1, k overridden to 100 and Y replaced by
# the stored replicate-1 data), and then evaluates the script's own `r_LASSO_<lambda>_BFGS = ...` calls.
# The only injected code is (i) a library() shim (optimx is loaded but never used and is not installed) and
# (ii) a logging wrapper around stats::optim that records value / convergence / counts and then returns
# optim's result unchanged.
#
# usage: Rscript make_fixtures.R sim <METHOD> <out.rds>      METHOD in LASSO aLASSO RoCLASSO GLASSO aGLASSO RoCGLASSO
#        Rscript make_fixtures.R app <variant> <out.rds>     variant in coded paper_described

args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]; which_m <- args[2]; out <- args[3]
ROOT <- "/Users/jay/.claude/jobs/a63518af/tmp/p3/Research_project3"
RDATA <- "/Users/jay/.claude/jobs/a63518af/tmp/p3/scratch/rdata"
real_library <- base::library
library <- function(package, ...) { p <- as.character(substitute(package)); if (p == "optimx") return(invisible(NULL)); real_library(p, character.only = TRUE, ...) }
source(file.path(ROOT, "Basic_Final.R")); rm(library)
real_library("parallel")

make_logger <- function(store) {
  function(par, fn, gr = NULL, ..., method = c("Nelder-Mead", "BFGS", "CG", "L-BFGS-B", "SANN", "Brent"),
           lower = -Inf, upper = Inf, control = list(), hessian = FALSE) {
    t0 <- proc.time()[[3]]
    res <- stats::optim(par = par, fn = fn, gr = gr, ..., method = method, lower = lower, upper = upper,
                        control = control, hessian = hessian)
    store$log[[length(store$log) + 1]] <- list(start = par, method = method[1], control = control,
      par = res$par, value = res$value, counts = as.list(res$counts), convergence = res$convergence,
      message = if (is.null(res$message)) NA else res$message, elapsed_sec = proc.time()[[3]] - t0)
    res
  }
}
is_assign_to <- function(s, pattern) is.call(s) && (identical(s[[1]], as.name("=")) || identical(s[[1]], as.name("<-"))) &&
  is.name(s[[2]]) && grepl(pattern, as.character(s[[2]]))

loglik_sum <- function(par, Y) { f <- 0; for (i in 1:nrow(Y)) f <- f + as.numeric(log_likelihood(Bx_i = par, Y[i, ])); f }
groups_B <- list(7:8, 9:11, 12:15, 16:20)        # gamma groups as coded (indices into B = c(mu, gamma)), d = 6

if (mode == "sim") {
  spec <- list(
    LASSO     = list(script = "LASSO_Comparisons/v2/LASSO_v2.R",                       stored = "l2_Regular_LASSO_k100.RData"),
    aLASSO    = list(script = "LASSO_Comparisons/v2/Weighted_LASSO_v2.R",              stored = "l2_Weighted_LASSO_k100.RData"),
    RoCLASSO  = list(script = "LASSO_Comparisons/v2/RoC_Weighted_LASSO_v2.R",          stored = "l2_RoC_Weighted_LASSO_k100.RData"),
    GLASSO    = list(script = "Group_LASSO_Comparisons/v3/Group_LASSO_v3.R",           stored = "g3_Group_LASSO_k100.RData"),
    aGLASSO   = list(script = "Group_LASSO_Comparisons/v3/Group_Weighted_LASSO_v3.R",  stored = "g3_Group_Weighted_LASSO_k100.RData"),
    RoCGLASSO = list(script = "Group_LASSO_Comparisons/v3/Group_RoC_Weighted_LASSO_v3.R", stored = "g3_Group_RoC_Weighted_LASSO_k100.RData"))[[which_m]]
  st <- new.env(); load(file.path(RDATA, spec$stored), envir = st); stored1 <- st$result[[1]]
  Y_stored <- stored1$Y
  exprs <- parse(file = file.path(ROOT, spec$script))
  idx <- which(sapply(exprs, function(e) is_assign_to(e, "^test_par$")))[1]
  stmts <- as.list(exprs[[idx]][[3]][[3]])[-1]
  env <- new.env(parent = globalenv()); store <- new.env(); store$log <- list()
  env$optim <- make_logger(store); env$h <- 1
  calls <- list(); in_calls <- FALSE
  for (s in stmts) {
    if (is_assign_to(s, "^r_LASSO_.*_BFGS$")) { calls[[as.character(s[[2]])]] <- s; in_calls <- TRUE; next }
    if (in_calls) next                                   # result tables / return(): not needed
    eval(s, env)
    if (is_assign_to(s, "^k$")) env$k <- 100
    if (is_assign_to(s, "^Y$")) { env$Y_regenerated <- env$Y; env$Y <- Y_stored }
  }
  pre_log <- store$log                                    # the unpenalized MLE (adaptive methods only)
  lam_tags <- c("1", "16", "128")
  fits <- mclapply(lam_tags, function(tag) {
    store$log <- list()
    nm <- paste0("r_LASSO_", tag, "_BFGS")
    res <- eval(calls[[nm]][[3]], env)
    op <- store$log[[length(store$log)]]
    lam <- as.numeric(tag); Y <- env$Y; n <- nrow(Y)
    par <- res$est_lasso_nontrun
    pen <- switch(which_m,
      LASSO = sum(abs(par[7:20])),
      aLASSO = { W <- 1 / env$unp_MLE$par; W[1:6] <- 0; sum(abs(W * par)) },
      RoCLASSO = { W <- abs(1 / env$unp_MLE$par); W[1:6] <- 0
                   roc <- max(sqrt(sum(env$unp_MLE$par[1:6]^2)) / sqrt(sum(par[1:6]^2)), 1); sum(abs(W^roc * par)) },
      GLASSO = sum(sapply(groups_B, function(g) sqrt(sum(par[g]^2)))),
      aGLASSO = { W <- 1 / sapply(groups_B, function(g) sqrt(sum(env$unp_MLE$par[g]^2))); sum(W * sapply(groups_B, function(g) sqrt(sum(par[g]^2)))) },
      RoCGLASSO = { W <- 1 / sapply(groups_B, function(g) sqrt(sum(env$unp_MLE$par[g]^2)))
                    roc <- max(sqrt(sum(env$unp_MLE$par[1:6]^2)) / sqrt(sum(par[1:6]^2)), 1)
                    sum((W^roc) * sapply(groups_B, function(g) sqrt(sum(par[g]^2)))) })
    roc <- if (grepl("RoC", which_m)) max(sqrt(sum(env$unp_MLE$par[1:6]^2)) / sqrt(sum(par[1:6]^2)), 1) else NA
    ll_nt <- loglik_sum(par, Y); ll_tr <- loglik_sum(res$est_lasso, Y)
    srow <- which(stored1$table[, "lambda"] == lam)
    stored_nt <- unname(stored1$table[srow, paste0(c(paste0("a", 1:6), paste0("b", 1:14)), "_nt")])
    stored_r3 <- unname(stored1$table[srow, c(paste0("a", 1:6), paste0("b", 1:14))])
    list(lambda = lam, call = paste(deparse(calls[[nm]]), collapse = " "), optim = op,
         par_untruncated = unname(par), par_thresholded = unname(res$est_lasso), par_thresholded_rounded3 = round(unname(res$est_lasso), 3),
         n_zero_thresholded_gamma = sum(res$est_lasso[7:20] == 0),
         loglik_sum_untruncated = ll_nt, penalty_as_coded_untruncated = pen, roc_at_optimum = roc,
         objective_sum_scale = op$value, objective_recomputed = ll_nt - lam * pen,
         objective_mean_scale = op$value / n,
         loglik_sum_thresholded = ll_tr, BIC = res$BIC, AIC = res$AIC, SBIC = res$SBIC,
         df_BIC = sum(res$est_lasso != 0),
         stored_paper_run = list(par_untruncated = stored_nt, par_thresholded_rounded3 = stored_r3,
                                 BIC_rounded1 = unname(stored1$table[srow, "BIC"])),
         max_abs_diff_untruncated_vs_stored = max(abs(unname(par) - stored_nt)),
         thresholded_zero_pattern_matches_stored = identical(unname(res$est_lasso) == 0, stored_r3 == 0))
  }, mc.cores = 3)
  names(fits) <- paste0("lambda_", lam_tags)
  unp <- if (length(pre_log)) pre_log[[1]] else NULL
  saveRDS(list(method = which_m, script = spec$script, stored_file = spec$stored,
               Y = env$Y, Y_regen_maxdiff = max(abs(env$Y_regenerated - Y_stored)), truth = env$B,
               unpenalized = unp, fits = fits), out)
}

if (mode == "app") {
  rmd <- readLines(file.path(ROOT, "Applications/real_data_application_final.Rmd"))
  starts <- grep("^```\\{R\\}", rmd); ends <- grep("^```\\s*$", rmd)
  chunk <- function(k) { s <- starts[k]; e <- ends[ends > s][1]; paste(rmd[(s + 1):(e - 1)], collapse = "\n") }
  env <- new.env(parent = globalenv()); store <- new.env(); store$log <- list()
  env$optim <- make_logger(store)
  env$source <- function(...) invisible(NULL)      # skips source('C:/.../Basic2.R'); Basic_Final.R is already loaded
  owd <- setwd(file.path(ROOT, "Applications"))
  for (s in as.list(parse(text = chunk(1)))) eval(s, env)             # load, Y = mgp154[,c(2,5,7,8,9,10)], std, data_spherical
  setwd(owd)
  if (which_m == "paper_described") {
    taxa <- c("Bacteroides", "Blautia", "Caloramator", "Clostridium", "Eubacterium", "Faecalibacterium")
    C <- as.matrix(env$mgp154[, taxa]); env$data_spherical <- sqrt(C / rowSums(C)); env$Y <- env$mgp154[, taxa]
  }
  calls <- list()
  for (s in as.list(parse(text = chunk(2)))) {
    if (is_assign_to(s, "^r_LASSO_.*_BFGS$")) { calls[[as.character(s[[2]])]] <- s; next }
    if (is.call(s) && identical(s[[1]], as.name("c"))) next          # the trailing BIC print
    eval(s, env)
  }
  unp <- store$log[[1]]
  fits <- mclapply(names(calls), function(nm) {
    store$log <- list(); res <- eval(calls[[nm]][[3]], env); op <- store$log[[length(store$log)]]
    lam <- as.numeric(sub("r_LASSO_([0-9]+)_BFGS", "\\1", nm)); Y <- env$data_spherical; par <- res$est_lasso_nontrun
    W <- abs(1 / env$unp_MLE$par); W[1:6] <- 0
    roc <- max(sqrt(sum(env$unp_MLE$par[1:6]^2)) / sqrt(sum(par[1:6]^2)), 1)
    pen <- sum(abs(W^roc * par))
    list(lambda = lam, optim = op, par_untruncated = unname(par), par_thresholded = unname(res$est_lasso),
         loglik_sum_untruncated = loglik_sum(par, Y), penalty_as_coded_untruncated = pen, roc_at_optimum = roc,
         objective_sum_scale = op$value, objective_recomputed = loglik_sum(par, Y) - lam * pen,
         loglik_sum_thresholded = loglik_sum(res$est_lasso, Y), BIC = res$BIC, AIC = res$AIC, SBIC = res$SBIC,
         df_BIC = sum(res$est_lasso != 0))
  }, mc.cores = 8)
  names(fits) <- names(calls)
  saveRDS(list(variant = which_m, data_spherical = env$data_spherical, columns = colnames(env$Y),
               raw_counts = as.matrix(env$Y), unpenalized = unp, fits = fits), out)
}
