args <- commandArgs(trailingOnly = TRUE); basic <- args[1]; dir <- args[2]
real_library <- base::library
library <- function(package, ...) { p <- as.character(substitute(package)); if (p == "optimx") return(invisible(NULL)); real_library(p, character.only = TRUE, ...) }
source(basic); rm(library)
B <- c(c(2,-5,3,-4,5,6), c(5,2,3,0,5,1,0,1,5,0,3,2,0,0))
set.seed(1); Y1 <- ESAG(100, mu = B[1:6], gamma = B[7:20])
for (f in c("l2_Regular_LASSO_k100.RData","l2_Weighted_LASSO_k100.RData","l2_RoC_Weighted_LASSO_k100.RData","g3_Group_LASSO_k100.RData","g3_Group_Weighted_LASSO_k100.RData","g3_Group_RoC_Weighted_LASSO_k100.RData")) {
  e <- new.env(); load(file.path(dir, f), envir = e); r <- e$result
  Ys <- r[[1]]$Y; tb <- r[[1]]$table
  cat(sprintf("%-40s rep1: max|Y_stored - Y_regen| = %.3g; table dims %s; lambdas %s\n", f, max(abs(Ys - Y1)), paste(dim(tb), collapse="x"), paste(tb[, "lambda"], collapse=",")))
}
e <- new.env(); load(file.path(dir, "l2_Regular_LASSO_k100.RData"), envir = e); Ya <- e$result[[1]]$Y
e2 <- new.env(); load(file.path(dir, "g3_Group_LASSO_k100.RData"), envir = e2); Yb <- e2$result[[1]]$Y
cat("Y identical between LASSO v2 and Group v3 replicate 1:", identical(Ya, Yb), "\n")
cat("R:", R.version.string, " RNGkind:", paste(RNGkind(), collapse = "/"), " MASS", as.character(packageVersion("MASS")), "\n")
