# SPEC — what the Project 3 R code does

Scope: the user's research code in `Research_project3/` (extracted to
`tmp/p3/Research_project3/`), checked against the draft *LASSO-based inference for
ESAG distributions* (Yu & Huang, Statistica Sinica, 24 pp). Paths below are relative to
`Research_project3/` unless they begin with `tmp/p3/`. `file:L` means line L.

All numerical claims here were reproduced in R 4.6.1 (MASS 7.3.65, Mersenne-Twister /
Inversion / Rejection). The analysis scripts are in `tmp/p3/scripts/`; stored results were read
from single members of `Research_project3.zip` (in `tmp/p3/scratch/rdata/`).

---------------------------------------------------------------------------------------------

## 0. Which code produced the paper

| paper element | code that produced it | true γ in that code | verified |
|---|---|---|---|
| Sec 5.2, exp 1 (LASSO, aLASSO, RoC-LASSO), Fig 1 (n = 50) | `LASSO_Comparisons/v2/{LASSO_v2,Weighted_LASSO_v2,RoC_Weighted_LASSO_v2}.R` (n = 50, 100, 200) + `non_cv_k400/{LASSO_v2,Weighted_LASSO_v2,RoC_Weighted_LASSO_v2}.R` (n = 400); summarized by `LASSO_Comparisons/v2/LASSO_comparisons_v2.Rmd` | (5,2,3,0,5,1,0,1,5,0,3,2,0,0) = paper | yes. Recomputed CR/SE/SP/μ-bias/V-error/RMSE from `v2/*_k50_v2.RData` reproduce every Fig 1 curve, e.g. RoC-LASSO CR 0.824, 0.860, 0.899, 0.922 at λ = 0.5, 1, 2, 4; LASSO CR → 5/14 = 0.357, μ-bias → 6.3, V-bias → 31.3, RMSE → 0.457 |
| Sec 5.3, exp 2 (GLASSO, aGLASSO, RoC-GLASSO), Fig 2 (n = 100) | data: `Group_LASSO_Comparisons/{Group_LASSO,Group_Weighted_LASSO,Group_RoC_Weighted_LASSO}.R` (top level = "v1", γ = paper's exp 2) + `non_cv_k400/Group_*.R`; **metrics: scored with the truth vector of v2** (see D1) | data: (5,3,0,…,0) = paper; metrics: (5,3,0,5,1,0,…,0) | yes, see D1 |
| Sec 6 application | `Applications/real_data_application_final.Rmd` L13–233 (RoC-LASSO, d = 6) | n/a | reproduced up to equivalence sign flips (Sec 8) |
| "BIC/AIC/CV with RMSE" remark (end of 5.3) | `CV_Criteria/*.R`, `CV_Criteria/Group version/*.R`; BIC/AIC columns of every simulation table | n/a | CV result files read; not used in the paper's figures |

Version history (diffs checked line by line; **the estimation code is identical across all
versions of a method**: only the true γ, the cluster core count, the output names and, for
group v3, the λ grid change):

| folder | true γ | n | notes |
|---|---|---|---|
| `LASSO_Comparisons/v1` | (0,0,3,0,0,0,0,0,5,0,3,0,0,0) | 50,100,200 (LASSO.R only 400) | early setting |
| `LASSO_Comparisons/v2` | **(5,2,3,0,5,1,0,1,5,0,3,2,0,0)** | 50,100,200 | **paper exp 1** |
| `LASSO_Comparisons/v3` | (5,2,0,3,0,…,0) | 50,100,200 | Rmd drops 25 failed replicates (D9) |
| `LASSO_Comparisons/v4` | (5,3,0,…,0) | 50,…,400 | `v4/log.txt`: "used the setting to compare with GROUP LASSO version" (non-group methods on exp-2 data; not in the paper) |
| `Group_LASSO_Comparisons/` (v1) | **(5,3,0,…,0)** | 50,100,200 | **paper exp 2 data** |
| `Group_LASSO_Comparisons/v2` | (5,3,0,5,1,0,…,0) | 50,100,200 | its Rmd's truth vector was used to score Fig 2 (D1) |
| `Group_LASSO_Comparisons/v3` | (5,2,3,0,5,1,0,1,5,0,3,2,0,0) | 50,…,400 | group methods on exp-1 data; λ grid extended to 512, 1024; `v3/log.txt`: "for case in regular LASSO to compare" |
| `non_cv_k400/` | per version suffix | 400 | the n = 400 runs of the above |
| `CV_Criteria/`, `CV_Criteria/Group version/` | v1: (0,0,3,0,…)/(5,3,0,…); v2: exp 1/(5,3,5,0,…) | 125, 250, 500 | CV with RMSE (5.5) |
| `RoC_Weighted_LASSO_CV/` | (0,0,3,0,…) / "origin" (0,0,−3,0,0,0,0,5,−2,0,−3,0,0,0) | 200, 400 | exploratory λ-selection by an RoC criterion; not in the paper |
| `preliminary_sims/Group_LASSO.Rmd` | various | 50–200 | early versions; `research_project3_log_file.txt`: "ver1: mistaken AIC BIC; ver2: fixed AIC BIC; ver3: changed sum(f) to mean(f)" (ver3 abandoned: final code uses sum(f)) |

All simulation scripts `source("/work/zehaoy/Basic3.R")`; the archived
`RoC_Weighted_LASSO_CV/Basic3.R` is byte-identical to `Basic_Final.R`. The application and
several Rmd files source `Basic2.R`, which is **not in the archive** (D12).

---------------------------------------------------------------------------------------------

## 1. Common machinery (`Basic_Final.R`)

* L2 `library("optimx")`: loaded, never called anywhere (`grep optimx(` finds nothing). Not
  installed here; the fixtures shim `library()` for it.
* `ONB(mu)` L29–46: closed-form basis, last column μ/‖μ‖; first column
  (−μ₂, μ₁, 0, …)/‖(μ₁, μ₂)‖; column i−1 = (μ₁μᵢ, …, μ_{i−1}μᵢ, −Σ_{k<i}μ_k², 0, …)/norm;
  exact-zero fallbacks to e_j.
* `parameter(gamma)` L49–116: groups, K_i = ‖G_i‖ + 1, λ₁ = (Π K^{(d−2):1})^{−1/(d−1)},
  λ_j = K_{j−1}λ_{j−1}; θ₁ = ±acos(γ₁/‖(γ₁,γ₂)‖) (sign of γ₂) with **θ = 0 if both entries are
  exactly 0** (L68); φ = acos(G_j/‖G_{j..n}‖) with **φ = 0 if the tail is exactly 0** (L88);
  θ_i = ±acos(G_{n−1}/‖(G_{n−1},G_n)‖) with **θ_i = 0 if both are exactly 0** (L95). These
  conventions decide V on the singular sets discussed in METHODOLOGY §1.
* `rotation(gamma)` L121–156 (R = A_{d−2}⋯A_2·Lo(θ₁)); `covariance_matrix_gamma` L169–179.
* `ESAG(n, mu, gamma)` L220–226 → `ESAG_generator` L205–217 = `MASS::mvrnorm(n, mu, V)`
  normalized row-wise.
* `log_likelihood(Bx_i, y_i)` L289–311: per-observation ESAG log-density (forward M_p
  recursion; returns NaN/−Inf far in the tail, see the crate's MATH.md).
* Every method defines `joint_log_likelihood(B)` = Σ_i log f(y_i; B x̃_i) — the **sum**, with
  `X = matrix(1, n, 1)` (intercept only), and evaluates `Bx = t(matrix(B, nrow = B_dim) %*% t(x))`
  inside the loop over i (n² work, harmless at these n).

Parameter vector everywhere: `B = c(mu[1:6], gamma[1:14])` (d = 6, B_dim = 20). γ groups as
B-indices: G1 = 7:8, G2 = 9:11, G3 = 12:15, G4 = 16:20 (γ-indices 1:2, 3:5, 6:9, 10:14).

---------------------------------------------------------------------------------------------

## 2. The six objectives as coded

All six **maximize**

  F(B) = Σ_{i=1}^n log f(y_i; μ, γ) − λ · P(B)      (sum scale; λ multiplies the penalty directly)

with `optim(par = B_start, fn, control = list(fnscale = -1, maxit = 10000, reltol = 1e-7), method = "BFGS")`.
Dividing by n gives the paper's Q_n = L_n − λ_n P with L_n the mean log-likelihood and
**λ_n = λ/n**, so the code matches (3.2)/(3.3) with λ_n = λ/n exactly. μ is never penalized
(`p_alpha0 = FALSE`, or weights set to 0 on B[1:6]).

| method | script (paper version) : lines | P(B) as coded |
|---|---|---|
| LASSO (r = 0) | `LASSO_Comparisons/v2/LASSO_v2.R:14–57` (penalty L36–46) | Σ_{k=7}^{20} \|B_k\| (`sum(abs(B[-(1:d)]))`, L37) |
| aLASSO (r = 1) | `LASSO_Comparisons/v2/Weighted_LASSO_v2.R:26–54` | Σ_k \|W_k B_k\|, `W = 1/unp_mle$par` (L42, signed, no guard), `W[1:d] = rep(0,6)` (L43) ⇒ Σ_{γ} \|γ_k\|/\|γ̃_k\| |
| RoC-LASSO | `LASSO_Comparisons/v2/RoC_Weighted_LASSO_v2.R:27–60` | Σ_k \|W_k^r B_k\|, `W = abs(1/unp_mle$par)` (L46), `W[1:d] = 0` (L47), **r = max(‖μ̃‖/‖μ‖, 1)** (L40) where μ = `B[1:d]` is the **current argument** of the objective and μ̃ = `unp_mle$par[1:d]` (L48) |
| GLASSO (r = 0) | `Group_LASSO_Comparisons/Group_LASSO.R:14–51` (v3 identical) | Σ_{j=1}^4 ‖B[G_j]‖₂ (L38), no √(group size) factor |
| aGLASSO (r = 1) | `Group_LASSO_Comparisons/Group_Weighted_LASSO.R:26–57` | Σ_j ‖B[G_j]‖₂/‖B̃[G_j]‖₂ (W L44–45; penalty L38–42) |
| RoC-GLASSO | `Group_LASSO_Comparisons/Group_RoC_Weighted_LASSO.R:27–64` | Σ_j (1/‖B̃[G_j]‖₂)^r ‖B[G_j]‖₂, r = max(‖μ̃‖/‖μ‖, 1) (L45–51) |

`Basic_Final.R:636–717` (`RoC_Weighted_LASSO`, used by the CV scripts) is the same RoC-LASSO
with two differences: the weight guard `W = ifelse(unp_mle$par != 0, abs(1/unp_mle$par), 10^7)`
(L669), and the penalized start `B_start + 0.1` where callers pass `B_start = B` (the truth), so
it is again truth + 0.1 (L673).

Details common to all six:

* **RoC.** Recomputed at every function evaluation from the current μ; floored at 1 (`max(·,1)`,
  not in the paper). Asymptotically r → 1 (both μ̃ and μ̂ are consistent), so RoC-(G)LASSO ≡
  a(G)LASSO in the limit; it is a finite-sample device. Since the weight is |γ̃|^{−r},
  ∂w/∂r = −w ln|γ̃|: increasing r **lowers** the weight only where |γ̃| > 1 and **raises** it where
  |γ̃| < 1 (METHODOLOGY §2.3).
* **Epsilon guards.** Only `Basic_Final.R:669` guards exact zeros (weight 1e7). The simulation
  scripts and the application have no guard (a continuous MLE never has exact zeros).
* **Smoothing.** None: |·| and ‖·‖ enter BFGS directly; gradients are optim's central finite
  differences (`ndeps = 1e-3`, 2·20 evaluations per gradient).
* **Start values.** Every penalized fit starts at **B_start = truth + 0.1** (e.g.
  `LASSO_v2.R:59–69`, `MLE_RoC_weighted_LASSO(B_start = B+0.1, …)`), i.e. an oracle start. The
  application instead starts at (1,…,1, 0,…,0) (Rmd L34, L72–79).
* **Convergence.** All fixture fits return `convergence = 0`, which only means BFGS's relative
  function-change test passed (`reltol = 1e-7`). On the identical data the cluster run (R 4.1.1)
  and this run stop at different points, and their objective values differ by up to 28
  log-likelihood units (RoC-LASSO, λ = 128: 269.85 here vs 241.72 at the stored point;
  λ = 16: 433.16 here vs 442.80 at the stored point). Neither is optimal. Restarting BFGS from
  the better point does not move it, but a Nelder–Mead polish does: λ = 16 goes 442.80 → 443.09
  and λ = 128 goes 269.85 → 278.11 (`scripts/polish.R`). BFGS with finite differences stalls at
  the kinks of |·|.
* **Post-hoc thresholding.** `est_lasso = ifelse(abs(op$par) < 0.01, 0, op$par)` (all 20
  coordinates including μ, **elementwise even for the group methods**). Tables store
  `round(est_lasso, 3)` and the untruncated `est_lasso_nontrun` (`*_nt` columns).
* **BIC/AIC/SBIC** (e.g. `LASSO_v2.R:52–55`), computed at the **thresholded** estimate, no refit:
  `log_L = joint_log_likelihood(est_lasso)`;
  `BIC = −2 log_L + #{est_lasso ≠ 0}·log n`; `AIC = −2 log_L + 2·#{≠0}`;
  `SBIC = −2 log_L + #{≠0}·log((n+2)/24)`. df counts all nonzero coordinates **including the 6
  of μ**, and for group methods counts nonzero *entries*, not groups. The thresholding can move
  the estimate across a discontinuity of γ ↦ V: aGLASSO at λ = 16 zeroes γ₁₃ = −0.0002 and
  γ₁₄ = −0.008, so the trailing pair of group 4 vanishes and the log-likelihood drops from 539.7
  to 21.0 (BIC +41 vs −943 for the neighbouring fits). Similar drops of 50–190 units occur for
  LASSO λ = 16 (188.8), RoC-LASSO λ = 1 (73.5) and the application (Sec 8).

---------------------------------------------------------------------------------------------

## 3. Initial estimator (μ̃, γ̃)

`unp_MLE = optim(par = B + 0.1, joint_log_likelihood, control = list(fnscale = -1, maxit = 10000), method = "BFGS")`
(`Weighted_LASSO_v2.R:26`, `RoC_Weighted_LASSO_v2.R:27`, `Group_Weighted_LASSO.R:26`,
`Group_RoC_Weighted_LASSO.R:27`; default reltol = 1.49e-8): the unpenalized MLE started at the
**truth + 0.1**, no Hessian. LASSO and GLASSO do not compute it.

Start dependence (replicate h = 1, exp 1, n = 100; `scripts/start_dependence.R`):

| start | log-lik | max\|V − V_oracle\| | max\|γ − γ_oracle\| |
|---|---|---|---|
| truth + 0.1 (what the code uses) | 544.5215 | 0 | 0 |
| (1,…,1, 0,…,0) (application-style) | 543.6397 | 0.46 | 8.97 |
| random #3, #4, #5 | 544.67, **544.72**, 544.70 | 0.21–0.27 | 4.8–14.6 |
| random #1, #2 | 407.5, 511.1 | — | — |

So the γ̃ behind the paper's adaptive weights is a local maximum selected by the oracle start.
The global maximum found has a different γ representation: e.g. group 3 is
(1.08, −0.03, −4.57, −0.10) against (1.05, −0.01, −2.77, 3.48) at the oracle start, so the
truth's γ₉ = 5 sits near zero there. Elementwise adaptive weights and zero patterns therefore
depend on the start (METHODOLOGY §1.4–1.5).

---------------------------------------------------------------------------------------------

## 4. λ grids

| code | grid (sum scale; λ_n = λ/n) |
|---|---|
| LASSO_Comparisons v1–v4, Group v1–v2, non_cv_k400 | 0, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256 |
| Group v3 | 0, 0.5, 1, …, 256, 512, 1024 |
| `Basic_Final.R` `RoC_Weighted_LASSO` (CV) | 0, 1, 2, 4, …, 256 (no 0.5) |
| application Rmd | 0, 1, 2, 4, 8, 16, 32, 64 |

The paper says "λ ranges from 1 to 2⁸". The figures in fact plot λ = 0 (unpenalized; drawn at
log λ ≈ −0.9) and λ = 0.5, 1, …, 256. Every fit at every λ restarts from truth + 0.1: there are
no warm starts and no path.

---------------------------------------------------------------------------------------------

## 5. Cross-validation

`CV_Criteria/RoC_Weighted_LASSO_CV_v1.R:5–27` (and v2, and the Group versions):

1. No `set.seed`: the CV simulations are not reproducible (mclapply children reseed).
2. `custom_cross_validation(Y, X, 5)` (`Basic_Final.R:796–812`): **contiguous** folds of size
   ⌊n/5⌋ (the last fold takes the remainder), no shuffling.
3. On each training fold, `RoC_Weighted_LASSO(index, cv_index, B = truth, X, Y)`
   (`Basic_Final.R:636`) refits the unpenalized MLE from truth + 0.1 and the penalized fits at
   λ ∈ {0, 1, …, 256} from truth + 0.1.
4. `ESAG_rmse(result, Y_test)` (`Basic_Final.R:780–794`) computes, for each λ, sqrt(mean over
   test rows of (P_μ̂ y)ᵀ V̂⁻¹ (P_μ̂ y)), with P_μ̂ = I − μ̂μ̂ᵀ/‖μ̂‖², using the 3-decimal rounded,
   thresholded training estimates. d = 6 is hard-coded (`diag(1,6,6)`).
5. λ* = argmin of the fold-averaged RMSE (first minimum), then a refit on all data by
   `RoC_Weighted_LASSO_specified(L = λ*)` (`Basic_Final.R:719–777`). This is a penalized refit
   at λ*, not a relaxed or unpenalized refit.
6. `Weighted_LASSO`, `Weighted_LASSO_specified`, `Group_Weighted_LASSO[_specified]` and
   `Group_RoC_Weighted_LASSO[_specified]`, called by the aLASSO and group CV scripts, are **not
   defined in any archived file**. The stored results exist and are valid matrices, so these came
   from an unarchived version of `/work/zehaoy/Basic3.R`.
7. Outcome (stored `RoC_Weighted_LASSO_CV_k500_v2_rev2.RData`, exp 1, n = 500, 100 replicates):
   λ* = 0 in 43, 1 in 16, 2 in 5, 4 in 5, 8 in 13, 16 in 14, 32 in 2, 64 in 2. CV-RMSE often
   selects **no penalization**. RMSE ignores ‖μ̂‖ entirely, because it depends on μ̂ only through
   μ̂/‖μ̂‖ and the frame (METHODOLOGY §5).

`RoC_Weighted_LASSO_CV/*.R` (exploratory): adds `ESAG_roc` (`Basic_Final.R:837–857`), which
refits the full and the zero-restricted MLE on the test fold (both from truth + 0.1) and chooses
the largest λ whose mean RoC is < 1.005, 1.01, 1.015 or 1.02. Not in the paper.

---------------------------------------------------------------------------------------------

## 6. Data generation and metrics

Generation (every simulation script, e.g. `LASSO_v2.R:5–12`): `set.seed(h)`, h = 1,…,500;
`Y = ESAG(k, mu = (2,−5,3,−4,5,6), gamma = truth)`; `X = matrix(1, k, 1)`. `mclapply(1:500,
test_par, mc.cores = 20 or 40)` with the default `mc.preschedule = TRUE`. With
`set.seed(1); ESAG(100, …)`, R 4.6.1 reproduces the stored replicate-1 Y to 7.2e-16.

Metrics (`LASSO_Comparisons/v2/LASSO_comparisons_v2.Rmd:115–185`, `get_result`; the group Rmds
are identical apart from the truth vector):

| quantity | code | paper |
|---|---|---|
| FZR | share of the **true nonzero** γ entries estimated as 0 (L121) | — |
| TZR | share of the **true zero** entries estimated as 0 (L122) | SE = nz(γ̂⁻)/card(γ⁻) ✓ |
| SP | 1 − FZR (L180) | SP = nnz(γ̂⁺)/card(γ⁺) ✓ |
| CR | (#true-nonzero estimated nonzero + #true-zero estimated zero)/14 (L123) | ✓ |
| μ bias | ‖μ̂ − μ‖₂ per replicate (L128–130) | ✓ (an error norm, not a bias) |
| V bias | ‖V − V̂‖_F, V̂ = V(μ̂, γ̂) (L145–147) | ✓ |
| RMSE | sqrt(mean_i r_iᵀ V̂⁻¹ r_i), r_i = (I − μ̂μ̂ᵀ/‖μ̂‖²) y_i, **in-sample** on the fitting data (L132–148) | (5.5) ✓, but described as "in prediction" |

All metrics use the 3-decimal rounded, thresholded estimates. Each metric is averaged per λ as
a **98%-upper-trimmed mean** (values above the 98th percentile are dropped, L151–162; this
applies to CR/SE/SP as well). The figure bands are mean ± 2·sd(trimmed)/**√N with N = the sample
size n**, not the number of replicates (L163–176, `sd_values/sqrt(N)`, called with N = 50,
100, …) (D3). `output[-c(2,3,11),]` (L182) drops rows from the printed table only.
`best_BIC/best_AIC/best_sBIC` (L437–574) select λ per replicate by `which.min`.

---------------------------------------------------------------------------------------------

## 7. Figures

* **Fig 1 (n = 50) is consistent with its data.** Recomputed from
  `LASSO_Comparisons/v2/*_k50_v2.RData` with exp-1 truth: RoC-LASSO SE/SP/CR at λ = 0.5 are
  0.549/0.977/0.824, at λ = 4 0.866/0.952/0.922, at λ = 256 0.960/0.727/0.806; LASSO at λ = 256
  is 0.997/0.006/0.357.
* **Fig 2 (n = 100) is scored against the wrong truth (D1).** Its curves are reproduced exactly
  by the v1 data (true γ = (5,3,0,…,0)) scored with the v2 truth vector (5,3,0,5,1,0,…,0) from
  `GROUP_LASSO_comparisons_v2.Rmd:122–127`.

  | λ | GLASSO as plotted (SE/SP/CR, V-err) | GLASSO with the correct truth |
  |---|---|---|
  | 0.5 | 0.181 / 0.887 / 0.383, 3.13 | 0.191 / 1.000 / 0.306, 0.90 |
  | 32 | 0.618 / 0.731 / 0.648, 3.56 | 0.591 / 0.903 / 0.635, 1.12 |
  | 256 | 0.980 / 0.025 / 0.707, 3.65 | 0.981 / 0.038 / 0.846, 1.21 |

  The "V bias" panel (3.1–3.6) is mostly the distance between the two truths; the correct V
  error is 0.90–1.22. μ bias is unaffected (same μ). The RoC-GLASSO "CR peak 0.77 at log λ ≈ 4.1"
  becomes 0.900 at λ = 64 with the correct truth. Script: `scripts/check_fig2c.R`.
* Both figures label the proposed method "RoCaLASSO", and Fig 2's legend says
  LASSO/aLASSO/RoCaLASSO although the caption describes the group methods.

---------------------------------------------------------------------------------------------

## 8. Microbiome application (`Applications/real_data_application_final.Rmd`)

* L14 `load('mgp154.rda')`: data.frame 164 × 21. Columns: id, Alistipes, Bacteroides,
  Barnesiella, Blautia, Butyrivibrio, Caloramator, Clostridium, Eubacterium, Faecalibacterium,
  …, all_others, Age, Gender, Stratum, Diet.Group, BMI (integer read counts).
* L15 `Y = mgp154[, c(2,5,7,8,9,10)]`, commented "#Bacteroides, Blautia, Caloramator,
  Clostridium, Eubacterium, Faecalibacterium". Column 2 is **Alistipes**; Bacteroides is column 3
  (D6). The later d = 4 chunk, `mgp154[, c(5,7,8,10)]` (L238), does match its own comment, so the
  id column was present when the code was written.
* L17–20 `data_spherical = t(apply(Y, 1, function(x) x/sqrt(sum(x^2))))`: **raw counts
  normalized to unit Euclidean norm**. This is not the square root of proportions the paper
  describes (D7). Zeros are kept: Alistipes and Eubacterium have zero counts; ESAG is fine with
  that.
* L24–35: unpenalized MLE from `b_start = c(rep(1,6), rep(0,14))`, BFGS, maxit 10000, default
  reltol.
* L38–69 `MLE_RoC_weighted_LASSO`: as in Sec 2, with `W = abs(1/unp_mle$par)` (no guard) and the
  penalized start `b_start` (not the unpenalized MLE, not truth + 0.1). L72–79 grid
  λ ∈ {0, 1, 2, 4, 8, 16, 32, 64}; L81–85 prints BIC.
* The Rmd sources `Basic2.R` (missing). This reproduction uses `Basic_Final.R`.

Reproduction (`fixtures/p3_microbiome.json`, `as_coded`):

| λ | 0 | **1** | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| BIC (this run) | −174.570 | **−182.873** | −179.414 | −169.449 | −111.904 | −96.251 | −14.151 | 6.089 |
| df | 20 | 18 | 18 | 18 | 15 | 14 | 10 | 7 |
| r (RoC) | 1 | 1 | 1 | 1 | 1.020 | 1.046 | 1.133 | 1.161 |

BIC selects λ = 1 (λ_n = 1/164), giving μ̂ = (2.50, 0.76, 0.41, 2.47, 1.65, 1.78), against the
paper's (2.50, 0.76, 0.42, 2.48, 1.65, 1.78), and
γ̂ = (0.00, −0.97 | 0.63, 0.16, 0.00 | −0.35, 0.58, 0.21, −0.13 | −0.45, 0.56, −1.31, 0.22, 0.56),
against the paper's (0.00, 1.07 | 0.56, 0.14, 0.00 | 0.35, −0.58, −0.21, −0.14 |
0.46, −0.56, 1.33, −0.22, −0.56). The zeros are the same. The other entries agree up to the
exact equivalence sign flips (group 1 negated; group 3's first three entries negated; group 4
negated) and ≈0.1 optimizer-path differences. **So the paper's application numbers were computed
with Alistipes and the x/‖x‖ transformation.** The BIC values in the Rmd comment (L85:
−174.5697, −178.9395 ×3, −179.3893 ×5, i.e. 9 values for an 8-point grid) are not reproduced
beyond λ = 0 (−174.570 ✓). They come from another run or code version.

The **paper-described** variant (Bacteroides, √(x/Σx) over the 6 taxa), fitted with the same
Rmd code (`variant_paper_described` in the fixture), gives μ̂ = (5.22, 1.84, 1.41, 3.51, 2.66,
2.85), a very different shape, and BIC selects **λ = 0** (no sparsity). Its BIC path
(−1000.6, −994.9, −941.1, −984.0, −940.7, −632.0, −756.8, −806.2) is non-monotone because
thresholding costs up to 169 log-likelihood units (λ = 16).

---------------------------------------------------------------------------------------------

## 9. Fixtures (`tmp/p3/fixtures/`)

Produced by `scripts/make_fixtures.R` and `scripts/write_json.R`. The driver parses each
method script and evaluates the statements of its first `test_par(h)` body **verbatim** in a
fresh environment, then evaluates the script's own `r_LASSO_<λ>_BFGS = …` calls. It injects only
four things: h = 1; k = 100; Y replaced by the stored replicate-1 data (identical to the
regenerated Y to 7e-16); and `optim` wrapped to record `value/convergence/counts` while
returning optim's result unchanged. No method code was rewritten. Runtime: about 1 min per
method (3 λ in parallel) and about 2 min for the application.

`p3_sim_exp1_n100.json`: `Y` (100 × 6, row-major), truth, `unpenalized_mle` (start, par,
log-likelihood 544.5214953478, counts), and for each of the six methods the penalty formula, the
weights, and three fits (λ = 1, 16, 128). Each fit records: start, optim control/convergence/
counts/value; untruncated and thresholded par; zero indices; log-likelihood (untruncated,
thresholded); coded penalty; r; `objective_sum_scale` (= optim value = Σℓ − λP, checked to
≤ 1e-12) and `objective_mean_scale` (/n); BIC/AIC/SBIC; the stored cluster run's estimates and
the objective at that point. All 18 fits report convergence 0.

`p3_microbiome.json`: raw counts and Y (164 × 6) for both variants, the unpenalized MLE, eight
RoC-LASSO fits, the BIC path, the BIC-selected fit, the paper-reported values and the Rmd's
BIC comment.

**Comparison rule for Rust.** Compare objective values: for the same P, a Rust fit should reach
`objective_sum_scale ≥ R − 1e-6·|R|`. R's points are not optima (Sec 2), so compare coordinates
only loosely and zero patterns not at all.

---------------------------------------------------------------------------------------------

## 10. Discrepancies and bugs (with corrections)

| # | issue | evidence | correction |
|---|---|---|---|
| D1 | Fig 2 SE/SP/CR and V-bias computed against the wrong true γ (v2's (5,3,0,5,1,0,…)) | Sec 7, `check_fig2c.R` | recompute with γ = (5,3,0,…,0); correct values in Sec 7 |
| D2 | Fig 1/2 x-axis includes λ = 0 drawn at log λ ≈ −0.9; text says λ from 1 to 2⁸ | Sec 4, 7 | state the grid {0.5,1,…,256}; show λ = 0 as a separate reference mark |
| D3 | "95% bands" use sd/√n (n = sample size) instead of sd/√R (R = 500 replicates); too wide by √(500/n) = 3.2 (n = 50), 2.2 (n = 100) | Rmd L176 | divide by √(number of replicates kept) |
| D4 | Monte Carlo "averages" are 98%-upper-trimmed means of every metric, including CR/SE/SP | Rmd L151–162 | report plain means (or medians) and say so |
| D5 | SE/SP naming is reversed relative to the usual convention (positive = nonzero); code and paper agree with each other | Sec 6 | rename SE→"zero-detection rate" and SP→"nonzero-retention rate", or swap the names with "positive = nonzero" |
| D6 | application uses Alistipes (column 2), not Bacteroides | Sec 8 | `mgp154[, c("Bacteroides","Blautia","Caloramator","Clostridium","Eubacterium","Faecalibacterium")]`; or correct the paper's text |
| D7 | application transformation is x/‖x‖₂ of raw counts, not √proportions | Sec 8 | `sqrt(x/sum(x))` as the paper says (results change completely: variant in fixture) |
| D8 | oracle starts: every fit (unpenalized and penalized) starts at truth + 0.1; the unpenalized "MLE" is a start-selected local maximum | Sec 2–3 | data-driven start (moment start or multi-start); warm starts along the λ path |
| D9 | mclapply prescheduling: one `optim` error ("non-finite finite-difference value [14]", LASSO v3 aLASSO n = 100) wiped all 25 replicates of that worker (h ≡ 16 mod 20); v3 Rmd silently drops them | `l3_W_k100` check | `mc.preschedule = FALSE` or a per-replicate `tryCatch`; report failures |
| D10 | BIC uses the thresholded, shrunken, unrefitted estimate; df counts μ and individual group entries; thresholding can jump across a discontinuity (log-lik 539.7 → 21.0) | Sec 2 | BIC at the relaxed refit with model-dimension df (METHODOLOGY §5) |
| D11 | elementwise 0.01 threshold applied to the group methods (and to μ) | Sec 2 | exact zeros from a prox algorithm; groupwise zeros for group penalties |
| D12 | `Basic2.R`, `Weighted_LASSO*`, `Group_*_LASSO*` not archived; CV runs not seeded | Sec 1, 5 | archive the exact sources; seed every replicate |
| D13 | `RoC = max(·,1)` floor is not in the paper; (A4) as stated needs it | Sec 2 | define RoC with the floor in the paper |
| D14 | Rmd comment BIC values not reproducible with archived code | Sec 8 | rerun and update |
| D15 | hard-coded d = 6 (`W[1:d] = rep(0,6)`, group indices 7:8…16:20, `diag(1,6,6)`) | Sec 2, 5 | generic group index map |
| D16 | preliminary code: df = #(est ≠ truth) (ver1, "mistaken AIC BIC"); `p_beta0` branch overwritten (Group_LASSO.Rmd L45); `MLE_group_LASSO` uses undefined `est_lasso` (L141) | preliminary only | none needed (superseded) |
| D17 | paper typos: "r = ‖μ̃‖/μ" (Sec 3.1 end); Prop 4's covariance written with E{Ψ} (the score) instead of E{H}; Sec 6 contains the placeholder "(Consider using all six methods …)"; Sec 7 Discussion is empty | paper | fix text |
