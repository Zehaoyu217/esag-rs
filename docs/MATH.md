# The mathematics implemented by `esag`

References: Paine, Preston, Tsagris & Wood (2018), *Statistics and Computing*
28:689–697 (ESAG); Yu & Huang (2024), *Electronic Journal of Statistics*
18:301–334 (parameterization for arbitrary `d`, residuals, goodness of fit);
Yu & Huang (2025), *Computational Statistics & Data Analysis* 208:108167
(regression, hypothesis tests, prediction regions); Z. Yu (2024), PhD
dissertation, University of South Carolina.

## 1. The ESAG distribution

If `X ~ N_d(μ, V)` then `Y = X/‖X‖ ∈ S^{d-1}` has the angular Gaussian
distribution. ESAG imposes `Vμ = μ` and `|V| = 1`, which makes the density

```
f(y) = (2π)^{-(d-1)/2} (yᵀV⁻¹y)^{-d/2} exp{½(α² − ‖μ‖²)} M_{d-1}(α),
α = yᵀμ / (yᵀV⁻¹y)^{1/2},
M_p(α) = ∫_0^∞ t^p φ(t − α) dt.
```

`μ/‖μ‖` is the mode, `‖μ‖` controls concentration and `V` the elliptical
shape. The free parameters are `μ ∈ ℝ^d` and `γ ∈ ℝ^{(d-2)(d+1)/2}`.

### `M_p` and numerical stability (`src/loglik.rs`)

`M_1 = αΦ(α) + φ(α)`, `M_2 = (1+α²)Φ(α) + αφ(α)`,
`M_k = α M_{k−1} + (k−1) M_{k−2}`. For `α < −2` this forward recursion
cancels catastrophically. The original R code returns `NaN` or `−Inf` far in
the tail. `esag` uses instead `M_p(α) = φ(α) I_p(b)`, `b = −α`, with
`I_k = ∫_0^∞ t^k e^{−bt−t²/2} dt`. The ratios `r_k = I_k / I_{k−1}` satisfy
the continued fraction

```
r_k = k / (b + r_{k+1}),     I_0 = 1 / (b + r_1),
```

whose terms are all positive. It is evaluated backwards from depth
`⌈170/b⌉ + 2p + 8`, which reaches full double precision for every `b ≥ 2`
(measured), and agrees with R's `integrate` to about 1e-14. Hence
`log M_p = log φ(α) + log I_0 + Σ_{k≤p} log r_k` never underflows.

`log M_p` is always computed in `f64`, together with its derivatives
`(log M_p)' = p M_{p−1}/M_p` and
`(log M_p)'' = p(p−1) M_{p−2}/M_p − ((log M_p)')²` (from `M_k' = k M_{k−1}`,
`M_0 = Φ`). Dual numbers receive it through the Taylor step
`l + d₁δ + ½d₂δ²`, `δ = α − α.re`. This is exact to the order the type
carries, and costs a few operations whatever the number of derivative
directions.

## 2. Yu & Huang's parameterization γ → V (`src/param.rs`, `src/onb.rs`, `src/rotation.rs`, `src/frame.rs`)

1. **Groups.** `γ` is split into `d−2` groups. Group `i` has `i+1` entries
   `G = (G_1, …, G_{i+1})`. Each group is a point in spherical coordinates
   `(r_i, θ_i, φ_{i,1..i−1})`:
   `r_i = ‖G‖`, `θ_i = atan2(G_{i+1}, G_i)`,
   `φ_{i,j} = acos(G_j / ‖G_{j..i+1}‖)`.
2. **Eigenvalues.** `λ_{j+1} = (1 + r_j) λ_j` and `Π λ_j = 1`, so
   `λ_1 = {Π_j (1+r_j)^{d−1−j}}^{−1/(d−1)}`. The eigenvalue of `μ` is 1.
   `γ = 0` gives `V = I`, the isotropic angular Gaussian.
3. **Eigenvectors.** `U = onb(μ)` is a closed-form orthonormal basis with
   last column `μ/‖μ‖`, with fallbacks `e_j` for degenerate `μ`. The first
   `d−1` columns are rotated by `R ∈ SO(d−1)`, a product of plane rotations
   `R = Π_{m=1}^{d−3} [R_{12}(θ_{d−m−1}) Π_{j=1}^{d−m−2} R_{j+1,j+2}(φ_{…})] · R_{12}(θ_1)`.
   `esag` applies the factors directly to the columns of `U`, so `R` is never
   formed in the likelihood loop.
4. `V = P diag(λ_1, …, λ_{d−1}, 1) Pᵀ` with `P = [U_{1..d−1} R | μ/‖μ‖]`.
   Then `yᵀV^s y = Σ_j λ_j^s (v_j·y)² + (μ̂·y)²` costs `O(d²)`.

**Fast evaluation (`loglik::quad_form_parts`).** The likelihood needs only
`q = yᵀV⁻¹y`, `y·μ` and `‖μ‖²`, and none of them requires forming `U`, `R` or `V`:

* `z = U_{1..d−1}ᵀ y` uses the closed form of the basis with running sums
  `Σ_{k<i} μ_k²` and `Σ_{k<i} μ_k y_k`, in O(d).
* `p = Rᵀ z` applies the transposed plane rotations to the (d−1)-vector `z`,
  in the order of `R`'s factors, in O(d²).
* Each angle enters only through its cosine and sine, which are ratios of
  suffix norms of its group: `cos φ_{i,j} = G_j/‖G_{j..}‖`,
  `sin φ_{i,j} = ‖G_{j+1..}‖/‖G_{j..}‖` and
  `(cos θ_i, sin θ_i) = (G_{n−1}, G_n)/‖(G_{n−1}, G_n)‖`. No trigonometric
  function is evaluated.
* `q = Σ_j p_j²/λ_j + (μ̂·y)²` with `1/λ_1 = (Π_i K_i^{d−1−i})^{1/(d−1)}`
  and `1/λ_{j+1} = (1/λ_j)/K_j`.

`adjoint::loglik_grad_adjoint` differentiates exactly this sequence in
reverse mode, at 1.3 to 2.3 times the cost of the value. The rotations are
differentiated in polar form for accuracy, and the guarded points
(`sqrt'(0) := 0`, constant branches) match the dual-number reference.

**Inverse map (`param::gamma_from_covariance`).** Given `μ` and a target `V`,
eigen-decompose `U_{−d}ᵀ V U_{−d}`. Sort the eigenvalues ascending, rescale
their product to 1, and choose eigenvector signs canonically: the
largest-magnitude entry positive, then the determinant made +1. Peel the
rotation group by group from the outside in. The last column of the remaining
rotation is a unit vector, and its spherical coordinates are that group's
angles `(φ_{i,·}, θ_i)`. Group norms are `‖G_i‖ = λ_{i+1}/λ_i − 1`. This gives
the canonical member of an equivalence class, and also the moment-based
starting value.

**Identifiability.** `v_j` and `−v_j` give the same `V`, so several `γ` map
to one distribution. Yu & Huang call them *equivalent*; they share the group
norms `r_j`. Negating the first group shifts `θ_1` by π and negates
`v_1, v_2`. This is an exact linear symmetry for every `d`, and in a
regression it means negating rows `d` and `d+1` of the coefficient matrix.
`esag` canonicalises fits so the first non-zero of those rows is positive
(`regression::canonicalize_coef`). For `d ≥ 4` further, non-linear
equivalences exist. Compare fitted distributions (`μ_i`, `V_i`) rather than
raw `γ` coefficients, and use the bootstrap for uncertainty, as the papers
recommend.

**Singular set.** Where the trailing coordinates of a group `j ≥ 2`
vanish, for example `(γ_4, γ_5) = 0` with `γ_3 ≠ 0` for `d = 4`, the angle
`θ = atan2(γ_5, γ_4)` is undefined but still rotates `V`. So `γ ↦ V` is
discontinuous on this codimension-2 set (Yu & Huang 2024, Appendix B), and
gradients blow up near it. The set also contains every point where a
*whole* group `j ≥ 2` vanishes while another group does not: `K_j = 1` ties
only `λ_j = λ_{j+1}`, but the group's undefined angles still rotate
coordinates `1..j+1`, which carry unequal eigenvalues. Approaching such a
point from different directions gives limits of `V` that differ by O(1)
however close one gets (measured: 0.27 for group 2 and 0.81 for group 3 at
`d = 5`, flat for ε = 1e−3 … 1e−9). `V` is continuous only at "prefix"
zero patterns `γ_(1) = … = γ_(m) = 0` (the ties `ξ_1 = … = ξ_{m+1}`), which
include group 1 alone and `γ = 0`; it is not differentiable at any zero
group. The frame `onb(μ)` is itself discontinuous at `μ_1 = μ_2 = 0`. `param::singular_set_tests` and the R script
`reference/checks/gamma_discontinuity.R`, which uses the original R
functions, document this. Two consequences:

* **Penalised fits (Project 3).** Both simulation truths of the LASSO paper
  lie on the set: `γ_(4) = (0, 3, 2, 0, 0)` in the first experiment (nudging
  its trailing pair by 1e−9 changes `V` by 4.6), and `γ_(2..4) = 0` in the
  second (nudging them by 1e−9 changes `V` by 0.8–1.0). A penalty that sets
  a group `j ≥ 2`, or its trailing pair, exactly to zero puts the estimate
  where `V` is defined only by the convention `atan2(0, 0) = 0`. Penalties
  whose zero set is all of `γ`, or group 1, are unaffected.
* **Regression.** With `γ_i = Γ x̃_i`, observations whose `γ_i` lie near the
  set make `ℓ(B)` jump. At `d ≥ 7` with five or six covariates the
  likelihood then has many cliffs and local maxima, and L-BFGS can stall
  against a cliff with a huge gradient (one observation pinned at
  `‖(γ_{n−1}, γ_n)‖ ≈ 1e−12`). See "Starts" below for how fitting copes.

**Traps.** Near the set, an observation's density depends on the angle of
the vanishing pair, and a move of `B` of the size of that pair turns the
angle all the way round. So wherever some `γ_i` is on the set at the angle
that suits observation `i` best, `ℓ(B)` has a local supremum that is not a
regular point: nearby, observation `i` can only lose and the others hardly
change. Every observation brings such a codimension-2 family of traps in
coefficient space, and a fit with shape covariates reaches them: its free
shape slopes can put an observation on the set at little cost to the rest,
most easily one with large covariate values (trapped observations sit at a
median 78th–84th percentile of `‖x_i‖`; `trapped_leverage`), even when the
true coefficients keep every observation far from it. L-BFGS that comes near a trap stalls against it with a huge
gradient. Traps are local suprema; regular maxima exist too, but gradient
methods do not reach them reliably.

Evidence (the hard scenario of `examples/hard_scenario.rs`, `n = 2000`,
5 covariates; `src/regression/fit_profile.rs`):

* in every γ fit at `d = 5` that did not converge, the observation with the
  largest gradient (230× the median) had its trailing pair at `1.2e−5` to
  `1e−3` of its group norm, and its log-density equalled the maximum over
  the pair's angle (−1.762 against a range of −3.349 to −1.762);
* the true coefficients keep every observation far from the set (the
  smallest trailing-pair ratio over 16,000 draws is 0.26–0.43), yet fits
  started at them end in traps (13 of 16 at `d = 5`;
  `gamma_regular_maximum`). Where one of the two fits of a dataset reached
  a regular maximum and the other a trap, the regular maximum was higher in
  4 of 5 cases (by 2.3–8.4; the trap was higher once, by 0.6);
* one shape covariate is enough: with the true shape slopes on `x1` alone,
  7 of 8 fits at `d = 5` end in traps, as many as with `x1, x3, x4`
  (`gamma_trap_reach`). At `d = 4`, whose only trailing pair is group 2's,
  21 of 24 fits converge at `n = 2000` (but 1 of 12 at `n = 200`, and
  19–110× slower than log-tangent (n = 2000 to 200));
* the race never converged (0 of 12 at `d = 5` and `d = 8`, n = 500 and
  2000); the perturbed restarts never rescued a trapped fit (0 of 51, hard
  and standard scenarios), and only found other traps, up to 67
  log-likelihood units higher: there are many, and more search finds
  higher ones;
* at small n the effect is the same (γ at `d = 5, 8`: 0–1 of 12 fits converge
  at every n from 100 to 1000).

Fits detect this: `EsagRegressorFit::singular_obs` lists the observations
within `1e−2` of a singular set (relative to the group norm, or
`√(μ̂_0² + μ̂_1²)` for the frame's pole) whose gradient exceeds 20 times the
median; the restart waves are skipped when they are not empty (they cannot
help), and the summary says that the fit stopped in a trap, so the estimate
is not an MLE in the usual sense. On the standard scenario (`examples/big_fit.rs`, slopes ~0.3) the
γ fits at `n = 2000` converge (11 of 12), and the waves still rescue some
fits that stall without a pinned observation (4 of 9).

**Can the traps be avoided within γ?** On the estimation side, partly.
A trap is not a root of the score equations: its gradient is huge, not
zero. So the estimator to look for is the regular root, Cramér's
consistent root, and some fits do reach one. Four ways of reaching it were
compared on the hard scenario (`gamma_regular_root`,
`gamma_exclusion_rounds`):

* exact Newton crawls, even from the true coefficients (0 of 8 reach a
  root in 40 steps), because near-singular observations make the Hessian
  badly conditioned;
* a ridge continuation on the shape slopes followed by Newton also stops
  short (0 of 8);
* excluding the trapped observations, refitting the rest warm and adding
  them back (up to 8 rounds) reaches a regular root in 3 of 12 trapped fits
  at `d = 5, n = 2000`, 5 of 12 at `n = 500` and 0 of 12 at `d = 8`, at
  2.5–11 times the time. Every root it reached had a higher likelihood
  than the trap (+1.3 to +11.7). With at most 3 rounds the counts are the
  same at `d = 5`, 2 of 12 at `d = 6`, and 0 of 12 at `d = 7` (at 10 times
  the time). So `EsagRegressor::fit` runs these rounds (at most 3, stopping
  when a round does not raise the likelihood) for trapped γ fits at
  `d ≤ 6`. On the hard scenario (`hard_scenario -- fit`, one core) γ then
  converges in 3 of 12 fits at `d = 5` and `d = 6` with `n = 2000`, in a
  median 1.0 and 1.3 s (0.69 s before at `d = 5`). Regression test:
  `trapped_gamma_fit_is_rescued`.

At a regular root, estimation noise puts some high-leverage observation's
fitted trailing pair near zero, so a trap lies next to the root and the
root's basin is narrow. This should fade as `n` grows, because the true
pairs stay away from zero (smallest ratio 0.26).

On the parameterization side, no, as long as a group is an eigenvalue gap
plus the direction of an eigenvector. At `d = 4`, group 2's angles place
the top eigenvector at `(sin θ sin φ, −cos θ sin φ, cos φ)` on `S²`, and
`Lo(θ₂) P₂(φ) Lo(θ₁)` completes the other two around it. With the lower
eigenvalues distinct (group 1 nonzero), a `V` continuous in group 2's
direction would need the other two eigenvector lines to be continuous
tangent line fields on `S²`. None exists (Euler characteristic 2, the
hairy-ball theorem). In general group `j`'s completion would need `j`
orthogonal tangent line fields on `S^j`. That is impossible for even `j`,
and for odd `j ≥ 3` it needs a parallelizable sphere, so `j ∈ {3, 7}`
(Bott–Milnor, Kervaire). A redesign of γ in this form can move the jumps
(for example to one point per sphere) but cannot remove them for `d ≥ 4`.
Only a parameterization that does not split `V` into gaps and directions
avoids them, which is what the log-tangent shape does.

**A refined γ (`src/gamma_smooth.rs`, an experiment).** It keeps γ's
coordinates, its eigenvalue ratios `K_j = 1 + ‖G_j‖` and the eigenvector
each group places, which is continuous: `v_j = (1/‖G_j‖) Σ_k (−1)^k
G_{j,k} e_{j−k}`, the image of `e_j` under the group's plane rotations. It
replaces only the completion of the lower eigenvectors, by the minimal
rotation from a reference `c_j`, singular at `v_j = −c_j` alone, and the
frame `onb(μ)` by the transported frame. Its fits converge where γ's are
trapped (8 of 8 against 0 of 8, hard scenario, `d = 5`). But:

* they take 150–1600 iterations at `d = 8` and on the hard scenario,
  because the eigenvector directions there sweep nearly the whole sphere
  and reach `−c_j` and the zero groups;
* it is a different regression model, and on γ's own data it recovers
  `V(x)` no better than γ (`V̂` error 0.35 against 0.25 on the standard
  scenario at `d = 5`, 0.32 against 0.32 on the hard one).

So it is not wired into the regression; the module's documentation has
the numbers.

**The pole of `onb(μ)`.** The frame is singular where `μ_0 = μ_1 = 0`, and
the γ model, unlike the log-tangent model with its reference at the data's
mean direction, cannot move that set away from the data. With the mean
direction near `e_d` (on the set), a γ fit at `d = 3` took 105 ms and 200
iterations, and exact Newton from the full fit crawled on bootstrap data;
with a generic mean direction it takes 28 ms (33 iterations). The γ
regression model is not equivariant under rotations of the response.

`Shape::GammaTransported` (`src/gamma_transported.rs`) writes γ's map in
the frame transported from a reference direction `m`, the data's mean
direction by default, instead of `onb(μ)`. Its only singular point is
`μ̂ = −m`. At `μ̂ = m` the two frames coincide (`E` is `onb(m)`'s tangent
part), so without covariates the fit is γ's (same log-likelihood and
covariance, `transported_gamma_without_covariates_is_gamma`). With
covariates the model differs through how the frame turns as `μ̂(x)` moves.
The kernel is γ's adjoint with the projections replaced by the log-tangent
frame's (46 / 93 / 185 against 40 / 89 / 165 ns per observation at
`d = 3 / 5 / 8`). Production fits on γ data from the hard scenario
(`gamma_transported_fits`, `n = 2000`, 12 fits):

| mean direction | d | γ (onb) | γ-transported |
|---|---|---|---|
| near the pole (√(μ̂₀² + μ̂₁²) = 0.12–0.19) | 3 | 12/12, 92 ms | 12/12, 17 ms |
| | 4 | 5/12, 577 ms | 10/12, 121 ms |
| | 5 | 1/12, 985 ms | 7/12, 320 ms |
| generic (0.60–0.78) | 3 | 12/12, 23 ms | 12/12, 21 ms |
| | 4 | 11/12, 98 ms | 10/12, 68 ms |
| | 5 | 5/12, 574 ms | 1/12, 653 ms |

(converged fits, median time). The `V̂(x)` errors are within 0.04 of each
other; the onb model, which generated the data, is slightly better near
the pole. So onb stays the default. γ fits whose mean direction at `x = 0`
has `√(μ̂₀² + μ̂₁²) < 0.3` (about 17° from the pole set) get a summary note
recommending the transported frame, and Python's `shape="auto"` picks it
there.

The log-tangent parameterization (§7) describes the same distributions
without these sets, and its likelihood is smooth: on the same kind of data
251 of 252 of its fits converge from `n = 200` on (§3).

## 3. Regression (`src/regression.rs`)

`Y_i | x_i ~ ESAG(μ_i, γ_i)`, `(μ_i, γ_i) = B x̃_i` with `x̃_i = (1, x_i)`.
The identity link is possible because the parameter space is all of
`ℝ^{d + (d−2)(d+1)/2}`. `B` has rows `(μ_1..μ_d, γ_1..γ_g)` and one column
per design column.

**Fitting.** `esag` maximises `ℓ(B) = Σ_i log f(y_i; B x̃_i)` with L-BFGS
(`src/optim/lbfgs.rs`). The line search is strong Wolfe with Hager–Zhang
approximate-Wolfe acceptance near the optimum. The gradient is **exact**:
`g_i = ∂ log f/∂η` comes from the hand-written reverse-mode kernel
(`src/adjoint.rs`; the forward-mode dual-number version is kept as the
test reference). Then `∂ℓ/∂B = Σ_i g_i x̃_iᵀ`, computed in
parallel over observations. The R code uses `optim(method = "BFGS")` with
central finite differences of step 1e-3, which costs `2·dim(B)` likelihood
evaluations per gradient.

**Evaluation.** Per chunk of 64 observations, `η = B X_cᵀ` and the
gradient `G X_c` are matrix products; the per-observation kernel is the
reverse-mode gradient above. Chunk results are summed in chunk order, with
or without `parallel`, so the likelihood, and therefore every fit, is
bit-for-bit the same whatever the number of threads, and the same for
`with_parallel(false)` (which used to sum all observations as one chunk:
on multimodal γ problems such rounding differences send L-BFGS elsewhere,
and changed the log-likelihood of 29 of 40 hard-scenario fits at d = 4,
n = 300, between the two settings). nalgebra
multiplies through `matrixmultiply` only when every dimension exceeds 5,
so at d = 3 (`η` of length 5) both products ran column by column; the
same operations, in the same order, in fixed-size loops (`small_eta`,
`small_grad`, bit-identical) bring the gradient at d = 3 from 0.15 to
0.10 ms (n = 2000, 5 covariates, one core).

**Starts.** By default (`Init::Auto`) `esag` uses two starts:

* a moment start: `μ` along the mean resultant, with
  `‖μ‖ = (Π σ_j)^{−1/(2(d−1))}` from the eigenvalues of the tangent scatter
  `S ≈ V/‖μ‖²`, and `γ` from the shape of `S` via the inverse map;
* the R code's start: `μ` coefficients 1, `γ` ≈ 0.

*Log-tangent.* The likelihood is smooth and both starts reach the same
maximum: 192 of 192 simulated datasets (hard and standard scenarios,
`d = 3, 5, 8`, `n = 300` and 2000; the differences were below 5e−4, the
hand-over tolerance; `fit_profile::single_start_safety`). So the moment
start runs alone, whitened with the exact information there
(`L Lᵀ = I(B₀)⁻¹`, L-BFGS on `z` with `B = B₀ + L z`): 9–10 evaluations
instead of 23–34, for one Hessian (2–4 gradients). The R code's start runs
only if the first fails the convergence check. One-core fits with standard
errors, `n = 2000`, 5 covariates: 6.7 → 2.8 ms (`d = 3`), 34 → 11.9 ms
(`d = 5`), 129 → 50 ms (`d = 8`).

*Gamma.* The starts end at different maxima in 2–10 of 16 datasets (by up to
9 log-likelihood units at `d = 3`), so they are raced. Measured against the
race (`regression::gamma_profile`, 40 datasets per cell, n = 300, 1000,
2000, d = 3–5), no cheaper strategy reaches the same maxima:

* the moment start alone, whitened with the exact information as for
  log-tangent fits, matches the race in 80 of 80 standard datasets at
  d = 3 and 77 of 80 at d = 4 (n = 1000, 2000), 2–3× faster, but ends
  lower in 12 of 40 standard datasets at d = 3, n = 300 (never higher), in
  6–22 of 40 hard ones (d = 3, 4), and in 5–11 of 40 standard ones at
  d = 5 (one by 23). At small n the R-style start
  finds the maximum with large shape slopes that the moment start, plain
  or whitened, misses. Whitening with the empirical information (outer
  product of the scores) instead does worse at d = 5 (lower in 14 of 48,
  by up to 45);
* gating that single start on how close the fitted observations come to
  the non-smooth sets of γ (the cones `G_m = 0`, the trailing pairs, the
  frame pole) does not help: losses remain where every observation stays
  farther than 30% of the median group norm from them;
* stopping the race at a mean gradient of 1e−2 per observation and
  finishing with exact Newton (L-BFGS follows exactly the race's paths up to
  there) ends lower in 4–8 of 40 hard datasets (d = 3) and 6 of 40
  standard ones at n = 300: at that point the log-likelihood is still
  0.4–26 below the end, the Hessian is mostly indefinite, and even a fresh
  L-BFGS from that point ends elsewhere than the race.

These likelihoods are so multimodal that the maximum reached depends on the
whole optimizer history.

The race runs in lockstep: every start advances 25 L-BFGS iterations per
round (in parallel), and all decisions are taken between rounds, so they do
not depend on thread timing.

* When a start converges after `N` iterations, the others stop at
  `max(2N, N + 50)` iterations unless they already beat its likelihood.
* After 100 iterations, a start trailing the leader by more than 0.05
  log-likelihood units per observation is stopped (a much worse basin).
* A start that gains less than 1e−3 per observation over 100 iterations
  while its gradient is still more than 10× the convergence threshold is
  stopped (crawling against a cliff).
* A start whose largest per-observation gradient stays above 1e5 times the
  median for 25 evaluations is stopped (an observation pinned at the
  singular set; healthy fits stay below ~10²).

The winner is the start with the highest likelihood; a converged start wins
only ties. A start counts as converged when the optimiser converged *and*
the mean per-observation gradient is ≤ 1e−3. If the winner has not
converged, up to two waves of four randomly perturbed starts are raced (600
iterations each; slope perturbations are divided by each covariate's root
mean square), unless observations sit on a singular set of γ (§2) or the
likelihood is unbounded (below): more starts cannot converge there. A fit
that still has not converged is returned with `converged = false`, and
`singular_obs` / `degenerate_obs` say why. The L-BFGS memory defaults to
the number of free parameters, clamped to [10, 200].

**When the MLE does not exist (small samples).** With almost as many
coefficients as observations, the likelihood of the full model is
unbounded: the fit drives `V(x_i)` towards a singular matrix at
high-leverage observations, whose density then grows without bound. On the
hard scenario at `n = 100` (log-tangent; 84 coefficients at `d = 5`, 210 at
`d = 8`) the fits that failed (2 of 12 at `d = 5`, 11 of 12 at `d = 8`) all
did this: the largest log-density exceeded the median one by 17–75 nats,
`‖S(x_i)‖` reached 8–21 (eigenvalue ratios of `e^{11}` and more), and the
observed information had eigenvalues up to 10¹¹–10²² of both signs.
Converged fits stay below 9 nats (n = 100 to 2000, `d = 5, 8`). From
`n = 200` on, 251 of 252 log-tangent fits converged (`d = 3, 5, 8`; the
exception stalled at `d = 8`, `n = 200` without degenerating). Exact
Newton from the stopping point of a degenerate fit does not converge
either: no optimizer can, since there is no maximum.

`EsagRegressorFit::degenerate_obs` lists the observations whose
log-density exceeds the median one by more than 12 nats in a fit that did
not converge; the restarts are then skipped and the summary points to a
penalized fit (§8), which is well defined. The penalized paths handle this
case (§8, "Small samples").

**Standard errors.** `esag` reports the inverse observed information. It is
assembled as `Σ_i (x̃_i x̃_iᵀ) ⊗ ∇²_η ℓ_i`. The per-observation Hessian of γ
is exact (`adjoint_hess`): the adjoint's forward pass and reverse sweep,
one forward sweep of tangents in the angles of the rotation chain, where
`z` is a chain of rotations by one angle each, then the polar steps of
the reverse sweep back to γ. It costs 0.09, 0.42 and 2.1 µs at d = 3, 5
and 8 (1.7–14 gradients), 2.7× less than second-order dual numbers at
d = 3 and 5–7× less than the central differences of the gradient that
were used for d ≥ 4, and agrees with the dual numbers to rounding. The
Kronecker sum is one matrix product per chunk of observations over the
unique entries (`x_{j1}x_{j2}`, `j1 ≤ j2`, times `H_i[k1,k2]`, `k1 ≤ k2`).
On one core (n = 2000, 5 covariates) the joint Hessian takes 0.31, 1.3 and
6.1 ms at d = 3, 5 and 8, 3, 7 and 16 gradients (0.55, 5.4 and 26 ms
before). When standard errors are requested, L-BFGS hands
over once the mean gradient per observation is below 1e−4, and up to eight
simplified Newton steps reuse this one Hessian to polish the optimum to
‖∇ℓ‖ ≈ 1e-10 (if the Hessian is not negative definite there, L-BFGS
finishes at full tolerance first). Fits that did not converge stop next to
kinks of γ; there `finalize` keeps the central differences of the
gradient for d ≥ 4, whose average over a small step gives the polishing
Newton steps a better model than the pointwise Hessian (with the exact
Hessian, 39 of 640 fits at d = 4 and 5, all unconverged, ended lower, by
up to 1.2, and 8 higher). Fits
are then the same as before the exact Hessian (1071 of 1080 identical
over d = 3–5, n = 300–2000, the rest within 2e−7), and standard errors of
converged fits change by up to 2e−3 relative at d ≥ 4, the error of the
differences. Intercept-only fits
report the canonical γ of their fitted V. Yu & Huang show this can be
unreliable near `γ = 0`, where the likelihood is non-smooth, and for `γ`
generally because of equivalence. `inference::bootstrap_coef` gives pairs
bootstrap SEs and percentile intervals.

**Covariates.** The papers rescale each covariate to `[1, 2]`;
`preprocessing::MinMaxScaler` does this. The model is linear in `x`, so
rescaling changes `B` but not the fitted distributions.

## 4. Residuals and goodness of fit (`src/inference/gof.rs`)

`r_i = (I − μ̂_iμ̂_iᵀ) y_i`, `Q_i = r_iᵀ V_i⁻¹ r_i`,
`T1_i = (‖μ_i‖² + Σ_j λ_{ij} + 1) Q_i = E‖X_i‖² · Q_i`, which is approximately
`χ²_{d−1}`.

* `GofMethod::TwoSampleReference` (default) follows Yu & Huang 2024,
  Algorithm 1. `KS_p` is the p-value of a two-sample KS test between the
  observed `T1` and `T1` of a reference sample drawn from the fit. Its null
  distribution comes from a parametric bootstrap that refits on every
  simulated data set, and `p = #{KS_p^(b) < KS_p}/B`.
* `GofMethod::OneSampleChiSquare` follows the author's regression R code. The
  statistic is the one-sample KS p-value of `T1` against `χ²_{d−1}`, with the
  same bootstrap, and `p = #{stat ≥ stat^(b)}/B`.

## 5. Prediction regions (`src/inference/regions.rs`)

`PR_a = {y : (y − μ̂)ᵀ V̂⁻¹ (y − μ̂) ≤ q_a}` where `μ̂ = μ/‖μ‖`. `q_a` is the
`(1−a)` quantile of the statistic over `m` draws from the plug-in model,
pooled with `m` draws from each of `B` nonparametric-bootstrap refits
(CSDA Algorithm 4). Among ellipsoidal regions with `Vμ = μ` and `|V| = 1`,
this has the smallest volume.

## 5b. Estimating V(x) (`src/inference/covariance.rs`)

The object of interest is often the covariance `V(x)` itself, not the
coefficients: `V` depends on every coefficient (through `μ(x)` and the
shape coordinates) in a nonlinear way, and per-coefficient standard errors
from the observed information say little about it. `V̂(x) = V(B̂ x̃)` is the
plug-in MLE (`predict_covariance`), and `covariance_intervals` adds
pairs-bootstrap uncertainty at chosen covariate vectors:

* the shape eigenvalues (those of `V̂` on `μ̂⊥`; the one along `μ̂` is 1)
  and principal axes;
* *basic* intervals `[2θ̂ − q_{1−α/2}, 2θ̂ − q_{α/2}]` for each eigenvalue and
  entry, with bootstrap standard errors;
* a geodesic ball `{V : ‖log(V̂^{−1/2} V V̂^{−1/2})‖_F ≤ r}`, `r` the
  level-quantile of `δ(V̂*, V̂)`.

**Validation** (`examples/v_recovery.rs`; data from a known regression with
5 covariates + const; V̂(x) against the true V(x) on 200 covariate points;
geodesic distance, mean over 20 replications):

| model | d | n = 500 | n = 2000 | n = 10000 | √n × error | converged at n = 500 |
|---|---|---|---|---|---|---|
| γ | 3 | 0.30 | 0.11 | 0.048 | 6.8 → 5.0 → 4.8 | 20/20 |
| γ | 5 | 0.75 | 0.27 | 0.11 | 16.8 → 11.9 → 10.7 | 16/20 |
| γ | 8 | 1.40 | 0.46 | 0.19 | 31.2 → 20.7 → 18.7 | 3/20 |
| log-tangent | 3 | 0.24 | 0.12 | 0.055 | 5.3 → 5.3 → 5.5 | 20/20 |
| log-tangent | 5 | 0.53 | 0.24 | 0.11 | 11.8 → 10.9 → 10.9 | 20/20 |
| log-tangent | 8 | 0.91 | 0.42 | 0.18 | 20.4 → 18.9 → 18.0 | 20/20 |

Both are consistent at the `1/√n` rate. At n = 2000, the shape eigenvalues
are within 4–12% (log scale) and the mean direction within 1.3–1.6°.

Coverage of the 95% intervals at x₀ = 0 and x₀ = e₁ (n = 2000, B = 200, 200
replications; Monte-Carlo SE 0.015):

| model | d | eigenvalues, percentile | eigenvalues, basic | geodesic ball |
|---|---|---|---|---|
| γ | 3 | 0.90–0.94 | 0.93–0.96 | 0.95–0.96 |
| log-tangent | 3 | 0.93–0.96 | 0.95 | 0.935–0.945 |
| γ | 5 | 0.56–0.93 | 0.95–0.98 | 0.975 |
| log-tangent | 5 | 0.93–0.99 | 0.93–0.98 | 0.955–0.97 |

The γ MLE of the smallest eigenvalue at d = 5 is biased by about one
standard error (−0.0103 against a sampling SD of 0.0110; the bootstrap SD,
0.0112, is right). Percentile intervals double that bias, so their
coverage falls to 0.56; basic intervals correct it to first order.

## 6. Hypothesis tests (`src/inference/tests.rs`)

Restricted fit (`μ_{0i}`, `γ_{0i}`) versus full fit (`μ_{ai}`):

```
RoC = mean_i ‖μ_{ai}‖ / ‖μ_{0i}‖
D   = mean_i {2 − cos∠(μ_{0i}, μ_{ai})} ‖μ_{ai}‖ / ‖μ_{0i}‖
M   = ‖ mean_i ( y_i² − E_0 y_i² ) ‖²     (elementwise squares; E_0 exact, see below)
LR  = −2 (ℓ_0 − ℓ_a)
LM  = gᵀ I⁻¹ g − g_fᵀ I_ff⁻¹ g_f          (score statistic at the restricted fit, below)
```

`M` is the squared norm, as in the R code and the papers' tables (CSDA eq. (4)
prints the norm; p-values are unaffected). Null distributions come from a
parametric bootstrap under the restricted fit,
refitting both models each time, and `p = #{stat^(b) > stat}/B`. Supported
nulls are `μ` free of feature `j`, `γ` free of feature `j`, both, the shape
free of every feature (`ConstantShape`: an anisotropic shape, the same at
every `x`), isotropy (`γ ≡ 0`), or any custom mask. Two differences from the R code: bootstrap
responses are drawn from each observation's own null parameters (the R code
indexes them by the bootstrap counter), and every `d` is supported (the R
isotropy and γ-restricted fits hard-code `d = 4`).

**Exact second moments** (`distribution::second_moment_weights`). The R code
estimates `E_0 y_i²` with Monte-Carlo draws (10⁴ per observation in the
papers), which adds noise of order `n_mc^{−1/2}` to `M` and dominated the
cost of a test. They are available in closed form up to one integral. With
`1/‖x‖² = ∫₀^∞ e^{−t‖x‖²} dt` and `X ~ N(μ, V)`, the Gaussian integral in the
eigenbasis of `V` (where `μ = ‖μ‖μ̂` has one coordinate, because `μ` is an
eigenvector) gives `E[YYᵀ] = Σ_j w_j v_j v_jᵀ + w_μ μ̂μ̂ᵀ` with

```
c(t) = (1+2t)^{−1/2} Π_j (1+2tλ_j)^{−1/2} exp(−t‖μ‖²/(1+2t)),
w_j  = ∫₀^∞ c(t) λ_j/(1+2tλ_j) dt,
w_μ  = ∫₀^∞ c(t) [1/(1+2t) + ‖μ‖²/(1+2t)²] dt,      Σ w + w_μ = 1.
```

After `t = e^s` the integrands are analytic and decay exponentially; the
factor `exp(−t‖μ‖²/(1+2t))` limits the useful strip to `|Im s| < π/2`, so
the trapezoid rule converges like `exp(−π²/h)` uniformly in the
concentration: step 0.35 (0.3 for `‖μ‖ > 20`) gives ≲ 1e−12.

Most of a plain trapezoid sum is spent in the tails: at `t → 0` the
integrands are constant in `t`, so in `s` they decay only like `e^s` (about
60 nodes from `s ≈ −32`), and at `t → ∞` like `e^{−sd/2}` (about 70 more at
d = 3). Both tails of the *infinite* trapezoid sum are evaluated in closed
form instead, which keeps its accuracy:

* left, nodes `t_L e^{−kh}` with `t_L K = 10⁻⁴` (`K = max(1 + ‖μ‖² + Σλ,
  2λ_max, 2)`): with `F(t) = F₀ + F₁t + F₂t² + O(t³)`,
  `h Σ_{k≥1} F(t_k) t_k = h Σ_m F_m t_L^{m+1} q^{m+1}/(1 − q^{m+1})`, `q = e^{−h}`,
  with `F₀, F₁, F₂` from `c(t) = 1 − A₁t + (A₁²/2 + A₂)t² + O(t³)`,
  `A₁ = 1 + ‖μ‖² + Σλ`, `A₂ = 1 + 2‖μ‖² + Σλ²`. The omitted terms are
  below `(t_L K)⁴/4 ≈ 2.5e−17` relative.
* right, in `u = 1/(2t)`: `t·c(t)·g(t) = ½ (2t)^{−d/2} (Πλ)^{−1/2}
  e^{−‖μ‖²/2} G(u)` with `G` analytic at 0 (radius `min(1, λ_min)`): `ln G`
  has the explicit series `Σ_m (−1)^{m+1} u^m [‖μ‖²/2 − 1/(2m) −
  Σ_j λ_j^{−m}/(2m)]` plus the series of the weight factor, so the nodes
  beyond `u ≤ 0.02 min(1, λ_min, 2/‖μ‖²)` sum to
  `Σ_m G_m u^m r_m/(1 − r_m)`, `r_m = e^{−h(d/2 + m)}`, to order 8 (omitted
  terms below `0.02⁹`). For `‖μ‖² ≥ 80` the right tail is below `e^{−40}`
  and is cut instead.

About 50–100 nodes remain. The nodes are stepped by `t ← t e^h`, so a node
costs one `exp`. A call takes 0.3–0.6 µs (1000 Monte-Carlo draws cost
~25 µs), and the result agrees with the plain trapezoid rule to 3e−12 over
d = 3–10, `‖μ‖` up to 80 and eigenvalue ratios up to e⁹.
`n_mc = 0` selects the exact moments (the default in the examples); the
published `M` values are reproduced (2.17e−4 vs 2.2e−4, 9.47e−5 vs 9.4e−5).

**Refits.** Bootstrap refits (here, and in the goodness-of-fit and region
procedures) start near the original optimum, where the observed information
is nearly the same, so L-BFGS runs in whitened coordinates `θ = θ₀ + Lz`,
`LLᵀ = I(θ̂)⁻¹`, where the Hessian is close to the identity. Three details
follow from that scaling:

* the first trial step is 1 (a Newton step), not `min(1, 1/‖g‖₁)`;
* a refit stops once the predicted remaining gain `½‖g_z‖²` is below 1e−6,
  so every coefficient is within about 0.0014 standard errors of its
  maximum and the log-likelihood within 1e−6. That is far below the
  bootstrap's own noise. Against the 1e−6 gradient tolerance of a fit, it
  moves no replicate across an observed statistic, and it changes RoC and D
  replicates by ≤ 2% and M by ≤ 0.4% of their bootstrap SDs, and LR by
  ≤ 1e−6;
* the θ-gradient at the end is `L⁻ᵀ g_z`, a triangular solve, not another
  evaluation.

A refit then needs about 6 likelihood evaluations at d = 3 and 10 at d = 5
(n = 2000, 5 covariates), against 11 and 17–19 before. The whitened Hessian
at a replicate's own maximum has eigenvalues in about `(1 ± √(k/n))²` (k
parameters). That is sampling variation of the information, not a defect of
the preconditioner, and it is why L-BFGS still takes several iterations.

**Refits where γ has cliffs (d ≥ 5 with covariates).** There the original
fit's information is a poor metric. An observation whose trailing pair
`(γ_{n−1}, γ_n)` in a group `j ≥ 2` has small norm `r` (§2) adds curvature
of order `1/r²`. For the closest observations that is as much as the rest
of the sample, and it depends on their responses, so it changes from
replicate to replicate. Worse, the full model's information at the
restricted fit, the metric for full refits in tests, is usually not
positive definite there, and those refits ran unpreconditioned. So refits
in this regime, and any refit without a positive-definite metric, are
whitened with the empirical information `Σ_i s_iᵀ s_i` at their start
(`s_i` the score of observation `i` over the free coefficients). It costs
about one evaluation, it grows like `‖s_i‖² ~ 1/r²` exactly where the
curvature does, and it is recomputed every 50 iterations because it goes
stale as a refit travels. Exact Newton steps are no remedy: a refit needs
about 14 Hessians (one is 32 evaluations at d = 5), against 4–5 on a smooth
likelihood. On the `x5` test at d = 5 (n = 2000, 5 covariates), the
evaluations per refit change as follows:

| refit | before | after |
|---|---|---|
| restricted | 106 | 74 |
| full, from the restricted refit | 244 | 84 |
| full, from the original full fit | 310 | 194 |

**Local maxima.** These likelihoods have many local maxima. Near each
near-singular observation `ℓ` winds around the singular set, and on top of
the smooth part this leaves several ridges. Refits from the same start that
differ only in metric reach different maxima. Each is a true maximum
(exact Newton decrement ≤ 1e−6), and the gaps go both ways:

| test | replicates that differ | largest gap | mean (± s.e.) |
|---|---|---|---|
| `x5` | 45% | 4.5 | −0.13 ± 0.09 |
| shape free of `x1` | 25% | 1.6 | 0.00 ± 0.03 |

That is why the full model is refitted from two starts, and the bootstrap
`LR` at d ≥ 5 still carries this search noise. Under the isotropy null
every `γ_i` is near 0, so every observation is near the set. The full
refits then never converge, and their log-likelihoods differ between
metrics by up to 19 (mean +3.4 ± 2.1 for the new one). The γ isotropy
`LR` at d ≥ 5 is therefore unreliable. `M` uses only the restricted fit
and is unaffected. The log-tangent model has the same null hypothesis
(`V ≡ I`) and a smooth likelihood around it.

**The isotropic model is concave.** With `V = I` the log-density is
`ℓ_i(μ) = log ∫_0^∞ r^{d−1} exp(−r²/2 + r yᵀμ − ‖μ‖²/2) dr + const`. The
integrand is jointly log-concave in `(r, μ)`: its Hessian
`[[−(d−1)/r² − 1, yᵀ], [y, −I]]` has Schur complement `−(d−1)/r² < 0`
(`‖y‖ = 1`). By Prékopa's theorem `ℓ_i` is concave in `μ`, and with
`μ_i = M x̃_i` the isotropic regression likelihood is concave in `M`. The
restricted model of the isotropy null therefore has no local maxima, and
its fits (4 evaluations per bootstrap refit) cannot stall
(`logtangent::tests::isotropic_loglik_is_concave_in_mu` checks the Hessian
at 600 points, d = 3, 5, 8). Nothing of the kind holds once `V ≠ I`: the
frame `Ξ(μ̂)` couples the mean and the shape.

**Size and power on the hard scenario** (`examples/hard_scenario.rs
tests`; correlated covariates, μ and V(x) strongly x-dependent; n = 2000,
B = 199, 200 replications, s.e. 0.015). The true nulls are μ free of x3,
the shape free of x2 and x5 without effect; the false ones are a constant
shape, the shape free of x1, and x4 (a quarter-size effect) without effect.
Rejection rates at 5%, over RoC, D, M, LR and LM for the true nulls and
RoC, D, LR and LM for the false ones:

| shape, d | true nulls | constant shape | shape free of x1 | x4 (weak) |
|---|---|---|---|---|
| log-tangent, 3 | 0.030–0.065 | 1.00 | 1.00 | 0.92–0.96 |
| γ, 3 | 0.030–0.065 | 0.87–0.92 | 0.70–0.79 | 0.75–0.84 |
| log-tangent, 5 | 0.045–0.075 | 1.00 | 1.00 | 0.87–0.96 |
| log-tangent, 8 (score test, 100 replications; LM / M) | 0.020–0.070 / 0.030–0.040 | 1.00 | 1.00 | 0.93 |

`M` rejects the false nulls 3.5–31% of the time. (The γ and log-tangent data
are generated from the same coefficients read in each parameterization, so
their effect sizes differ; compare across rows with care.) γ at d ≥ 5 is
not in the table: on this data its full fits stop in traps (§2).

**Score statistic and score test.** With `s_i` the score of observation `i`
over the full model's free coefficients at the restricted fit, `g = Σ s_i`
and `I = Σ s_i s_iᵀ` (the empirical information),
`LM = gᵀ I⁻¹ g − g_fᵀ I_ff⁻¹ g_f`, where `f` indexes the coefficients the
restricted model frees. At an exact restricted maximum `g_f = 0` and the
second term vanishes; otherwise it removes what the optimizer left of their
gradient (`LM` is then the efficient score `g_r − I_rf I_ff⁻¹ g_f` in the
metric `[I⁻¹]_rr`). `LM` and `LR` are asymptotically equivalent under the
null and under local alternatives, and the bootstrap calibrates `LM` like
the other statistics. `LM` needs only the restricted fit, so `score_test`
bootstraps it, with `M`, from restricted refits alone. The full refits
(from two starts for γ) are most of the cost of a test, and on γ's rough
likelihood at d ≥ 5 they carry all of the full model's local-maxima noise.
The information has Kronecker structure, `s_i s_iᵀ = (x̃_i x̃_iᵀ) ⊗
(g_i g_iᵀ)`, so its entries are `pt(pt+1)/2 × bd(bd+1)/2` sums
`Σ_i x̃_ij x̃_ij' g_ik g_ik'`, one matrix product per chunk (1.7× fewer
operations than the Gram matrix of the `s_i` at d = 8, in one large
product), and `LM` comes from one Cholesky factor with the restricted
model's coefficients first (`Lz = g`, `LM = Σ_{a > k_f} z_a²`). `LM` costs
0.25 ms per replicate at d = 3, 0.55–1.1 ms at d = 5 and 2.2–4.2 ms at
d = 8 (n = 2000, 5 covariates; 2.3 ms of the last is the log-tangent
kernel). For the γ isotropy null `γ = 0` is a cone without a score, and
`LM` is `NaN`.

**Refit settings that were measured and kept** (`inference::tests::
refit_choices`, hard scenario, 40 replicates per null). A refit tolerance
of 1e−5 instead of 1e−6 saves about 10% of the evaluations but moves RoC
replicates by up to 0.12 of their bootstrap SD (0.5 at 1e−4): RoC's
bootstrap spread is small. γ's second full-refit start (from the original
full fit) finds a higher maximum at d = 3 in 3–10 of 40 replicates, by up to
3.9 log-likelihood units, so it stays although it doubles the cost of a γ
replicate.

**Shape-constant refits.** Under the shape-constant null (and in
intercept-only fits) every observation has the same log-tangent `S`, and
only the frame `Ξ(μ̂_i)` varies. One eigen-decomposition per evaluation
then serves all observations (`logtangent::SharedShape`): each costs
`O(dp + p²)` (`w_i = Qᵀz_i`), and the shape gradient is accumulated in the
eigenbasis, `W = Σ_i q̄_i w_i w_iᵀ`, and mapped back once as
`Q (F ∘ W) Qᵀ`. Restricted refits under that null: 3.3 → 0.8 ms at d = 5,
11.8 → 1.3 ms at d = 8 (n = 2000, one core).

**`M` has no first-order power against covariate effects.** `M` averages
the moment residuals `y_i² − E_0 y_i²` over all observations. An
alternative `b_0 + cδ` that adds an effect of a centred covariate `x_j`
changes `E y_i²` by `c (∂E y_i²/∂b · δ) x_ij` to first order, and the
average over `i` of that is `O(c · mean_i x_ij) ≈ 0`. Only second-order
terms remain. So `M` detects alternatives that move the average second
moment, such as anisotropy against the isotropy null as in the papers, and
not slopes. In the study below at d = 3, at the effect size where RoC, D,
LR and LM reject 14–95% of the time, `M` rejects 3–8%.

**Size and power when V is anisotropic** (`examples/aniso_study.rs`). The
setting is n = 2000, 5 standard-normal covariates + const and B = 199. The
data come from the coefficients of `examples/big_fit.rs::truth`, with the
null's coefficients set to zero (size; 300 replications, s.e. 0.013) or
scaled by c (power; 100 replications). No truth is isotropic. The median
over x of λ_max/λ_min, the shape eigenvalues of V(x), is:

| d | γ | log-tangent |
|---|---|---|
| 3 | 2.0 | 4.4 |
| 5 | 8.0 | 10.2 |
| 8 | 85 | 33 |

Under the shape-constant null the ratio is the same at every x: 2.0, 7.2
and 69 for γ, and 4.2, 8.4 and 26 for log-tangent. Log-tangent fits use the
simulator's frame (reference `e_d`), so the null tested is the null
simulated.

Rejection rates at 5% under the null (RoC / D / M / LR / LM):

| d | null | γ | log-tangent |
|---|---|---|---|
| 3 | μ free of x1 | .040 .050 .060 .047 .037 | .050 .050 .037 .047 .057 |
| 3 | shape free of x1 | .040 .037 .060 .050 .043 | .043 .040 .057 .060 .063 |
| 3 | x5 no effect | .053 .050 .047 .040 .040 | .033 .030 .053 .027 .030 |
| 3 | shape constant | .060 .053 .050 .053 .047 | .040 .040 .037 .053 .050 |
| 5 | μ free of x1 | .023 .023 .047 .030 .037 | .040 .033 .043 .040 .050 |
| 5 | shape free of x1 | .033 .033 .043 .023 .027 | .043 .043 .047 .043 .047 |
| 5 | x5 no effect | .040 .047 .057 .050 .053 | .057 .057 .060 .073 .067 |
| 5 | shape constant | .053 .053 .060 .043 .037 | .047 .047 .073 .040 .050 |
| 8 | μ free of x1 | – – .030 – .070* | .047 .047 .043 .030 .037 |
| 8 | shape free of x1 | – – .015 – .025* | .067 .070 .033 .070 .053 |
| 8 | x5 no effect | – – .030 – .065* | .033 .030 .047 .030 .027 |
| 8 | shape constant | – – .035 – .040* | .040 .043 .043 .037 .030 |

\* γ at d = 8: the score test only (M and LM; 200 replications, s.e. 0.015),
since a full γ test there takes 12–34 s on 15 threads.

No refit failed in any cell. The log-tangent tests at every d and the γ
tests at d = 3 hold their level, and so does γ's LM at d = 8 (0.050 on
average; M there is slightly conservative, 0.015–0.035). The γ tests at
d = 5 are slightly conservative: LR, RoC and LM reject at 0.037–0.039 on
average over the four nulls (s.e. 0.006). This is the restricted refits'
share of the local-maxima problem above. A bootstrap
restricted refit starts at the bootstrap truth and stops at the first
maximum it reaches. Against a second fit of the same data from scratch, it
is lower in 3–8 of 40 replicates (by up to 0.97) and higher in 0–6
(`gamma_refits::restricted_refit_maxima`). A low ℓ_0^(b) inflates LR^(b)
and LM^(b), which makes the test conservative rather than liberal. `M`
does not use the refit's log-likelihood and holds its level (.043–.060).

Power at c = 0.25, and at c = 0.1 for d = 5 (RoC / D / M / LR / LM; γ at
d = 8 not run):

| d | alternative | γ | log-tangent |
|---|---|---|---|
| 3 | μ depends on x1 | .93 .94 .08 .95 .94 | .90 .91 .03 .85 .86 |
| 3 | shape depends on x1 | .14 .14 .08 .15 .15 | .23 .24 .04 .23 .24 |
| 3 | x5 has an effect | .81 .82 .07 .80 .78 | .94 .94 .03 .94 .93 |
| 3 | shape depends on x | .27 .26 .04 .24 .26 | .62 .62 .07 .61 .60 |
| 5 | μ depends on x1 | 1 1 .33 1 1 | 1 1 .30 1 1 |
| 5 | shape depends on x1 | .99 .99 .11 .99 .98 | .53 .53 .06 .58 .53 |
| 5 | x5 has an effect | 1 1 .06 1 1 | 1 1 .10 1 1 |
| 5 | shape depends on x | .99 .99 .16 1 1 | 1 1 .04 .99 .98 |
| 5, c = 0.1 | μ depends on x1 | .35 .39 .12 .40 .41 | .57 .52 .04 .70 .72 |
| 5, c = 0.1 | shape depends on x1 | .26 .29 .08 .32 .27 | .14 .14 .05 .14 .13 |
| 5, c = 0.1 | x5 has an effect | .52 .52 .07 .49 .51 | .71 .70 .09 .75 .75 |
| 5, c = 0.1 | shape depends on x | .26 .26 .09 .26 .34 | .20 .20 .10 .21 .23 |
| 8 | μ depends on x1 | – | 1 1 .17 1 1 |
| 8 | shape depends on x1 | – | 1 1 .06 1 .99 |
| 8 | x5 has an effect | – | 1 1 .07 1 1 |
| 8 | shape depends on x | – | 1 1 .07 1 1 |

`LM` has the power of `LR`: the two differ by at most 0.08, in either
direction, and mostly by 0.02 or less. RoC and D are close to both, except
for a log-tangent effect on the mean at c = 0.1, which LR and LM detect
more often (0.70–0.72 against 0.52–0.57). `M` has almost no power (see above).

## 7. Log-tangent shape parameterization (smooth option) (`src/logtangent.rs`)

An opt-in alternative to `γ` that maps smoothly onto the same ESAG family,
for regressions whose likelihood would otherwise have cliffs at the `γ`
singular set (§2). `EsagRegressor::with_shape(Shape::LogTangent)` fits it
(the frame's reference direction defaults to the data's mean direction and
is stored with the model), and the penalized estimators of §8 use it.

**The model depends on the frame.** `s` is defined relative to the frame
`(m, E)`, and transporting from another reference rotates the tangent basis
by a `μ̂`-dependent amount. So "`s` linear in x" in one frame is not linear
in another, and each reference direction defines its own regression model.
The default `m` is the data's mean direction, which keeps `μ̂` far from the
one singular point `−m`. It is stored with the fit and kept fixed by every
refit (and by every cross-validation fold of a penalized fit, which
before took the mean direction of its own training rows and so fitted a
slightly different model, up to 1e−3 relative in the CV score). Summaries
print it, and `results.lt_reference` (Python) returns it for reuse on other
data. It is still chosen from the data. On data from a log-tangent model
defined with `m = e_d` (§5b), the fit with the default `m` has an error in
`V̂` that stops shrinking at the `1/√n` rate. At d = 3, n = 10000 the
geodesic error is 0.108, against 0.055 with `m = e_d`; at d = 8 it is 0.44
against 0.18. To compare with a model defined in a given frame, set
`EsagRegressor::with_lt_reference` (Python: `lt_reference=`). The `γ`
model has the analogous dependence on the coordinate axes of `y`, through
`ONB(μ)`, but there it is a fixed convention.

**Same family, different regression.** For one distribution, `γ` and `s`
are two coordinate systems on the same set of ESAG laws. In a regression
each is made *linear* in the covariates, `γ_i = Γ x̃_i` or `s_i = S x̃_i`,
and those are different models for how the shape moves with `x`. They are
not nested, and the mean model and `μ`'s coefficients are the same in both.
On data simulated from a `γ`-linear model (d = 3, n = 2000, 5 covariates)
the `γ` fit has log-likelihood −2312.6 and the log-tangent fit −2381.4.
Choose between them by AIC/BIC (same number of parameters) or by
interpretation. The log-tangent model is the default only where the `γ`
model's likelihood is unreliable to optimise (d ≥ 5 with covariates; the
Python front end's `shape="auto"`, which raises a `ShapeWarning` when it
makes that choice).

**Definition.** Let `p = d − 1`. Fix a reference direction `m ∈ S^{d−1}` and
an orthonormal basis `E` (`d × p`) of `m⊥` (`LtFrame`; by default `m = e_d`,
`E = [e_1, …, e_{d−1}]`). For `μ̂ = μ/‖μ‖ ≠ −m` let `R(m→μ̂)` be the minimal
rotation taking `m` to `μ̂`. It rotates in `span(m, μ̂)` and fixes the
orthogonal complement. Let `Ξ(μ̂) = R(m→μ̂) E`, a smooth orthonormal basis
of `μ̂⊥`. Then

```
V = μ̂μ̂ᵀ + Ξ exp(S) Ξᵀ,     S = Sᵀ ∈ ℝ^{p×p},  tr S = 0,
```

so `Vμ = μ` and `det V = e^{tr S} = 1`. Given `μ`, `S = log(Ξᵀ V Ξ)` is
unique (`s_from_covariance`), so there are no equivalent parameters and the
map `(μ, S) ↦ (μ, V)` is a diffeomorphism wherever `μ̂ ≠ −m`.

**Coordinates.** `s ∈ ℝ^{(d−2)(d+1)/2}`, the length of `γ`, holds orthonormal
coordinates of `S` in the Frobenius inner product (`s_to_matrix`,
`matrix_to_s`):

* `s_{k−1}`, `k = 1..p−1`: the diagonal on the Helmert basis
  `h_k = (1, …, 1, −k, 0, …, 0)/√(k(k+1))` (`k` ones), so
  `S_jj = Σ_{k>j} s_{k−1}/√(k(k+1)) − j s_{j−1}/√(j(j+1))`;
* then the pairs `a < b` in row-major order, `S_ab = S_ba = s/√2`.

That is `(p−1) + p(p−1)/2 = (d−2)(d+1)/2` coordinates, with `‖s‖ = ‖S‖_F`.
`‖s‖` is the affine-invariant Riemannian distance of the tangent block
`Ξᵀ V Ξ = e^S` from `I`, and `s = 0` is isotropy.

**The frame in O(d).** With `H_x = I − 2xxᵀ/(xᵀx)` and `v = μ̂ + m`,
`R(m→μ̂) = H_v H_m`: `H_m m = −m`, `H_v(−m) = μ̂` (as
`m·v/(v·v) = (1 + c)/(2(1 + c)) = ½`, `c = mᵀμ̂`), and both reflections fix
`span(m, μ̂)⊥`. This is the closed form `R = I + K + K²/(1 + c)`,
`K = μ̂mᵀ − mμ̂ᵀ`; the tests check one against the other. Since `H_m E = E`,

```
Ξ(μ̂) = H_v E,     z = Ξᵀ y = Eᵀ u,     u = H_v y = y − κ v,     κ = 2 (v·y)/(v·v).
```

`v·v = 2(1 + c)` vanishes only at `μ̂ = −m`. Near that point the frame turns
fast: `‖dΞ‖_F ≤ 2√2 ‖dμ̂‖/‖μ̂ + m‖`. The singular point cannot be avoided,
since `S^{d−1}` has a global smooth tangent frame only for `d ∈ {2, 4, 8}`,
so `m` should be chosen near the data (for example the mean direction).
Guards: at `μ̂ = −m` exactly, `v = E_0` (the rotation by `π` in the plane
`(m, E_0)`). At `μ = 0`, `μ̂ = m`. Both frames are constant there.
`LtFrame::new` takes `E` from `onb(m)`, so that `Ξ(m)` is the basis `γ` uses at
`μ̂ = m`. Entries of `m̂` below `1e-70` are zeroed first, because `onb`
underflows to NaN for inputs such as `(1e-155, 0, 0, 1)`. Two Gram–Schmidt
passes then restore orthonormality against the true `m̂`.

**Density.** `q = yᵀV⁻¹y = b² + zᵀ e^{−S} z` with `b = μ̂·y`, and the log-density
is that of §1. With `S = QΛQᵀ`, `w = Qᵀz`,

```
zᵀ e^{−S} z = Σ_a e^{−λ_a} w_a².
```

`loglik_lt` scales `S` to unit Frobenius norm. It diagonalises `S` by cyclic
Jacobi in round-robin order: in each of the `p − 1 + (p mod 2)` rounds the
`⌊p/2⌋` rotations act on disjoint planes, so their angles are computed
together and the latencies overlap. It updates only the upper triangle and
applies each rotation to `z` as it goes, so `Q` is never formed for the
value. The angle uses `h = a_yy − a_xx`, `g = 2a_xy`, `r = √(h² + g²)`,
`c = (r + |h|)/√(2r(r + |h|))`, `s = sgn(h) g/√(2r(r + |h|))`, which needs two
square roots and one division in sequence. For `|a_xy| < 5e-9 |h|` it uses
`t = a_xy/h`, `c = 1`, which is exact to rounding. Jacobi stops at an
off-diagonal norm `≤ 1e-15 ‖S‖_F`. On random `S` with `p = 7` that takes 5.0
sweeps on average and at most 6. At `S = 0` the decomposition is skipped.

**Gradient** (`loglik_lt_grad`, reverse mode, value bit-identical to
`loglik_lt`). As in §2, `∇ log f = q̄ ∇q + ȳmu y − μ (μ block)` with
`ᾱ = alpha + (log M_p)'(alpha)`, `q̄ = −(d + ᾱ alpha)/(2q)`, `ȳmu = ᾱ/√q`,
so it remains to find `∇q`.

* *Shape.* By the Daleckii–Krein formula,
  `D e^{−S}[H] = Q (F ∘ (QᵀHQ)) Qᵀ`, where `F_ab = (e^{−λ_a} − e^{−λ_b})/(λ_a − λ_b)`
  and `F_aa = −e^{−λ_a}`. Hence `dq = zᵀ D e^{−S}[dS] z = Σ_ab F_ab w_a w_b (QᵀdS Q)_ab`,
  so `∂q/∂S = G = Q (F ∘ wwᵀ) Qᵀ`. It is computed as `G = Aᵀ F A` with
  `A = diag(w) Qᵀ`, at about `1.5p³` operations. For a gap
  `δ = λ_b − λ_a ∈ [0, ½)` (sorted so that `δ ≥ 0`),
  `F_ab = e^{−λ_a} expm1(−δ)/δ`, which is accurate for every `δ` and tends to
  `−e^{−λ_a}`. `G` is smooth in `S` even where eigenvalues coincide, where
  `Q` is not unique. Finally `∂q/∂s_i = ⟨G, B_i⟩` for the basis matrices:
  `(Σ_{j<k} G_jj − k G_kk)/√(k(k+1))` for the diagonal coordinates and
  `√2 G_ab` for the off-diagonal ones. At `S = 0`, `G = −zzᵀ` exactly.
* *Projection.* `∂q/∂z = 2 e^{−S} z = 2Q(e^{−λ} ∘ w)` and `ū = E z̄`. From
  `u = y − κv` and `dκ = (2/(v·v)) u·dv`,
  `v̄ = −κ ū − (2 (ū·v)/(v·v)) u`.
* *Mean direction.* `v = μ̂ + m` and `b = μ̂·y` give `μ̂̄ = v̄ + 2b y`, and
  `μ̂ = μ/‖μ‖` gives `∂q/∂μ = (μ̂̄ − (μ̂·μ̂̄) μ̂)/‖μ‖`. On the guarded branches
  the frame terms are zero, and at `μ = 0` all of `∂q/∂μ` is.

**Hessian** (`loglik_lt_hess`, exact, one eigen-decomposition, value and
gradient bit-identical to the functions above). With
`Φ(q, ymu) = −(d/2) log q + ψ(alpha)`, `ψ(a) = a²/2 + log M_p(a)` and
`e = (y, 0)`,

```
∇² log f = Φ_q ∇²q + Φ_qq ∇q∇qᵀ + Φ_qy (∇q eᵀ + e∇qᵀ) + Φ_yy eeᵀ − [I 0; 0 0],
Φ_qq = d/(2q²) + ψ'' a_q² + ψ' a_qq,  Φ_qy = ψ'' a_q a_y + ψ' a_qy,  Φ_yy = ψ'' a_y²,
a_q = −alpha/(2q), a_y = q^{−1/2}, a_qq = 3 alpha/(4q²), a_qy = −q^{−3/2}/2.
```

* *Shape block.* By the second-order Daleckii–Krein formula, for symmetric
  `H, K`,
  `zᵀ D²e^{−S}[H, K] z = 2 Σ_abc T_cab w_a w_b H̃_ac K̃_cb` with
  `T_cab = f[λ_a, λ_c, λ_b]`, the second divided differences of
  `f(x) = e^{−x}`. With the eigenvalues sorted, `lo ≤ mid ≤ hi`:
  * for `hi − lo ≥ ½`, `T = (F[lo, mid] − F[mid, hi])/(lo − hi)`;
  * below that,
    `T = e^{−lo} Σ_{n=2}^{17} (−1)ⁿ h_{n−2}(mid − lo, hi − lo)/n!`, using the
    complete homogeneous polynomials `h_m(x, y) = xᵐ + y h_{m−1}(x, y)`.
    This holds to `2e-14` against quadrature, ties included.

  Expanding `H = e_x e_uᵀ`, `K = e_y e_vᵀ` gives
  `2 Wt(u, y, x, v)` with `Wt(u, y, x, v) = Σ_c Q_uc Q_yc G_c[x][v]` and
  `G_c = R_w T_c R_wᵀ`, `R_w = Q diag(w)`. `Wt` is symmetric within
  `{u, y}` and within `{x, v}`, so it is a `p(p+1)/2`-square matrix
  `W = A Gᵀ`. For basis matrices `B_ab`, `B_a'b'`,
  `∂²q/∂s∂s' = Wt(b, a', a, b') + Wt(b, b', a, a') + Wt(a, a', b, b') + Wt(a, b', b, a')`.
  Helmert coordinates take `h_k`-weighted sums of `Wt(x, a', x, b')` or
  `Wt(x, y, x, y)`, by prefix sums. The block costs about
  `p⁵/4 + 1.5p⁴` operations, 9k at p = 7. Gram products of the transformed
  basis matrices `QᵀB_iQ` would need about `ns² p²`, which is 36k.
* *Cross block.* `∂/∂s_i (∂q/∂z) = 2 D e^{−S}[B_i] z = 2Q(F ∘ B̃_i)w`, with
  `B̃ = QᵀBQ`. Writing `Φ = F R_wᵀ`, the vector `(F ∘ B̃_ab)w` has entries
  `(Q_aα Φ_αb + Q_bα Φ_αa)/√2`. It is chained through
  `K = Qᵀ ∂z/∂μ̂ = −κ (EQ)ᵀ − (2/(v·v)) (EQ)ᵀv uᵀ` and `∂μ̂/∂μ = P/‖μ‖`, where
  `P = I − μ̂μ̂ᵀ`.
* *Mean block.* With `μ̂` taken as a free vector,
  `A = 2yyᵀ + 2 Kᵀe^{−Λ}K + Σ_k ū_k ∇²u_k`, where for `u = y − κv`
  `Σ_k ū_k ∇²u_k = (2κ(ū·v)/(v·v)) I + (4(ū·v)/(v·v)²)(vuᵀ + uvᵀ) − (2/(v·v))(uūᵀ + ūuᵀ)`.
  Then `μ̂ = μ/‖μ‖` gives
  `∂²q/∂μ² = (PAP − (g⊥μ̂ᵀ + μ̂g⊥ᵀ + γP))/‖μ‖²`, with `g = ∂q/∂μ̂`,
  `γ = g·μ̂` and `g⊥ = Pg`.

The Hessian matches fourth-order differences of the gradient to `2e-10` or
better at the best step, and to `1.5e-8` in the test sweep (d = 3..10, including
`s = 0` and tied eigenvalues). It is exactly symmetric.

**Relation to `γ`.** It is the same family. For random `(μ, s)`, the
canonical `γ = gamma_from_covariance(μ, V)` reproduces `loglik_lt` through
`loglik_eta` to `2e-12` (`tests/logtangent.rs`), and conversely every
`V(μ, γ)` has exact log-tangent coordinates. The `γ` side has two blind
spots that the tests document:

* *Exact ties* `λ_j = λ_{j+1}` of the tangent eigenvalues give a group of
  norm 0. That erases the group's rotation, so the canonical `γ` misses a
  generic tied `V` by `O(1)`: tied `V` are limits of the `γ` family, not
  members.
* *Frame-aligned shapes* (diagonal `S`): the structure of `onb` often puts
  the canonical `γ` exactly on the singular set, where it need not
  reproduce `V`. This happens in 20% of random cases at d = 6 and 75% at
  d = 8.

**Cost** (Apple M5 Pro, ns per call, random `η` with a full `S`;
`examples/logtangent_bench.rs`):

| d | `loglik_eta` | `loglik_eta_grad` | `loglik_lt` | `loglik_lt_grad` | `loglik_lt_hess` |
|---|---|---|---|---|---|
| 3 | 32 | 39 | 46 | 58 | 116 |
| 4 | 42 | 67 | 156 | 195 | 362 |
| 5 | 51 | 99 | 258 | 335 | 674 |
| 6 | 74 | 119 | 454 | 586 | 1248 |
| 7 | 88 | 144 | 626 | 720 | 1882 |
| 8 | 102 | 169 | 881 | 1170 | 3149 |

The eigen-decomposition dominates the value and the gradient. The Hessian
costs 2 to 2.7 times the gradient for d ≤ 8. Central differences of the
gradient would need `2 dim(η)` gradients, 70 at d = 8, about 26 times
more. At `S = 0` the decomposition is skipped: the value then costs 1.1 to
1.3 times `loglik_eta`, and the value and gradient 0.8 to 1.2 times
`loglik_eta_grad`.

## 8. Penalized fits (Project 3) (`src/penalized.rs`, `src/optim/prox.rs`)

Yu & Huang's LASSO-based inference adds a penalty on the shape to the
negative log-likelihood, on the sum scale of the R code: `−Σℓ_i + λP`, so
`λ` is the paper's `nλ_n` (the Python front end's `alpha` is `λ/n`, as in
statsmodels). The intercept column is penalized only by `Target::Shape`. `docs/project3/SPEC.md` describes exactly what the R code does;
`docs/project3/METHODOLOGY.md` reviews the method and motivates the
corrections below.

**Why not penalize γ.** Exact zeros of a group `j ≥ 2`, or of a trailing
pair, sit on the discontinuity set of §2, and `V` is not differentiable at
any zero group. Both of the paper's simulation truths lie there. The paper's
six methods are therefore offered as a **reproduction mode**
(`Solver::PaperReproduction`; `PenalizedEsag::paper(name)`): the
weights `b_g^r` with `b = 1/|γ̃|`, `r = max(1, ‖μ̃‖/‖μ‖)` inside the
objective as coded, smoothing continuation (`|x| → √(x²+ε²)`, ε = 1e−2,
1e−3) with exact gradients from several starts, then the R rule
`|x| < 0.01 → 0` and its BIC/AIC/SBIC. On the R fixtures the coded
objective at R's 18 optima is reproduced to 1e−9 and never decreased; from
R's start the optimum reached depends on the path (the objective often has
no attained maximum: its supremum is approached on the favourable side of
a discontinuity), as it does in R across machines.

**The family.** A penalty is `Penalty × Target × Grouping × Weighting`:

* `Penalty`: `Lasso` (`Σ w_j |θ_j|`), `GroupLasso` (`Σ_g w_g ‖θ_g‖₂`,
  base weight `√|g|`), `SparseGroupLasso{l1_ratio}`
  (`α Σ w_j|θ_j| + (1 − α) Σ_g w_g ‖θ_g‖₂`, exact prox: soft-threshold,
  then block soft-threshold), `EigenGap` (below);
* `Target`: `Slopes` (every non-intercept coefficient; default),
  `MeanSlopes`, `ShapeSlopes`, `Shape` (shape rows, intercept included);
* `Grouping`: `Covariate` (a covariate's target coefficients), `CovariatePart`
  (its mean slopes and its shape slopes separately), `Parameter` (one row
  of `B`), `All`, `GammaGroups` (the papers' γ groups), `Custom`;
* `Weighting`: `Plain`, `Adaptive(r)` (`b_g^r`, `b_g = 1/φ_g(θ̃)`), `RoC`.

Slope penalties work in either shape (log-tangent by default). Covariates
are standardized internally, so the penalty acts on standardized slopes and
selection does not depend on units; coefficients are reported on the
original scale.

**Corrected designs (log-tangent shape `S`, §7).**

| preset | penalty | zero set |
|---|---|---|
| `isotropy()` | `w‖s‖₂` over all shape coefficients | isotropic errors (the paper's eq. 4.4) |
| `eigen_gap()` (i.i.d.) | `Σ_j w_j (σ_(j+1) − σ_(j))`, sorted eigenvalues of `S` | ties `ξ_j = ξ_{j+1}` anywhere (frame-free partial isotropy) |
| `adaptive_group_lasso().grouping(CovariatePart)` (regression) | `Σ_k w_k ‖B_μ[:,k]‖ + Σ_k w_k ‖B_s[:,k]‖` | covariate `k` does not move the mean / the shape |

The paper's group norm is spectral, `‖γ_(j)‖ = e^{gap_j} − 1`, so `EigenGap`
is the frame-free version of its group LASSO. Weights are `b_g^r` with
`b_g = 1/φ_g(θ̃)` from the unpenalized fit: `Plain` (`r = 0`; for `EigenGap`
the concave baseline `w_j = j(k−j)/k`), `Adaptive` (`r = 1`) or `RoC`
(`r = max(1, ‖μ̃‖/‖μ‖)`), either solved as a fixed point (freeze `r`, solve,
update; default) or jointly as in the paper (its `μ`-gradient
`λ Σ_g φ_g w_g ln(b_g) ∂r/∂μ` pushes `‖μ̂‖` down).

**Solver.** Proximal Newton (`optim::prox`): the model
`∇h·Δ + ½ΔᵀHΔ + λΣ w_g φ_g` with the exact Hessian (reused while full
steps match the predicted decrease, and carried along the λ path) is
minimised by FISTA with restarts using the exact prox of the atoms: block
soft-thresholding for group norms; eigen-decomposition plus weighted
isotonic regression (PAVA) for the gaps (exact for any weights, with exact
ties for infinite weights). Steps are accepted by backtracking on the true
objective; stationarity is certified by the gradient mapping. Zeros and
ties are exact, never thresholded.

**Selection.** BIC (default) or AIC at the *relaxed* refit, the
unpenalized maximum over the selected model, with the model dimension as
df: all coefficients minus those of zero groups, or for a tie pattern with
`K` distinct eigenvalues of multiplicities `m_i`,
`d + (K − 1) + k(k−1)/2 − Σ m_i(m_i−1)/2`. Ties in the criterion go to the
sparser model, then the smaller λ. Or K-fold cross-validation with shuffled,
seeded folds of the held-out log-likelihood (the paper's RMSE (5.5) is
available but ignores `‖μ̂‖`). The relaxed refit at the selected λ is
returned with standard errors when the selected model is a coefficient
mask.

**Small samples.** Where the unpenalized fit is degenerate (§3, "When the
MLE does not exist") three things change:

* the path runs from the null model down (every penalized atom zero, then
  decreasing λ with warm starts), instead of up from the degenerate fit:
  from there every λ stayed stuck at the degenerate point (all 70 atoms
  active and KKT residual 90 even at λ_max, d = 5, n = 100), and BIC chose
  the full model;
* a λ whose relaxed fit (or penalized fit, without relaxing) is degenerate
  gets an infinite BIC/AIC, so it cannot be selected however large its
  log-likelihood;
* adaptive weights come from the plain penalty's selected model (the
  two-stage adaptive LASSO for `p ≈ n`), not from the degenerate MLE.

At d = 5, n = 100 the five LASSO-type presets then select sparse models
with regular relaxed refits (0–2 of the 6 true covariate parts, no false
ones; at n = 100 the data cannot support more).

**Support recovery on the hard scenario** (`examples/hard_scenario.rs
lasso`; log-tangent, BIC at relaxed refits, n = 2000, 40 replications, 12
at d = 8; strong parts: x1 on μ and the shape, x2 on μ, x3 on the shape):
adaptive LASSO finds 4.00 of 4 strong parts at d = 3 and 5 with 0.00 and
0.28 of 4 false parts (3.75 and 0.50 at d = 8); plain LASSO finds as many
but admits 0.23–1.40 false parts; the group penalties (mean / shape parts
per covariate) find all strong parts at d = 3 (3.55–3.88 at d = 5) with
≤ 0.10 false, but at d = 8 none of the shape effects, because BIC charges
`27 log n` for a 27-coefficient shape group. Five-fold cross-validation of
the held-out log-likelihood errs the other way (nearly every part
admitted). One path takes 82–103 ms at d = 3, 0.33–0.55 s at d = 5 and
1.6–3.0 s at d = 8 on one core.

Proximal Newton also stops once three successive steps no longer decrease
`F` beyond rounding (relative 1e−13) with the KKT residual already below
`√tol`: near the tolerance an adaptive weight's atom can flip at its
threshold, and the solver used to refresh the Hessian at every iteration
until `max_iter` (4.95 s of a 5.7 s adaptive LASSO path at d = 5,
n = 2000; the path now takes 0.63 s).
