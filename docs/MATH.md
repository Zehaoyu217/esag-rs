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

**Singular set.** Where the trailing coordinates of a group vanish, for
example `(γ_4, γ_5) = 0` with `γ_3 ≠ 0` for `d = 4`, the angle
`θ = atan2(γ_5, γ_4)` is undefined but still rotates `V`. So `γ ↦ V` is
discontinuous on this codimension-2 set (Yu & Huang 2024, Appendix B), and
gradients blow up near it. An optimiser started badly can be drawn towards
it and stop with a huge gradient and a poor likelihood. `esag` detects this
with the gradient check (`MAX_MEAN_GRADIENT`). The default `Init::Auto` then
falls back to its other start or to random restarts, and warm-started refits
retry from the R-style start.

## 3. Regression (`src/regression.rs`)

`Y_i | x_i ~ ESAG(μ_i, γ_i)`, `(μ_i, γ_i) = B x̃_i` with `x̃_i = (1, x_i)`.
The identity link is possible because the parameter space is all of
`ℝ^{d + (d−2)(d+1)/2}`. `B` has rows `(μ_1..μ_d, γ_1..γ_g)` and one column
per design column.

**Fitting.** `esag` maximises `ℓ(B) = Σ_i log f(y_i; B x̃_i)` with L-BFGS
(`src/optim/lbfgs.rs`). The line search is strong Wolfe with Hager–Zhang
approximate-Wolfe acceptance near the optimum. The gradient is **exact**:
the per-observation log-density is generic over a `Scalar` type, and
evaluating it with forward-mode dual numbers `Dual<N>`, `N = dim η`, gives
`g_i = ∂ log f/∂η` in one pass. Then `∂ℓ/∂B = Σ_i g_i x̃_iᵀ`, computed in
parallel over observations. The R code uses `optim(method = "BFGS")` with
central finite differences of step 1e-3, which costs `2·dim(B)` likelihood
evaluations per gradient.

**Starts.** By default `esag` runs two starts in parallel and keeps the
better fit:

* a moment start: `μ` along the mean resultant, with
  `‖μ‖ = (Π σ_j)^{−1/(2(d−1))}` from the eigenvalues of the tangent scatter
  `S ≈ V/‖μ‖²`, and `γ` from the shape of `S` via the inverse map;
* the R code's start: `μ` coefficients 1, `γ` ≈ 0.

The starts share an iteration budget. When one converges after `N`
iterations, the other may run to at most `max(2N, N + 50)` iterations in
total. A start that converges at a comparable pace can still win, which
matters because regressions have genuine local maxima. A start trapped near
the singular set below is cut off. If neither start passes the convergence
check (the optimiser converged *and* the mean per-observation gradient is
≤ 1e-3), the fit retries from random perturbations. The L-BFGS memory
defaults to the number of free parameters, clamped to [10, 100]; with less
memory than parameters, convergence is 2 to 5 times slower.

**Standard errors.** `esag` reports the inverse observed information. It is
assembled as `Σ_i (x̃_i x̃_iᵀ) ⊗ ∇²_η ℓ_i`, with per-observation Hessians from
second-order dual numbers (`hdual`) at d = 3, and from central differences
of the exact gradient for d ≥ 4. The two agree to about 1e-8, and the
differences are faster there. Up to three simplified Newton steps reuse this
Hessian to polish the L-BFGS optimum to ‖∇ℓ‖ ≈ 1e-10. Intercept-only fits
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

## 6. Hypothesis tests (`src/inference/tests.rs`)

Restricted fit (`μ_{0i}`, `γ_{0i}`) versus full fit (`μ_{ai}`):

```
RoC = mean_i ‖μ_{ai}‖ / ‖μ_{0i}‖
D   = mean_i {2 − cos∠(μ_{0i}, μ_{ai})} ‖μ_{ai}‖ / ‖μ_{0i}‖
M   = ‖ mean_i ( y_i² − Ê_0 y_i² ) ‖²     (elementwise squares; Ê_0 by Monte Carlo)
LR  = −2 (ℓ_0 − ℓ_a)
```

`M` is the squared norm, as in the R code and the papers' tables (CSDA eq. (4)
prints the norm; p-values are unaffected). Null distributions come from a
parametric bootstrap under the restricted fit,
refitting both models each time, and `p = #{stat^(b) > stat}/B`. Supported
nulls are isotropy (`γ ≡ 0`), `μ` free of feature `j`, `γ` free of feature
`j`, both, or any custom mask. Two differences from the R code: bootstrap
responses are drawn from each observation's own null parameters (the R code
indexes them by the bootstrap counter), and every `d` is supported (the R
isotropy and γ-restricted fits hard-code `d = 4`).
