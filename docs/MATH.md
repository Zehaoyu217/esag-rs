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

The log-tangent parameterization (§7) describes the same distributions
without this set.

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
reverse-mode gradient above. Chunk results are summed in chunk order, so
the likelihood, and therefore every fit, is bit-for-bit reproducible
whatever the number of threads.

**Starts.** By default (`Init::Auto`) `esag` races two starts:

* a moment start: `μ` along the mean resultant, with
  `‖μ‖ = (Π σ_j)^{−1/(2(d−1))}` from the eigenvalues of the tangent scatter
  `S ≈ V/‖μ‖²`, and `γ` from the shape of `S` via the inverse map;
* the R code's start: `μ` coefficients 1, `γ` ≈ 0.

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
mean square). A fit that still has not converged is returned with
`converged = false`: its likelihood is the highest found, but it sits on a
cliff of the parameterization, so standard errors there are not meaningful.
The L-BFGS memory defaults to the number of free parameters, clamped to
[10, 200].

**Standard errors.** `esag` reports the inverse observed information. It is
assembled as `Σ_i (x̃_i x̃_iᵀ) ⊗ ∇²_η ℓ_i`, with per-observation Hessians from
second-order dual numbers (`hdual`) at d = 3, and from central differences
of the exact gradient for d ≥ 4. The two agree to about 1e-8, and the
differences are faster there. The Kronecker sum is one matrix product per
chunk over the unique entries (`x_{j1}x_{j2}`, `j1 ≤ j2`, times
`H_i[k1,k2]`, `k1 ≤ k2`). When standard errors are requested, L-BFGS hands
over once the mean gradient per observation is below 1e−4, and up to eight
simplified Newton steps reuse this one Hessian to polish the optimum to
‖∇ℓ‖ ≈ 1e-10 (if the Hessian is not negative definite there, L-BFGS
finishes at full tolerance first). Intercept-only fits
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

## 7. Log-tangent shape parameterization (smooth option) (`src/logtangent.rs`)

An opt-in alternative to `γ` that maps smoothly onto the same ESAG family,
for regressions whose likelihood would otherwise have cliffs at the `γ`
singular set (§2). Nothing else in the crate uses it yet.

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

| d | `loglik_eta` | `loglik_eta_grad` | `loglik_lt` | `loglik_lt_grad` |
|---|---|---|---|---|
| 3 | 30 | 38 | 45 | 57 |
| 4 | 41 | 65 | 154 | 200 |
| 5 | 49 | 96 | 254 | 334 |
| 6 | 73 | 118 | 454 | 594 |
| 7 | 86 | 143 | 622 | 721 |
| 8 | 102 | 167 | 878 | 1177 |

The eigen-decomposition dominates. At `S = 0` it is skipped: the value then
costs 1.1 to 1.3 times `loglik_eta`, and the value and gradient 0.8 to 1.2
times `loglik_eta_grad`.
