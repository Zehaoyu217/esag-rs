# METHODOLOGY — review, corrections and the recommended design for Rust

Labels used throughout:

* **[PAPER]**: as stated in the draft.
* **[CODE]**: as implemented in the R code (SPEC.md).
* **[DEVIATION]**: a change I recommend, with the reason.
* **[EXTENSION]**: not in the paper.

Every factual claim is backed by a script in `tmp/p3/scripts/`; the evidence index is §10. An
independent check, `tmp/sing_check.R`, written by the agent implementing the Rust side and built on
`esag_core.R` (the user's `ESAG_Project2.R` functions), agrees with mine (built on `Basic_Final.R`).

---------------------------------------------------------------------------------------------

## 0. Decisions in one page

1. **The paper's γ-parameterization cannot carry sparsity in general.** γ ↦ V is discontinuous
   at every zero group that has a nonzero group below it, and at every vanishing trailing pair of
   a nonzero group above a nonzero lower group. It is non-differentiable at every zero group,
   γ = 0 included. Coordinate zeros depend on the frame and on the order of the d coordinates.
   Both simulation truths sit on discontinuity points (§1). Of the paper's penalties, only (4.4)
   (zero set γ = 0) is well-posed. GLASSO-type penalties become well-posed only when their zero
   patterns are restricted to **prefixes** γ₍₁₎ = … = γ₍m₎ = 0, which is equivalent to
   ξ₁ = … = ξ_{m+1} (§1.7).
2. **Recommended estimator [DEVIATION]: penalize the smooth log-tangent shape S.**
   V = μ̂μ̂ᵀ + Ξ(μ̂) exp(S) Ξ(μ̂)ᵀ, S ∈ Sym₀(d−1), in Frobenius-orthonormal coordinates s (Helmert
   diagonal, √2-scaled off-diagonals). The penalties are (§2):
   * P_iso = w‖s‖₂, the analog of (4.4);
   * P_gap = Σ_j w_j (σ_(j+1) − σ_(j)) on the sorted eigenvalues of S. This is the frame-free
     completion of GLASSO/aGLASSO/RoC-GLASSO: the paper's group norm is exactly
     ‖γ₍j₎‖ = e^{σ_(j+1)−σ_(j)} − 1;
   * driver-level groups in regression;
   * an elementwise L1 on off-diagonal s, offered only with a declared, meaningful basis.

   All prox operators are exact and cheap, including eigen + PAVA for P_gap with arbitrary weights.
3. **Algorithm**: FISTA with backtracking and function-value restart on the sum-scale objective
   F = −Σℓ_i + λP (λ in the R code's units, λ_n = λ/n), then a support-restricted L-BFGS polish
   and a KKT certificate. RoC is computed either as the paper's joint optimum (block scheme, §3.3)
   or, recommended, as a fixed point (§3.4), because the joint version pushes ‖μ̂‖ down.
4. **λ selection**: BIC at the *relaxed* refit, with df = the selected model's dimension
   (closed forms in §5). CV uses shuffled, seeded folds and held-out log-likelihood. RMSE (5.5)
   stays only for reproducing the paper: it ignores ‖μ̂‖, and in the authors' CV runs it chose
   λ = 0 in 43% of replicates.
5. **Paper-reproduction mode** keeps the six methods exactly as coded in γ (SPEC §2): smoothing
   continuation + L-BFGS, the 0.01 threshold and R's BIC. Rust must reach an objective ≥ R's on
   the fixtures. It is labelled reproduction-only, with the warnings of §1.
6. **Regression (d = 7–8, 5–6 drivers) [EXTENSION]**: use S. Penalize driver-level shape groups
   ‖B_s[:,k]‖ and mean groups ‖B_μ[:,k]‖, never the intercept, with standardized covariates.
   Use fixed r = 1 by default and RoC(B) = max(1, mean_i ‖μ̃_i‖/‖μ_i‖) as an option.

---------------------------------------------------------------------------------------------

## 1. What the γ-parameterization does near zeros

Notation: γ₍g₎ = G = (G₁, …, G_{g+1}) are the hyperspherical Cartesian coordinates of group g.
K_g = ‖G‖ + 1 = λ_{g+1}/λ_g. R = A_{d−2}⋯A₂A₁, where A_g = Lo(θ_g)·P₂₃(φ_{g,g−1})⋯P_{g,g+1}(φ_{g,1})
acts on tangent coordinates 1..g+1 (`Basic_Final.R:121–156`). The last column of A_g is the group's
direction written backwards with signs, so v_{g+1} (eigenvalue λ_{g+1}) = (outer factors)·(that
direction), and the other columns of A_g fix the frame in which the lower eigenvectors live.

### 1.1 Continuity: only prefix zero patterns are safe

Base γ generic, d = 6, μ = (2,−5,3,−4,5,6). The listed groups are driven to 0 from 6 random
directions; the table reports max|V(u_s) − V(u₁)| (`disc_check5.R`):

| zero groups | ε = 1e−3 | 1e−6 | 1e−9 | verdict |
|---|---|---|---|---|
| {1} | 2.4e−4 | 2.4e−7 | 2.4e−10 | continuous |
| {1,2} | 5.5e−4 | 5.5e−7 | 5.5e−10 | continuous |
| {1,2,3} | 1.2e−3 | 1.2e−6 | 1.2e−9 | continuous |
| {1,2,3,4} (γ = 0) | 2.0e−3 | 2.0e−6 | 2.0e−9 | continuous |
| {2} / {3} / {4} | 0.20 / 0.47 / 1.16 | same | same | **discontinuous** |
| {1,3} / {2,3} / {1,2,4} | 0.57 / 0.26 / 0.42 | same | same | **discontinuous** |
| {2,3,4} (the exp-2 pattern) | 0.36 | 0.36 | 0.36 | **discontinuous** |

**Rule.** A zero group g is a continuity point iff the block it rotates is isotropic, i.e. iff all
groups below it are zero. A vanishing trailing pair (G_g, G_{g+1}) of a *nonzero* group g ≥ 2,
where θ_g = atan2(0,0) is set to 0 (`Basic_Final.R:95`), is a discontinuity whenever a lower group
is nonzero. The exceptions form a measure-zero set where the lower block happens to be isotropic in
the plane Lo(θ_g) rotates. With independent approach angles, d = 5 gives jumps of 0.19–0.55 that
stay flat in ε, but with group 1 = 0 they shrink as ε→0 when it is group 2's trailing pair
(`disc_check2.R`). `sing_check.R` gives 0.27 and 0.81 for groups 2 and 3 at d = 5.

**Simulation truths.** Exp-1 γ₍₄₎ = (0,3,2,0,0) has a vanishing trailing pair: perturbing it by
ε = 1e−9 changes V by 4.64. In exp 2, groups 2–4 are zero above γ₍₁₎ = (5,3): perturbing them by
1e−9 changes V by 0.8–1.05. V at both truths is defined only by the atan2(0,0)/acos conventions.

**The frame U(μ) = ONB(μ) is also discontinuous** (`Basic_Final.R:29–46`), on
{μ₁ = μ₂ = 0} and nested sets: approaching (0,0,3,−4,5,6) changes V by 4.77 (`disc_check.R`, test 6).
This matters in regression, where μ_i moves.

### 1.2 Non-differentiability at every zero group, γ = 0 included

At a zero group the directional derivative D_uV is positively homogeneous but **not linear** in u.
Linearity defects (`disc_check5.R`): 0.18 for zero set {1}, 0.29 for {1,2}, 0.44 for γ = 0,
against 0 at a generic point. At d = 3, V − I ≈ (r/2)R(θ)diag(−1,1)R(θ)ᵀ, which depends on 2θ: a
cone. So the likelihood has a cone-type kink at every sparse truth, and the score with respect
to γ does not exist there. Consequences:

* (A5)(iv) fails at any truth with a zero group;
* gradient-based KKT conditions are invalid in γ;
* a prox-gradient that uses the crate's guarded gradient (`sqrt'(0) := 0`) can never move a group
  away from zero.

### 1.3 Representability: zero groups are frame constraints

γ₍g₎ = 0 implies ξ_g = ξ_{g+1}, as the paper says, but **the converse fails** for g ≥ 2 unless
groups 1..g−1 are 0. {γ₍g₎ = 0} has codimension g+1, while {ξ_g = ξ_{g+1}} has codimension 2:
the extra g−1 equations pin eigenvectors to the arbitrary ONB(μ) frame.

Example at d = 4 (`disc_check4.R`): take a uniaxial V_t with λ₁ < λ₂ = λ₃ and eigenvector w.

* If w ∈ span(u₁,u₂), it is exactly representable, with γ₍₂₎ = 0 (error 4e−16).
* If w = (1,1,1)/√3, no γ with γ₍₂₎ = 0 represents it (best error 0.354). Error 1.1·ε is reached
  with γ₍₂₎ = ε·u, but never attained: the MLE runs into the singular set.

Prefix zeros (groups 1..m) are exactly ξ₁ = … = ξ_{m+1}: the dimensions match,
Σ_{i≤m}(i+1) = (m+1)(m+2)/2 − 1 = the codimension of an (m+1)-fold eigenvalue.

### 1.4 Coordinate zeros depend on the frame and on the coordinate order

* **Meaning of a single zero.** A single coordinate zero is a right-angle or zero-angle condition
  on one Givens angle. For the top group (j = d−2), the coordinates are, reversed and with signs,
  the components of the leading eigenvector v_{d−1} in the ONB(μ) frame: G₁ = 0 ⇔ v_{d−1} ⊥ u_{d−1}(μ),
  …, G_{d−1} = 0 ⇔ v_{d−1} ⊥ u₁(μ). The u_k(μ) are μ-weighted Helmert-type contrasts fixed by the
  order of the d coordinates. Lower groups refer to a frame already rotated by the higher groups.
* **Order dependence, demonstrated.** Relabel the coordinates of the exp-1 truth by 7 generic
  permutations (μ' = Pμ, V' = PVPᵀ). For each, all 16 sign-equivalent γ representatives
  reproduce V' to 1e−13, and **none has a zero** (min |γ_k| = 0.05–0.61) (`disc_check4.R`, which
  uses an R port of the crate's `angles_to_gamma`).
* **Sign flips are harmless.** Sign-flip equivalences preserve zero patterns and all |γ_k|, so
  sign canonicalization does not affect weights.

### 1.5 Multimodality and the oracle start

From neutral starts, the unpenalized likelihood reaches 544.72 against 544.52 at the R code's
truth + 0.1 start, with V differing by 0.2 and γ coordinates by up to 14.6 (SPEC §3). In that
better fit, the truth's γ₉ = 5 sits near 0. Elementwise adaptive weights and zero patterns are
therefore start-dependent; the paper's simulations used an oracle start.

### 1.6 Consequences for Section 4 of the paper

* **Props 2, 4, 5** assume f is C² near ω₀ (A5(iv)). That fails at both simulation truths
  (discontinuity) and at every zero-group truth (non-differentiability). The propositions do
  hold in a smooth chart of (μ, V), e.g. S (§2), where they are standard.
* **Prop 3 (γ₀ = 0)** is a continuity point, and its conclusion is plausible. The proof must use
  the conic expansion: γ̂ = 0 is a local optimum iff λ_n w_n ≥ sup_{‖u‖=1} D_uL_n(μ̂, 0), and that
  sup is O_p(n^{−1/2}). So λ_n n^{s·RoC + 1/2} → ∞ (A6(ii)) suffices for P(γ̂ = 0) → 1. Normality
  of μ̂ is unaffected.
* **(A2)/(A6)** applied coordinatewise to γ̃ are doubtful at sparse truths. Only V̂ and the group
  norms ‖γ̃₍j₎‖ = ξ̂_{j+1}/ξ̂_j − 1 are consistent; the coordinates depend on the representative
  reached (§1.4–1.5). Group weights are fine; elementwise weights are not.
* **(A4)**: "lim P(RoC ≥ 1) = 1" is false for a √n-consistent μ̃ unless RoC is floored. Without
  the floor, P(‖μ̃‖ ≥ ‖μ₀‖) → 1/2, and "μ₀ ≠ 0 and Ω compact" only gives the upper bound M.
  [CODE] floors at 1 (`max(·,1)`); state that in the paper.
* **RoC is asymptotically 1**: n^{(RoC−1)s} → 1, so RoC-(G)LASSO and a(G)LASSO with r = 1 have
  the same asymptotics. The proposed advantage is purely finite-sample.
* **The fixed-λ simulation design** (λ_n = λ/n, λ fixed) satisfies neither (A6)(i)
  (λ n^{s−1} → ∞) nor (A6)(ii) (λ n^{s−1/2} → ∞ fails at s = 1/2). So no fixed-λ curve in
  Figs 1–2 is an oracle sequence. A BIC-selected λ, which grows roughly like log n, is.

### 1.7 Which penalties are well-posed under the paper's parameterization

Well-posed here means: zero set at continuity points of γ ↦ V, frame-free meaning, and the sparse
submodel equal to the eigen-structure it claims.

| penalty | zero sets it produces | verdict |
|---|---|---|
| LASSO / aLASSO / RoC-LASSO (3.2) | arbitrary coordinates; includes trailing pairs and non-prefix whole groups | **not well-posed**: discontinuities + frame/order dependence (§1.1, §1.4). Keep for reproduction only |
| GLASSO / aGLASSO / RoC-GLASSO (3.3) | any subset of groups | **not well-posed as stated**: non-prefix zero groups are discontinuities and frame-dependent submodels; the exp-2 truth is one |
| GLASSO restricted to prefix patterns (nested penalty §2.6) | {1..m} | **well-posed**: continuous, frame-free (ξ₁ = … = ξ_{m+1}); still conic at zeros; only ties at the bottom of the spectrum |
| (4.4) single group ‖γ‖ | γ = 0 | **well-posed** (continuous, frame-free; conic KKT at 0) |

This is broader than "only γ = 0, or possibly γ₍₁₎ alone": every prefix pattern is admissible,
and γ₍₁₎ alone is the m = 1 case. Nesting *within* a group's tail (a trailing pair may vanish only
if the whole tail does) does not help, because any zero tail of length ≥ 2 contains the vanishing
trailing pair. A "safe elementwise" patch exists: give the trailing pair of each group g ≥ 2 the
non-convex penalty λ·min(w_a|a|, w_b|b|), which can zero at most one of the two, and forbid zero
groups for g ≥ 2. It restores continuity but not frame-invariance, so I do not recommend
elementwise sparsity in γ at all.

---------------------------------------------------------------------------------------------

## 2. Recommended design: penalize the log-tangent shape S [DEVIATION]

### 2.1 Parameterization (the kernel being implemented in Rust)

V(μ, s) = μ̂μ̂ᵀ + Ξ(μ̂) exp(S) Ξ(μ̂)ᵀ, where μ̂ = μ/‖μ‖, S = S(s) ∈ Sym₀(k), k = d−1, and s ∈
R^{(d−2)(d+1)/2} are Frobenius-orthonormal coordinates (d−2 Helmert diagonal coordinates, then
√2-scaled off-diagonals; ‖s‖₂ = ‖S‖_F).

* **Frame.** Ξ(μ̂) = Q(m→μ̂)E_m, with Q = I + K + K²/(1 + mᵀμ̂), K = μ̂mᵀ − mμ̂ᵀ (the minimal rotation
  taking m to μ̂) and E_m an orthonormal basis of m⊥. Ξ is analytic except at μ̂ = −m.
* **ESAG constraints are automatic**: Vμ̂ = μ̂, and det V = e^{tr S} = 1.
* **Smoothness and uniqueness.** V is C^∞ in (μ, s) except at μ = 0 and μ̂ = −m. s is unique
  given the frame: there are no equivalence classes, no canonicalization, and no singular sets
  in s. Isotropy ⇔ s = 0. The eigenvalues of V_t are exp(eig S), independent of the frame.
* **Choice of m.** For iid fits, any m gives the same fitted V (changing m conjugates S by an
  orthogonal matrix), so P_iso and P_gap fits do not depend on m. In regression m is a modelling
  choice, like a link function. Default: m = the mean resultant direction of Y. For compositional
  data, m = 1/√d with E_m = the Helmert (= standard ilr) basis, which makes S entries
  log-covariances of balances.
* **Topology.** A global continuous frame of μ⊥ exists only for d ∈ {2, 4, 8} (S^{d−1}
  parallelizable: complex numbers, quaternions, octonions). This is the same obstruction as
  SO(i+1) → S^i having a section only for i ∈ {1, 3, 7}. For d = 8 the octonion left
  multiplication L_μ̂ gives a polynomial global frame, as an option. Otherwise the anchored frame
  confines the unavoidable singularity to the single point −m, far from the data.
* **Why γ is discontinuous and S is not.** γ factors V through "eigenvalue gap + eigenvector
  direction". The direction→frame map is a would-be section, discontinuous for groups i ∉ {1,3,7}.
  A zero gap leaves the direction undefined while it still rotates a non-isotropic block (§1.1).
  S never factors through eigen-coordinates.
* **Link to the paper.** The paper's group norm is spectral: ‖γ₍j₎‖ = ξ_{j+1}/ξ_j − 1 =
  e^{g_j} − 1 with g_j = σ_(j+1) − σ_(j), the gaps of the sorted eigenvalues of S. So the
  (3.3)-family objective is a function of V alone. What breaks is only its γ-representation
  (discontinuity, non-attainment). P_gap below is that objective on the closure, with the
  linearization e^g − 1 → g.

### 2.2 Penalties and their exact prox (step t, level λ)

| name | P(s) | analog of | prox |
|---|---|---|---|
| P_iso | w‖s‖₂ | (4.4) | block soft-threshold: s ← (1 − tλw/‖s‖)₊ s |
| P_gap | Σ_{j=1}^{k−1} w_j (σ_(j+1) − σ_(j)) | GLASSO / aGLASSO / RoC-GLASSO | eigen + PAVA, below |
| P_ℓ1 | Σ_{a<b} w_ab \|s_ab\| (off-diagonal coordinates only) | LASSO family (frame-dependent) | elementwise soft-threshold |
| P_shape (regression) | Σ_k w_k ‖B_s[:,k]‖₂ | [EXTENSION] | block soft-threshold per driver column |
| P_mean (regression) | Σ_k w_k ‖B_μ[:,k]‖₂ | [EXTENSION] | block soft-threshold per driver column |

**P_gap prox.**

```
prox_gap(Z, τ = tλ, w):
  Z = Q diag(ζ) Qᵀ, ζ ascending
  c₁ = −w₁;  c_j = w_{j−1} − w_j (2 ≤ j ≤ k−1);  c_k = w_{k−1}    // Σc = 0, so the trace is kept
  x = PAVA_nondecreasing(ζ − τc)                                  // isotonic regression, equal weights, O(k)
  return Q diag(x) Qᵀ
```

* **Why it is exact for any weights.** P_gap is orthogonally invariant, so the prox acts on
  eigenvalues with the eigenvectors fixed. The penalty depends only on the sorted multiset, so an
  optimal x has the order of ζ (rearrangement inequality). On that cone the penalty is cᵀx, so the
  problem is isotonic regression of ζ − τc. Checked against brute force in 30 random trials,
  including non-concave w: never beaten, trace preserved to 1e−14 (`prox_check.R`).
* **Convexity and tie behaviour.**
  * P_gap is convex iff c is nondecreasing, i.e. w is concave in j (nonnegative combination of
    Ky Fan sums).
  * Uniform w gives P = σ_max − σ_min, which forms ties only from the extremes inward.
  * w_j = j(k−j) gives the pairwise Σ_{a<b}|σ_a − σ_b| (convex; ties anywhere).
  * Adaptive w_j = g̃_j^{−r} is non-convex but has the same exact prox. An interior tie j is
    "sticky" iff w_j > (w_{j−1} + w_{j+1})/2, which adaptive weights deliver for small initial
    gaps.
  * Use the linearized gaps g. The exact paper value Σw(e^g − 1) needs a 1-D nonlinear prox and
    is not worth it.
* **Meaning.** A tie σ_(j) = σ_(j+1) ⇔ ξ_j = ξ_{j+1}: partial isotropy at any position in the
  spectrum, frame-free. This is what the paper meant γ₍j₎ = 0 to express.

**P_ℓ1** is acceptable only when E_m is declared in advance and scientifically meaningful (ilr
balances, a physical basis), and the coordinate order is fixed. Never put it on the Helmert
*diagonal* coordinates: those are contrasts of log-variances with no sparsity meaning. Label it
exploratory.

### 2.3 Weights and RoC, case by case

* **Initial estimator.** The unpenalized MLE in S, unique, computed by the crate's multi-start
  (moment start S₀ = log of the normalized tangent scatter, plus the R-style start). For
  regression with many parameters, if the MLE is unstable, use a ridge-penalized fit (small ridge
  on the slopes; Zou 2006 §3).
* **Base weights** (a weight of ∞ means the coordinate or group is fixed at 0; cap finite weights
  at 1e10):

  | penalty | base weight |
  |---|---|
  | P_iso | b = ‖s̃‖₂^{−1} |
  | P_gap | b_j = g̃_j^{−1}, with g̃ the gaps of the sorted eigenvalues of S̃ |
  | P_ℓ1 | b_ab = \|s̃_ab\|^{−1} |
  | P_shape | b_k = ‖B̃_s[:,k]‖^{−1} |
  | P_mean | b_k = ‖B̃_μ[:,k]‖^{−1} |

* **Exponent r.**
  * "Lasso/GLASSO": r = 0 (all weights 1; for P_gap use the concave w_j = j(k−j)/k as the r = 0
    baseline).
  * "a-": r = 1.
  * "RoC-": r(μ) = max(1, ‖μ̃‖/‖μ‖) [PAPER+CODE floor]; weight w = b^{r(μ)}.
  * In regression, r(B) = max(1, (1/n)Σ_i ‖μ̃_i‖/‖μ_i(B)‖), the same mean-of-ratios as the crate's
    RoC test statistic (MATH.md §6).
* **Mechanism [PAPER claim corrected].** ∂w/∂r = −w ln|s̃|. Raising r lowers the penalty only on
  coordinates or groups with initial size > 1 and *raises* it on those < 1. It sharpens the
  weights rather than uniformly "dialling down", and the effect depends on the units of the
  parameter. The paper's truths have all nonzero |γ| ≥ 1, which is why it looks like a dial-down
  there.
* **Joint vs fixed point.** In the joint objective [PAPER], r depends on the argument μ, so the
  penalty has a μ-gradient. For elementwise weights (groups are analogous):

  ∇_μ[λΣ_k b_k^{r(μ)}|s_k|] = λ·r·(Σ_k |s_k| w_k ln|s̃_k|)·μ/‖μ‖²  (r > 1; 0 on the floor).

  This is the task's |γ|·w·ln|γ̃|·RoC·μ/‖μ‖². At the fixture optima Σ|s_k|w_k ln|s̃_k| = 5.6, 4.1,
  1.0 > 0 (RoC-LASSO, λ = 1, 16, 128), so the penalty pushes ‖μ̂‖ *down*. Observed: RoC-GLASSO
  ‖μ̂‖ = 9.80 vs aGLASSO 10.72 at λ = 16, and 6.63 vs 8.51 at λ = 128 (‖μ̃‖ = 10.82). That biases
  exactly the concentration RoC is meant to protect.
  * **Recommendation**: implement both. Make **fixed point** the default (§3.4) and "joint" the
    paper-faithful option (§3.3).
  * A uniform dial-down variant, λ_eff = λ·r(μ)^{−κ} with fixed b (κ = 1), is a further option
    [DEVIATION].

### 2.4 Mapping the paper's six methods onto the corrected design

| paper | corrected (S) | weights |
|---|---|---|
| LASSO / aLASSO / RoC-LASSO | no frame-free analog. P_gap for eigen-structure sparsity; P_ℓ1 (declared basis) if coordinate sparsity is really wanted | 1 / \|s̃\|^{−1} / \|s̃\|^{−r(μ)} |
| GLASSO / aGLASSO / RoC-GLASSO | P_gap | j(k−j)/k / g̃^{−1} / g̃^{−r(μ)} |
| (4.4) (+ a/RoC) | P_iso | 1 / ‖s̃‖^{−1} / ‖s̃‖^{−r(μ)} |

### 2.5 Theory in S [EXTENSION, standard]

f is C^∞, and s₀ is an interior point for every truth. Adaptive-lasso and adaptive-group-lasso
likelihood theory then applies directly (Zou 2006; Zhang & Lu 2007; Wang & Leng 2008). With
λ (sum scale) → ∞, λ/√n → 0 and √n-consistent initial weights with r > 0, the estimator is
selection-consistent and oracle-normal, and BIC tuning is consistent (Wang, Li & Leng 2009). For
P_gap the selected model is a tie-pattern manifold, the analog of adaptive fused lasso for change
points (Rinaldo 2009). The proof transfers but should be written out; I have not done so.

### 2.6 If you must stay in γ: nested-prefix group penalty [DEVIATION]

P_pre(γ) = Σ_{m=1}^{d−2} w_m ‖(γ₍₁₎, …, γ₍m₎)‖₂, with w_m = ‖(γ̃₍₁₎,…,γ̃₍m₎)‖^{−r}, optionally ×√(dim).
Its zero sets are exactly the prefixes. The groups form a chain, so the prox is the composition of
block soft-thresholds from the smallest group to the largest (Jenatton et al. 2011):

```
for m = 1..d−2:  v[H_m] ← (1 − tλw_m/‖v[H_m]‖)₊ v[H_m]   // H_m = first Σ_{i≤m}(i+1) coordinates
```

Verified against numerical minimization to 1.8e−15 (`prox_check.R`). Limits:

* it captures only ties at the bottom of the spectrum;
* zeros remain conic, so the KKT check must be directional (§3.6);
* elementwise and non-prefix group patterns remain unavailable.

---------------------------------------------------------------------------------------------

## 3. Optimization

### 3.1 Objective and scale

Minimize **F(θ) = −Σ_{i=1}^n ℓ_i(θ) + λ·P(θ)**. This is the sum scale of the crate and of the
R code, with λ in the R code's units, so λ = 0.5…256 means λ_n = λ/n in (3.2)/(3.3). The paper's
Q_n is −F/n. The API should take λ on the sum scale and report λ_n = λ/n. Internally, using F/n
improves conditioning without changing the minimizers. θ = (μ, s) in S mode, or (B_μ, B_s) in
regression. μ and the intercept column are never penalized.

### 3.2 Algorithm A: FISTA with backtracking and restart (any P with an exact prox)

```
input: x0 (warm start), λ, weights w, prox_P, f = −Σℓ, ∇f (crate adjoint; for S use the
       Daleckii–Krein divided differences of exp(−S))
L ← L0   (secant estimate ‖∇f(x0+δ) − ∇f(x0)‖/‖δ‖, δ = 1e−4·(1+|x0|)·random unit; or the last L on the path)
x ← x0; z ← x0; t ← 1; F_old ← F(x0)
for k = 1..20000:
    g ← ∇f(z); fz ← f(z)
    repeat                                                    // backtracking, η = 2
        x⁺ ← prox_{(λ/L)P}(z − g/L);  Δ ← x⁺ − z
        if f(x⁺) ≤ fz + gᵀΔ + (L/2)‖Δ‖² + 1e−12·|fz|: break
        L ← 2L
    G ← L·(z − x⁺)                                            // gradient mapping at z (before z changes)
    if F(x⁺) > F(x):                                          // function-value restart (O'Donoghue–Candès)
        t ← 1; z ← x; continue
    t⁺ ← (1 + √(1 + 4t²))/2;  z ← x⁺ + ((t − 1)/t⁺)(x⁺ − x);  x ← x⁺;  t ← t⁺;  L ← 0.9L
    stop if ‖G‖/n ≤ 1e−8 and |F(x) − F_old| ≤ 1e−12·(1+|F(x)|) for 3 consecutive iterations
    F_old ← F(x)
polish: once the support (zero coordinates / zero groups / tie pattern) is unchanged for 25 iterations,
        run the crate's L-BFGS on the smooth restricted problem (free coordinates only; for P_ℓ1 the
        penalty λΣw_k sign(x_k)x_k with signs frozen; for groups λw‖x_G‖, smooth off 0; for P_gap,
        projected L-BFGS onto the tie pattern = average the tied sorted eigenvalues); accept if the
        KKT certificate (§3.5) holds and F did not increase, else resume FISTA
```

Cost: one gradient per accepted step, about 1–3 likelihood evaluations per backtrack. At
d ≤ 8 and n ≤ 10⁴ a full path is milliseconds to seconds.

Alternatives considered:

* **OWL-QN** (orthant-wise L-BFGS) is good for P_ℓ1 only.
* **Proximal quasi-Newton** (variable-metric prox) is faster asymptotically, but its scaled prox
  is not closed-form for P_gap.
* **Smoothing** (√(x² + ε²)) needs a post-hoc threshold, which is the R approach and is what
  causes D10/D11.

Use FISTA + polish.

### 3.3 Algorithm B: RoC, joint objective [PAPER-faithful]

F(μ, s) = f(μ, s) + λΣ_k b_k^{r(μ)}|s_k| (groups and gaps are analogous). The nonsmooth part is a
product of a C¹ positive function of μ and a convex function of s.

```
repeat (outer, ≤ 200):
  μ-block:  ≤ 50 L-BFGS iterations on  h(μ) = f(μ, s) + λ Σ_k b_k^{r(μ)} |s_k|
            ∇h = ∇_μ f + λ r (Σ_k |s_k| w_k ln(1/b_k)) μ/‖μ‖²   (second term 0 if ‖μ‖ ≥ ‖μ̃‖ (floor))
  s-block:  ≤ 50 iterations of Algorithm A in s, with weights w_k = b_k^{r(μ)} frozen
until |ΔF| ≤ 1e−12(1+|F|) and both block certificates hold
```

**Joint KKT (Clarke) conditions**, valid because b^{r(μ)} > 0 is C¹ off the floor:

* ∇_μ f + λΣ|ŝ_k|∇_μw_k(μ̂) = 0;
* for ŝ_k ≠ 0: ∂_k f + λw_k sign(ŝ_k) = 0;
* for ŝ_k = 0: |∂_k f| ≤ λw_k.

F decreases monotonically and has bounded level sets (ℓ → −∞ as ‖μ‖ or ‖s‖ → ∞). Limit points are
block-stationary, which here equals stationary, by the standard BCD-with-prox-step argument
(Tseng 2001; Xu & Yin 2013).

### 3.4 Algorithm C: RoC fixed point [DEVIATION, recommended default]

```
r ← 1 (or r(μ̃))
repeat (≤ 50):
   (μ̂, ŝ) ← Algorithm A on f + λΣ b_k^r |s_k|     (μ unpenalized; warm start)
   r_new ← max(1, ‖μ̃‖/‖μ̂‖)
   stop if |r_new − r| ≤ 1e−8;  r ← r_new
```

The solution is stationary for the weighted problem with weights b^{r(μ̂)}: the paper's feedback
narrative, without the incentive to shrink ‖μ̂‖. If the map oscillates (not seen in practice), damp
with r ← (r + r_new)/2.

### 3.5 Exact zeros, infinite weights, KKT certificate

* **Exact zeros** come from the prox; never threshold [DEVIATION from CODE's 0.01 rule].
* **Infinite weights.** s̃_k = 0 (or g̃_j = 0) gives w = ∞: fix the coordinate (or tie) and drop
  it from the free set. For an exact-zero γ̃ the R simulation scripts have no guard (1/0 = Inf);
  `Basic_Final.R:669` uses w = 1e7.
* **Certificate.** Report max violation over:
  * smooth coordinates: |∂_k f|;
  * nonzero penalized coordinates: |∂_k f + λw_k sign|;
  * zero coordinates: (|∂_k f| − λw_k)₊;
  * groups: ‖∇_G f + λw_G x_G/‖x_G‖‖ if nonzero, (‖∇_G f‖ − λw_G)₊ if zero;
  * P_gap: the gradient-mapping norm ‖x − prox_{τP}(x − τ∇f)‖/τ.

  Default tolerance 1e−6·√n on the sum scale.

### 3.6 Paper-reproduction mode in γ (the six methods as coded)

The prox-gradient cannot be used naively in γ: ℓ is non-differentiable exactly at the zeros, and
the guarded gradient traps groups at 0. For reproduction:

```
for ε in (1e−2, 1e−4, 1e−6):          // smoothing continuation
    minimize −Σℓ + λ·P_ε with crate L-BFGS (exact gradients), warm-started
    where |x| → √(x²+ε²), ‖x‖ → √(‖x‖²+ε²); RoC r(μ) inside as coded (floor 1)
starts: (a) the canonicalized unpenalized MLE (default), (b) user-supplied (for exact R reproduction: truth + 0.1)
report: the untruncated optimum and its UNSMOOTHED objective F (comparable to R's optim value),
        then R's rule ifelse(|x| < 0.01, 0, x) and R's BIC/AIC/SBIC exactly as coded (SPEC §2)
flags: any zero group above a nonzero group; any vanishing trailing pair of a nonzero group above
       nonzero lower groups; loglik drop between untruncated and thresholded > 1 (discontinuity hit)
```

Acceptance test: F_Rust ≥ F_R − 1e−6|F_R| on every fixture fit. R's points are not optima (SPEC §2:
Nelder–Mead improves them by up to +8.3), so a Rust optimum may be strictly higher. If
exact-zero solutions are wanted in γ, use the nested-prefix penalty of §2.6 with a directional
check at zero blocks. Test re-entry by evaluating D_uF over 2·dim random unit directions u plus
the last nonzero direction; re-enter if any D_uF < 0.

### 3.7 λ path and warm starts

* **Grid.** λ_j = λ_max·ρ^j, j = 0..49, ρ = (1e−4)^{1/49} (sum scale), plus the unpenalized fit
  (λ = 0). Paper grid option: {0, 0.5, 1, 2, …, 256}.
* **λ_max** at the null fit (μ̂₀ = isotropic MLE, s = 0, the crate's isotropic mask), all in
  closed form in S because f is smooth there:
  * P_iso: λ_max = ‖∇_s Σℓ(μ̂₀, 0)‖₂ / w.
  * P_shape/P_mean: λ_max = max_k ‖∇_{B[:,k]}Σℓ‖ / w_k at the intercept-only fit.
  * P_ℓ1: λ_max = max |∂Σℓ/∂s_ab| / w_ab.
  * P_gap: the smallest λ such that eig(∇_S Σℓ(μ̂₀, 0)/λ) is majorized by c (bisection on λ; exact
    for convex weights, a local test otherwise).
* **Warm starts** go downward from λ_max. For non-convex weights (adaptive P_gap, RoC), also sweep
  upward from the λ = 0 fit and keep the better F at each λ.

---------------------------------------------------------------------------------------------

## 4. Identifiability and reproducibility

* **S mode.** s is unique; nothing to canonicalize.
* **γ mode (reproduction).**
  * Compute γ̃ with the crate's best-of-starts unpenalized fit; record the start used.
  * Sign canonicalization (`param::gamma_from_covariance`, `canonicalize_coef`) does not change any
    |γ̃_k| or group norm, so it does not change the weights. Use it for reproducible reporting.
  * Do not trust the inverse map at singular sets: at the exp-1 truth the R port of
    `angles_to_gamma` returns a γ whose V is off by O(1), because θ_g cannot be recovered when the
    peeled column has a vanishing trailing pair. Fall back to the fit's own γ when any group norm
    or trailing-pair norm is < 1e−6 relative.
  * Start the penalized fits from γ̃, not from truth + 0.1 [DEVIATION from CODE]. Offer a
    user-start option for exact replication.
* **What zeros mean.**
  * γ: a prefix ⇔ ξ₁ = … = ξ_{m+1}; a non-prefix zero group ⇒ ξ_g = ξ_{g+1} plus frame constraints;
    a single coordinate = a Givens-angle condition relative to ONB(μ) (§1.4).
  * S: s = 0 ⇔ isotropy; a tie ⇔ equal eigenvalues; an off-diagonal 0 ⇔ zero log-covariance between
    two declared basis directions.

---------------------------------------------------------------------------------------------

## 5. λ selection

**Default: BIC = −2ℓ(θ̂_relaxed) + df·log n**, computed at the **relaxed** refit (the unpenalized
MLE restricted to the selected model) [DEVIATION: CODE uses the thresholded penalized estimate,
and thresholding can cross a discontinuity, SPEC D10].

df = d (μ) + df_model, where:

| model | df_model |
|---|---|
| P_iso | 0 if s = 0, else (d−2)(d+1)/2 |
| P_gap, tie pattern with K distinct eigenvalue values of multiplicities m₁..m_K (k = d−1) | (K − 1) + k(k−1)/2 − Σ_i m_i(m_i−1)/2 |
| γ-prefix with groups 1..m zero | Σ_{j>m}(j+1) (equals the P_gap value for a bottom tie) |
| regression | #coefficients in active groups (intercept columns always counted) |
| reproduction | R's rule, #{≠0 after the 0.01 threshold}, μ included |

The P_gap formula gives (d−2)(d+1)/2 when all eigenvalues are distinct and 0 when all are equal.
The exp-2 pattern (1, 4) gives 5, not the 2 that counting γ coordinates gives.

* **Ties in BIC across λ**: R takes `which.min` (smallest λ). Recommend the largest λ among
  equal-BIC models (sparser) [DEVIATION, minor].
* **AIC**: same df, offered for prediction.
* **CV [DEVIATION]**: K = 10 (or 5) **shuffled, seeded** folds (R: contiguous, unseeded), criterion
  **held-out log-likelihood** (proper scoring rule), optional 1-SE rule, then refit on all data at
  λ* (penalized + relaxed).
* **RMSE (5.5)**: keep only for reproducing the paper's CV, for three reasons:
  1. it depends on μ̂ only through μ̂/‖μ̂‖ and the frame, so it cannot see the concentration loss
     RoC is about;
  2. it is not a proper scoring rule (it rewards V̂ aligned with residual scatter, given det = 1);
  3. in the authors' own runs it selected λ = 0 in 43% of exp-1, n = 500 replicates.

---------------------------------------------------------------------------------------------

## 6. Post-selection inference

* **Report two estimators**: the penalized θ̂(λ*) and the **relaxed** θ̂_R, the unpenalized MLE on
  the selected model.
  * P_iso: isotropic ESAG, μ only.
  * P_gap: projected gradient onto the tie pattern, averaging tied sorted eigenvalues (an exact
    Euclidean projection given the eigenvectors).
  * Groups: inactive groups fixed at 0.
* **Oracle SEs** (Props 4/5 analog, valid in S): the inverse observed information of the restricted
  model, from the crate's Hessian machinery. For a tie pattern, give SEs for frame-free quantities
  (μ, ‖μ‖, the distinct eigenvalues) by the delta method on a local chart. They are valid only if
  selection is consistent and ignore selection uncertainty.
* **Honest uncertainty**: a pairs (or parametric) bootstrap of the whole pipeline (λ by BIC,
  selection, relaxed refit; `inference::bootstrap_coef`-style), reporting percentile intervals and
  **selection frequencies** per group or tie.
* Never report Hessian SEs for γ coordinates at a sparse γ fit: the likelihood is not
  differentiable there.

---------------------------------------------------------------------------------------------

## 7. Regression extension [EXTENSION] (the user's d = 7–8, 5–6 drivers)

Model: μ_i = B_μ x̃_i, s_i = B_s x̃_i with x̃_i = (1, x_i). At d = 8 with p = 6 drivers, B_μ is 8×7
and B_s is 27×7: 245 parameters.

* **Parameterization.**
  * γ mode: the affine family {γ_i} meets the codimension-2 singular sets (§1.1) in 3–4-dimensional
    subsets of covariate space, and μ_i meets {μ_{i1} = μ_{i2} = 0}. Observations near them create
    cliffs: Rust regression fits at d = 7–8 stalled with |∇| ~ 1e6, pinned by one observation whose
    trailing pair was ~1e−12, and were multimodal.
  * S mode: smooth in B except at μ_i = 0 or μ̂_i = −m. Use S.
  * Gradient: S enters ℓ only through q = (μ̂ᵀy)² + zᵀe^{−S}z, z = Ξ(μ̂)ᵀy, and α = yᵀμ/√q.
    So ∂ℓ/∂S = −(1/(2q))·[d + α² + α·(log M_{d−1})'(α)] · Q(F ∘ Qᵀzzᵀ Q)Qᵀ, with S = Q diag(σ) Qᵀ,
    F_ab = (e^{−σ_a} − e^{−σ_b})/(σ_a − σ_b) and F_aa = −e^{−σ_a} (Daleckii–Krein). Project onto
    trace-zero and map to s-coordinates. Checked against finite differences to 9e−9
    (`grad_S_check.R`). The μ-part adds the terms through μ̂ in Ξ(μ̂) and in (μ̂ᵀy)², α and ‖μ‖².
    One (d−1)-eigendecomposition per observation.
* **Covariates**: penalize slopes of **standardized** covariates (mean 0, sd 1). Group norms depend
  on units; the papers' [1,2] range scaling is acceptable if used consistently. Never penalize the
  intercept column. Report on the original scale.
* **Penalties**:
  * P_shape = Σ_k w_k^s‖B_s[:,k]‖₂ (27-dim groups at d = 8): "does driver k change the shape?"
    Invariant to rotations of E_m.
  * P_mean = Σ_k w_k^μ‖B_μ[:,k]‖₂ (8-dim): "does driver k change the mean direction or
    concentration?"
  * Optional driver-level tree Σ_k [w_k‖(B_μ[:,k], B_s[:,k])‖ + w_k^μ‖B_μ[:,k]‖ + w_k^s‖B_s[:,k]‖]:
    nested groups, so the exact prox composes the inner groups first, then the driver group.
  * Elementwise L1 on B_s only with a declared basis. P_gap on the intercept shape only when no
    driver is active in shape.
  * The preliminary R code already grouped slopes this way: `preliminary_sims/Group_LASSO.Rmd:122–127`
    has one μ-slope group and one γ-slope group per covariate.
* **Weights**: from the unpenalized regression MLE if n ≫ 245; otherwise from a ridge fit. Default
  r = 1. RoC option r(B) = max(1, mean_i‖μ̃_i‖/‖μ_i(B)‖) with ∇_{B_μ}r = −(1/n)Σ_i ‖μ̃_i‖ μ_i x̃_iᵀ/‖μ_i‖³
  (off the floor), implemented by the fixed point (§3.4).
* **What's in the paper vs extension**: everything in this section is [EXTENSION]. The paper is
  iid only. The γ-mode version (per-driver ‖B_γ[:,k]‖, per-(j,k) blocks ‖B_γ[G_j,k]‖) is possible,
  but I recommend against it for the reasons above.

---------------------------------------------------------------------------------------------

## 8. Paper issues and concrete corrections (methodology; code bugs are in SPEC §10)

1. **Parameterization singularities** (§1). Restate Section 2: "γ₍j₎ = 0 ⇒ ξ_j = ξ_{j+1}; the
   converse holds for prefixes only; V is discontinuous at non-prefix zeros and non-differentiable
   at every zero group." Move the penalization (and Props 2–5) to the S chart, or restrict to
   prefix patterns.
2. **Simulation truths** sit on discontinuities. Choose truths whose zero pattern is a prefix
   (e.g. γ₍₁₎ = γ₍₂₎ = 0) for a GLASSO experiment, or define the truth by its eigen-structure
   (ties) and evaluate in V-space.
3. **Evaluation should be frame-free.** Score selection on eigenvalue-tie recovery (or on group
   norms) rather than coordinatewise CR/SE/SP, which change under relabeling (§1.4).
4. **(A4)/RoC**: put the floor max(1,·) into the definition. Note RoC → 1, so the oracle theory
   does not distinguish RoC- from a-methods. Correct the "dial down" sentence (§2.3) and disclose
   the μ-incentive of the joint objective, or switch to the fixed point.
5. **(A6) and the λ design**: either let λ grow with n in the simulations (e.g. λ ∝ log n, or
   BIC-selected), or say the fixed-λ curves are not oracle sequences.
6. **Group-size factor**: Yuan & Lin weight groups by √p_j. The groups here have sizes 2…d−1, so
   either include √(j+1) (r = 0 case) or justify omitting it.
7. **Metrics**: rename SE/SP (SPEC D5); call ‖μ̂−μ‖ an "estimation error", not a bias; describe
   (5.5) as an in-sample standardized residual (not "prediction"), and note it ignores ‖μ̂‖.
8. **Monte Carlo reporting**: plain means and SE = sd/√R (SPEC D3, D4); recompute Fig 2 against
   the right truth (SPEC D1).
9. **Application**: fix the taxa (Alistipes vs Bacteroides) and the transformation, or describe what
   was done (SPEC D6, D7); fill in the six-method comparison placeholder; add the relaxed refit
   and bootstrap selection frequencies.
10. **Prop 4** typo (E{Ψ} → E{H}); "r = ‖μ̃‖/μ" typo; empty Discussion.

---------------------------------------------------------------------------------------------

## 9. Rust API sketch and test plan

```
enum ShapeParam { Gamma, LogTangent { anchor: Anchor /* MeanResultant | Fixed(v) | Octonion(d=8) */, basis: Basis } }
enum Penalty   { Iso, Gap, L1Offdiag, Prefix /* γ only */, Shape, Mean, DriverTree,
                 PaperElementwise /* γ, reproduction */, PaperGroup /* γ, reproduction */ }
enum Adapt     { None, Fixed(f64), RoC { mode: Joint | FixedPoint } }   // r = max(1, ‖μ̃‖/‖μ‖)
struct PenalizedEsag { param, penalty, adapt, lambda_grid: Grid /* Auto{n:50,ratio:1e-4} | Paper | Custom */,
                       select: Bic | Aic | Cv { k, seed, score: HeldOutLogLik | Rmse }, relax: bool }
```

`fit` returns the path, with per-λ estimate, F, KKT residual, df, BIC/AIC and flags. The output
also carries the selected λ (sum scale and λ_n), the relaxed refit with SEs, and support flags.
`paper_method(PaperMethod::RocLasso, …)` presets the γ reproduction mode of §3.6.

Tests:

1. **Deterministic objective check**: for every fixture fit, evaluate −F at R's
   `par_untruncated` with the coded penalty (RoC floor included). It must equal
   `objective_sum_scale` to 1e−10 relative.
2. **Optimizer quality**: F_Rust ≥ F_R − 1e−6|F_R| (maximization sign) at λ ∈ {1, 16, 128} for all
   six methods.
3. **Application** (soft tests, because a better optimizer can move the λ = 1 optimum): the BIC
   path's argmin is λ = 1; μ̂ is within 0.05 of (2.50, 0.76, 0.42, 2.48, 1.65, 1.78); and γ̂ has
   zeros at γ₁, γ₅. Allow sign-flip equivalence.
4. **Prox unit tests**: PAVA prox against brute force; nested-prefix composition; block/elementwise
   soft-thresholding.
5. **S kernel**: continuity across the γ singular sets (V(s) smooth); det V = 1 and Vμ = μ; the
   gradient against finite differences.
6. **KKT certificate** < tolerance on all path points; the support is stable under 1e−8
   perturbations of the data weights.

---------------------------------------------------------------------------------------------

## 10. Evidence index (`tmp/p3/scripts/`)

| script | shows |
|---|---|
| `disc_check.R` | exp-1 trailing pair (4.64), exp-2 zero groups (1.05, 0.77), bottom-nested continuity, μ-frame jump (4.77), R conventions |
| `disc_check2.R` | d = 5 trailing pairs with independent angles (0.19–0.55); continuous when the lower block is isotropic |
| `disc_check3.R` | non-differentiability at γ = 0 (d = 3) |
| `disc_check4.R` + `inverse_map.R` | representability (0.354 vs 1.1ε); relabeling destroys all zeros (7 permutations × 16 representatives) |
| `disc_check5.R` | prefix vs non-prefix classification table; linearity defects at zeros |
| `../sing_check.R` (coordinator) | the same discontinuities with `ESAG_Project2.R` functions |
| `start_dependence.R` | multimodality; oracle start not the global maximum |
| `analyze_sim.R`, `polish.R` | R optima are path-dependent and improvable; thresholding drops (539.7 → 21.0) |
| `prox_check.R` | exactness of the eigen+PAVA prox and of the nested-prefix composition |
| `check_fig1.R`, `check_fig2*.R` | Fig 1 consistent; Fig 2 scored against the wrong truth |
