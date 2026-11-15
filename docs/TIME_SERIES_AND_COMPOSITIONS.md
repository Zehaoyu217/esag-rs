# Directions of time series and compositions

Can ESAG regression forecast the *direction* of a multivariate time series
(not its size), and how concentrated that direction is? How should the
series be prepared (levels, differences, standardized)? And how do
compositions fare on the sphere, with Yu & Huang's square-root map to the
positive orthant or with signed vectors that use the whole sphere?

Scripts: `python/examples/ts_directions.py`, `python/examples/compositional.py`,
shared tools in `python/examples/_pn.py` (the general projected normal
density, checked against the package's ESAG log-likelihood to 1e−14 and
against quadrature to 2e−12; Monte Carlo mean directions; scores). Every
forecast is one step ahead and out of sample: 2000 training steps, 1000 test
steps, 10 replicates (4 at d = 10), the γ-rot shape (Python's default). The
full output is in `docs/time_series_compositions_output.txt`.

## 1. Why the direction of a Gaussian vector is projected normal, and where ESAG fits

If `x_t | past ~ N(m_t, S)`, the direction `u_t = x_t/|x_t|` is projected
normal, `PN(m_t, S)`, with density on `S^{d−1}`

    f(u) = (2π)^{−d/2} |S|^{−1/2} A^{−d/2} exp(−½ mᵀS⁻¹m) I_{d−1}(uᵀS⁻¹m / √A),
    A = uᵀS⁻¹u,  I_k(α) = ∫_0^∞ s^k exp(−s²/2 + αs) ds.

Three consequences decide everything below.

1. **Scale invariance.** `PN(m, S) = PN(cm, c²S)`. A common volatility `s_t`
   leaves the axis of the distribution alone and changes only its
   concentration, through `|m_t| / s_t`. Directions are scale-free; their
   predictability is not.
2. **ESAG is the projected normal with `V μ = μ` and `det V = 1`**: the
   noise along the mean direction has unit variance and is uncorrelated with
   the rest. A VAR with isotropic noise has ESAG directions exactly
   (`μ = A x_{t−1}`). With anisotropic noise fixed in `R^d` and a mean that
   turns with the state, `S m_t ∝ m_t` fails for most `t`, and ESAG is an
   approximation.
3. **ESAG's shape lives in a frame carried by the mean direction**
   (`V = μ̂μ̂ᵀ + Ξ W Ξᵀ`, `Ξ` transported from a reference direction to `μ̂`).
   Constant shape coordinates mean axes that turn with `μ̂`. That suits data
   whose mean direction moves within a region (compositions in an orthant,
   paleomagnetic directions, wind with a prevailing direction). It does not
   suit a series whose mean direction roams the sphere or passes through
   zero (a mean-zero VAR, net flows that change sign): the frame is undefined
   at `μ = 0`, and the intercept-only model at the top of a penalized path is
   exactly such a point.

The exact model for VAR directions is the **projected normal regression with
a constant covariance**, `u_t ~ PN(B [1, x_{t−1}], S)`, `det S = 1` (the scale
is not identified): `_pn.PNRegression`, a prototype fitted by L-BFGS on
analytic gradients with the precision's Cholesky factor (log-diagonal summing
to zero). Its gradient agrees with finite differences to 1.5e−6; on an
anisotropic VAR at d = 5, n = 2000 it fits in 0.2 s from the directions alone
and recovers `A` and `S` up to the scale (largest errors 0.04 and 0.07). It
has `d(p + 1) + d(d + 1)/2 − 1` parameters (44 at d = 5 with 5 lags; ESAG
regression has 84).

## 2. Structuring the series

| series | target direction | covariates | why |
|---|---|---|---|
| stationary, mean 0 | `x_t/|x_t|` | `x_{t−1}` | `u_t ~ PN(A x_{t−1}, S)` |
| stationary, mean `c ≠ 0` | `(x_t − ĉ)/|x_t − ĉ|` | `x_{t−1} − ĉ` | otherwise the direction is mostly that of `c` |
| integrated (unit root) | `Δx_t/|Δx_t|` | `Δx_{t−1}` (error-correction terms if cointegrated) | the levels' direction tends to the drift's and moves by `O(1/√t)` |
| volatility clustering | as above; with coordinate-specific volatilities, of the devolatilized vector | `u_{t−1}` (or `x_{t−1}/σ̂_{t−1}`) | a common `s_t` cancels from the direction and sets the concentration `|m_t|/s_t`; per-coordinate scales would also bend the direction |
| composition levels | `√p_t` | ilr lags | Yu & Huang's map |
| composition changes | `Δilr_t/|Δilr_t|` | ilr lags | Aitchison geometry; losing parts point the negative way |
| signed compositions (net flows) | `f_t/|f_t|` | `f_{t−1}` | the whole sphere; Gaussian flows are exactly PN |

## 3. Directional time series

Scores: `cos`, the mean cosine between the forecast and the realized
direction (1 is perfect); `angle°`, the mean angle to the oracle's mean
direction; the mean log score and its gap to the oracle (nats per step);
for the concentration (mean resultant length ρ) its error against the
oracle's, their correlation (does the model know *when* the next direction
is predictable?), and calibration (mean ρ minus the realized mean cosine).
VAR-OLS is the Gaussian VAR fitted by least squares on the full vectors,
with its predictive projected normal: it sees the sizes, the directional
models do not.

**Stationary VAR(1), d = 5** (`A` with singular values .9, .8, .6, .4, .2):

| noise | model | cos | angle° | log-score gap | corr ρ |
|---|---|---|---|---|---|
| isotropic (ESAG exact) | oracle | 0.566 | 0 | 0 | 1 |
| | ESAG, `x_{t−1}` | 0.564 | 4.7 | 0.045 | 0.994 |
| | ESAG, `x_{t−1}`, auto | 0.564 | 4.7 | 0.023 | 0.994 |
| | ESAG, `u_{t−1}` only | 0.563 | 5.1 | 0.156 | 0.521 |
| | PN regression, `x_{t−1}` | 0.564 | 4.9 | 0.009 | 0.994 |
| | VAR-OLS | 0.564 | 4.7 | 0.007 | 0.995 |
| | persistence `u_{t−1}` | 0.085 | 81.6 | | |
| anisotropic (4, 2, 1, .5, .25) | oracle | 0.606 | 0 | 0 | 1 |
| | ESAG, `x_{t−1}` | 0.597 | 10.2 | 0.381 | 0.988 |
| | ESAG, `x_{t−1}`, auto | 0.596 | 10.3 | 0.388 | 0.989 |
| | PN regression, `x_{t−1}` | 0.605 | 4.3 | 0.009 | 0.996 |
| | VAR-OLS | 0.604 | 4.2 | 0.007 | 0.997 |

ESAG forecasts the direction almost as well as the oracle in both cases,
and knows when the next direction is predictable (ρ correlation 0.99). With
anisotropic noise its density is 0.38 nats per step worse: the constant
ambient covariance does not fit ESAG's mean-anchored shape (§1). The
projected normal regression, from the directions alone, is within 0.009 nats
of the oracle and within 0.002 of the Gaussian VAR that sees the full
vectors: the sizes add almost nothing to a direction forecast. With the
lagged direction only, the direction is still forecast well, but the
concentration is not (ρ correlation 0.52): the lag's size says how far the
state is from the origin, and so how predictable the next direction is.

**Stochastic volatility** (anisotropic VAR(1), `x_t = A x_{t−1} + s_t e_t`,
`log s_t` AR(1) with 0.97 and 0.2; the oracle knows `s_t`; `σ̂` is an EWMA,
0.94):

| model | cos | angle° | log-score gap | calib |
|---|---|---|---|---|
| oracle | 0.614 | 0 | 0 | −0.002 |
| ESAG, `x_{t−1}` | 0.545 | 24.6 | 1.183 | −0.135 |
| ESAG, `x_{t−1}/σ̂_{t−1}` | 0.589 | 15.4 | 0.686 | −0.066 |
| ESAG, `u_{t−1}` | 0.601 | 11.1 | 0.580 | 0.012 |
| PN regression, `x_{t−1}` | 0.601 | 9.5 | 0.822 | −0.205 |
| PN regression, `x_{t−1}/σ̂_{t−1}` | 0.611 | 5.7 | 0.381 | −0.075 |
| **PN regression, `u_{t−1}`** | **0.612** | **4.8** | **0.304** | **0.011** |
| VAR-OLS | 0.604 | 9.4 | 0.982 | −0.150 |
| VAR-OLS, standardized by `σ̂` | 0.611 | 5.7 | 0.405 | −0.074 |

Raw lags mislead every model: in a volatile stretch they are large, so
the models expect concentrated directions, while the noise is large too.
Scale-free covariates fix it, and the best forecaster is the directional
model on the lagged *direction*: it beats the volatility-standardized
Gaussian VAR in log score (0.30 against 0.41 nats from the oracle) and is
the only one whose concentration is calibrated. This is the case for
modeling directions instead of vectors: they do not care about volatility.

**Integrated series** (`Δx_t = δ + B Δx_{t−1} + e_t`, `|δ| = 0.3`,
`B = 0.5 × rotation`, isotropic noise):

| target | model | cos | angle° | log-score gap | corr ρ |
|---|---|---|---|---|---|
| levels `x_t/|x_t|` | oracle | 1.000 | 0 | 0 | 1 |
| | ESAG, `x_{t−1}`, `Δx_{t−1}` | 1.000 | 0.0 | 0.689 | 0.814 |
| | persistence | 1.000 | 0.1 | | |
| changes `Δx_t/|Δx_t|` | oracle | 0.463 | 0 | 0 | 1 |
| | ESAG, `Δx_{t−1}` | 0.462 | 5.8 | 0.043 | 0.986 |
| | ESAG, `Δx_{t−1}`, auto | 0.461 | 6.6 | 0.014 | 0.985 |
| | PN regression, `Δx_{t−1}` | 0.462 | 6.0 | 0.011 | 0.988 |
| | VAR-OLS on `Δx` | 0.462 | 5.7 | 0.010 | 0.989 |
| | ESAG, `u_{t−1}` only | 0.461 | 7.3 | 0.107 | 0.278 |
| | ESAG, `x_{t−1}` (levels as covariates) | 0.028 | 86.4 | 1.083 | −0.004 |
| | i.i.d. (the drift only) | 0.109 | 76.9 | 0.633 | |
| | persistence | −0.003 | 89.4 | | |

The direction of the levels is "forecast" perfectly by persistence (cosine
1.000): it converges to the drift's direction and moves by `O(1/√t)`, so it
says nothing about where the series goes next. The direction of the
*changes* is the meaningful target, and with lagged changes as covariates
ESAG and the PN regression reach the oracle. Using the levels as covariates
for it fails (cosine 0.03, and overconfident). Take differences.

**d = 10, sparse lags** (only the lags of x1..x4 matter, isotropic noise, so
ESAG is exact; n = 3000, 4 replicates):

| model | cos | angle° | log-score gap | corr ρ | time |
|---|---|---|---|---|---|
| oracle | 0.352 | 0 | 0 | 1 | |
| ESAG, `x_{t−1}` (594 parameters) | 0.349 | 11.4 | 0.416 | 0.987 | 0.7–1.0 s |
| ESAG, `x_{t−1}`, auto | 0.349 | 11.4 | 0.416 | 0.988 | 11–107 s |
| PN regression, `x_{t−1}` (164 parameters) | 0.350 | 10.5 | 0.020 | 0.990 | 0.1–0.2 s |
| VAR-OLS | 0.350 | 10.4 | 0.019 | 0.991 | |
| persistence | 0.042 | 83.5 | | | |

Here ESAG is the exact family, and still 0.42 nats per step from the oracle:
the series has mean zero, so the fitted mean direction covers the whole
sphere, including the antipode of the frame's reference, where the
transported frame is singular. `fit_regularized()` kept every lag, for the
mean and for the shape, in all four replicates: its path jumps from the
full model (λ = 0) to the empty one, whose mean vector is near zero (the
directions' mean has length 0.015), where ESAG's shape is not identified and
the first penalized solves did not converge (KKT residual 760). The PN
regression is within 0.02 nats of the oracle without any penalty.

## 4. Compositions

Truth: a logistic-normal VAR(1) in ilr coordinates (D = 5 parts, one of them
rare, 3%), the logistic normal being the true family.

**Levels** (point forecasts: Aitchison distance and the angle between
`√p̂` and `√p_t`; log score on the simplex, with sqrt-ESAG folded over the
2^D orthants and the Jacobian `2^{−(D−1)} ∏ p_i^{−1/2}`):

| model | Aitchison | Hellinger angle° | log score |
|---|---|---|---|
| oracle (logistic normal) | 0.563 | 6.52 | 8.780 |
| logistic-normal VAR (OLS on ilr) | 0.564 | 6.53 | 8.772 |
| sqrt-ESAG, ilr lag | 0.565 | 6.54 | 8.687 |
| sqrt-ESAG, `√p_{t−1}` lag | 0.566 | 6.56 | 8.667 |
| persistence | 0.884 | 9.56 | |

The square-root map loses nothing in the point forecasts against the true
family, and 0.085 nats per step in density.

**Zeros** (the compositions observed through multinomial counts of 40, 7.3%
zero parts; forecasts of the latent composition):

| model | Aitchison | Hellinger angle° |
|---|---|---|
| oracle (latent `z_{t−1}`) | 0.563 | 6.52 |
| logistic-normal VAR, +0.5 counts | 0.657 | 7.35 |
| sqrt-ESAG on the counts (zeros kept) | 0.728 | 7.57 |
| sqrt-ESAG, ilr(+0.5) lag | 0.728 | 7.58 |

Zeros alone do not make the square-root map win when the dynamics are
logistic normal: a pseudo-count logistic normal still recovers the latent
composition better. (Neither is a count model; a multinomial or
Dirichlet-multinomial observation layer would be the proper treatment.)

**Direction of compositional change** (`v_t = Δz_t/|Δz_t|` on `S³`, exactly
`PN((A − I)(z_{t−1} − c), S)`; in clr coordinates a part that gains share is
positive and a part that loses it negative, so this is the signed,
orthant-free use of the sphere):

| model | cos | angle° | log-score gap | signs right |
|---|---|---|---|---|
| oracle | 0.694 | 0 | 0 | 0.762 |
| ESAG, `z_{t−1}` | 0.691 | 4.2 | 0.257 | 0.761 |
| PN regression, `z_{t−1}` | 0.693 | 2.9 | 0.007 | 0.761 |
| VAR-OLS on z | 0.693 | 2.7 | 0.006 | 0.761 |
| ESAG, `v_{t−1}` | 0.514 | 39.5 | 1.087 | 0.677 |
| i.i.d. | −0.001 | 90.1 | 1.396 | 0.498 |

Which parts gain and which lose share is forecast right for 76% of the parts
(50% without covariates), by ESAG as well as by the true model; ESAG's
density is again 0.26 nats worse (the noise is anisotropic), the PN regression's
0.007.

**Signed compositions of net flows** (`w_t = f_t/|f_t|_1`, the whole sphere:
either the direction `f_t/|f_t|_2`, or the signed square root
`sign(w) √|w|`, Yu & Huang's map with the sign kept, whose positive orthant is
their case; L1 error of the forecast `ŵ`, share of signs right):

| flows | model | L1 | signs right |
|---|---|---|---|
| Gaussian VAR | ESAG, `f/|f|_2` | 0.996 | 0.674 |
| | ESAG, signed sqrt | 1.128 | 0.676 |
| | PN regression, `f/|f|_2` | 0.988 | 0.678 |
| | PN regression, signed sqrt | 1.136 | 0.679 |
| | VAR-OLS | 0.982 | 0.679 |
| | oracle | 0.981 | 0.680 |
| Poisson counts (in − out, sizes 50 … 2) | ESAG, `f/|f|_2` | 1.649 | 0.468 |
| | ESAG, signed sqrt | 1.603 | 0.532 |
| | PN regression, `f/|f|_2` | 1.077 | 0.569 |
| | PN regression, signed sqrt | 1.173 | 0.568 |
| | VAR-OLS | 1.063 | 0.571 |
| | persistence | 1.262 | 0.478 |

Signed data use the whole sphere naturally, and the plain direction is the
better embedding: for Gaussian flows it is exactly projected normal, and the
signed square root gets the same signs but larger share errors (1.13 against
0.99). ESAG fails on the count flows (fewer signs right than persistence):
their directions sit near `±e_1`, with a sign that depends on the lag, so
`μ(x)` passes through zero and ESAG's mean-anchored frame flips (its fit does
not converge, with raw or standardized lags, γ-rot or log-tangent). The
projected normal with a constant covariance has no frame to flip and matches
least squares.


## 5. What others do

A citation-checked survey (items matched on authors, year, title and venue in
search results; † checked through abstracts or citing papers only;
"preprint" = arXiv only).

*Directional models.* von Mises–Fisher mixtures and losses (Banerjee,
Dhillon, Ghosh & Sra 2005, JMLR 6) are the isotropic baseline; Kent /
Fisher–Bingham (Kent 1982, JRSS-B 44; Kume & Wood 2005, Biometrika 92) need
costly normalizing constants in high d, while ESAG's density is exact. The
angular central Gaussian is the law of the direction of *any* zero-centred
elliptical vector (Tyler 1987, Biometrika 74; Tyler's M-estimator, AoS 15),
the basis of robust scatter estimation for financial returns (Frahm &
Jaekel, Univ. Cologne DP 2/07); at `μ = 0` ESAG is the ACG, but its
mean-anchored frame is undefined there (§1). The general projected normal
regression — the model `_pn.PNRegression` fits by maximum likelihood — is
Presnell, Morrison & Littell's projected linear model (1998, JASA 93, by
EM) and Hernandez-Stumpfhauser, Breidt & van der Woerd's projected normal
of arbitrary dimension (2017, Bayesian Analysis 12, by MCMC, one
covariance entry fixed). Spherical regression: Paine, Preston, Tsagris &
Wood 2018 and 2020 (Stat Comput 28, 30; Kent or ESAG errors), Rosenthal,
Wu, Klassen & Srivastava 2014 (JASA 109; projective-linear mean link),
Scealy & Wood 2019 (JASA 114; scaled vMF), and heavy-tailed alternatives
(spherical Cauchy, Kato & McCullagh 2020, Bernoulli 26; regression with it,
Tsagris, Papastamoulis & Kato 2025, Stat Comput 35).

*Directional and sign time series.* Circular ARMA and Markov models (Fisher
& Lee 1994, JRSS-B 56†; Kato 2010, JRSS-B 72; score-driven von Mises,
Harvey, Hurn, Palumbo & Thiele 2024, J Econometrics 239); projected-normal
dynamics on the circle (Wang & Gelfand 2014, JASA 109; Nuñez-Antonio &
Gutiérrez-Peña 2014, CSDA 71); and in any dimension the projected dynamic
linear model of Zito & Kowal (preprint 2308.14996: Gibbs sampling and a
particle filter, applied to square-root electricity shares), the closest
existing work to these experiments. Zhu & Müller (2024, J Econometrics 239)
fit spherical autoregressions by rotations and chose the *differenced*
version for square-root energy shares because the levels trended — the
same conclusion as §3. Christoffersen & Diebold (2006, Management Science
52): volatility dynamics make the *sign* of a return predictable only when
its expected value is not zero — here, the concentration `|m_t|/s_t`.
Spatial-sign methods for multivariate series test for white noise or build
factor models (Hallin & Paindaveine 2002, Bernoulli 8; Fan, Liu & Wang 2018,
AoS 46); location estimates bias high-dimensional sign tests (Feng, Zou &
Wang 2016, JASA 111). The survey found no ESAG time-series or forecasting
work, and no model of the direction of return vectors with vMF, ESAG or ACG.

*Compositions.* Log-ratio analysis (Aitchison 1982, JRSS-B 44; ilr, Egozcue
et al. 2003, Math Geol 35); the square-root map with Kent regression, which
takes zeros as they are but is not Aitchison-coherent (Scealy & Welsh 2011,
JRSS-B 73†; small concentration and folding, 2014, Stat Comput 24); ESAG
truncated to the non-negative orthant with spatial effects (Schwob, Hooten,
Calzada & Keitt, preprint 2410.03648), the direct extension of square-root
ESAG; the α-transformation between the two (Tsagris, Preston & Wood,
preprint 1106.1451; folded, Tsagris & Stewart 2020, ANZJS 62); score
matching for compositions with zeros (Scealy & Wood 2023, JASA; Scealy,
Hingee, Kent & Wood 2024, Stat Comput 34); sparse log-contrast regression
(Lin, Shi, Feng & Li 2014, Biometrika 101). Compositional time series: alr
VARMA (Brunsdon & Smith 1998, J Off Stat 14), VAR on ilr coordinates
(Kynčlová, Filzmoser & Hron 2015, J Forecasting 34), Dirichlet ARMA (Zheng
& Chen 2017, JMVA 158; Bayesian, Katz, Brusch & Weiss 2024, IJF 40),
logistic-normal dynamic linear models for counts (Silverman et al. 2018,
Microbiome 6; 2022, JMLR 23).

*Signed compositions.* The accepted practice is to split, not to sign:
negative quantities are differences of positive ones, analysed as
sub-compositions of the positive parts plus the gross total (in and out
flows, long and short weights; Coenders & Arimany-Serrat, preprint
2305.16842). Compositional change is the perturbation `p_t ⊖ p_{t−1}`
(Aitchison & Ng 2005, Stat Model 5), whose ilr difference is the change
direction of §4. The survey found no use of the signed square root onto the
whole sphere in statistics (it appears in computer vision: Perronnin,
Sánchez & Mensink 2010, ECCV†; RootSIFT, Arandjelović & Zisserman 2012).
It and the direction `f/|f|_2` are one-to-one maps of the same signed
composition (`w = sign(s) s² = u/|u|_1`): both keep the signs and the
shares, net share `Σw` included, and drop the gross size `|f|_1`; they
differ only in geometry, and §4 found the direction's the better one.

*Evaluation.* Beyond the log score, angular error and mean-resultant-length
calibration used here: energy and kernel scores (Gneiting & Raftery 2007,
JASA 102), coverage of predictive regions, and for signs hit rates with the
Pesaran–Timmermann test (1992, JBES 10).

## 6. Conclusions

* **Direction forecasts work.** In the stationary, differenced and
  compositional designs ESAG regression's forecast direction is within
  4–10° of the oracle's mean direction (the PN regression's within 3–6°),
  its mean cosine within 0.01 of the oracle's, and it gets as many share
  changes' signs right; it also knows when the next direction is
  predictable (ρ correlation 0.99).
* **The density needs the right structure.** When the noise is anisotropic
  and the mean direction turns with the state (every VAR-type series), the
  projected normal with a constant covariance is the exact model; ESAG's
  mean-anchored shape costs 0.26–0.39 nats per step. When the mean
  direction passes through zero (mean-zero series, signed flows), ESAG's
  frame degenerates: its fits fail on count flows, and its penalized path
  cannot select (at d = 10 it kept every lag, for the mean and the shape,
  with unsolved λ's).
* **Prepare the series.** Difference integrated series, demean stationary
  ones, and use scale-free covariates under volatility clustering; with
  volatility the directional model on lagged directions beats the Gaussian
  VAR.
* **Compositions.** The square-root map matches the logistic normal in point
  forecasts when the logistic normal is the truth (and loses 0.085 nats), and
  zeros alone do not reverse that. For changes of a composition and for
  signed compositions, the direction on the whole sphere is the natural
  representation (losing parts negative); the signed square root is a valid
  extension of the square-root map but not a better one here.
* **Next steps.** A native projected normal regression with a constant
  covariance (the projected linear model of Presnell et al. 1998 and
  Hernandez-Stumpfhauser et al. 2017, here by maximum likelihood in 0.1–0.2 s
  at d = 10), with the package's penalties on the lag columns, bootstrap
  tests and prediction regions, for VAR-type and signed data; a warning when
  a penalized ESAG path starts from an intercept-only model with a mean
  vector near zero; count observation layers for compositions with zeros;
  coordinate-wise devolatilization and real series (returns, energy shares)
  against Zito & Kowal's dynamic model and Zhu & Müller's autoregression.
