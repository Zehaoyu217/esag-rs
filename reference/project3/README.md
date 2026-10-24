# Project 3 evidence scripts

R scripts behind `docs/project3/SPEC.md` (what the user's Project 3 R code
does) and `docs/project3/METHODOLOGY.md` (review and corrected design). They
were run against the extracted research archive (`Research_project3.zip`,
not in this repository) with R 4.6.1; paths inside point at that scratch
location.

* `make_fixtures.R`, `write_json.R`: produce `tests/fixtures/p3_*.json` by
  evaluating the user's method scripts verbatim (only `optim` is wrapped to
  record its result).
* `disc_check*.R`, `inverse_map.R`: discontinuities of `γ ↦ V` at zero
  groups and vanishing trailing pairs; the exp-1 / exp-2 truths.
* `check_fig*.R`, `analyze_sim.R`: figures 1–2 recomputed from the stored
  simulation results (D1: Fig. 2 scored against the wrong truth).
* `start_dependence.R`, `polish.R`: R's BFGS points are start-dependent and
  not optimal.
* `prox_check.R`, `grad_S_check.R`: the eigen-gap prox and the log-tangent
  gradient against brute force / finite differences.
