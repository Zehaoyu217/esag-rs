import json
import time

import numpy as np
import pandas as pd
import pytest

import esag

D, P = 3, 3


def simulate(n, B, seed):
    """Responses from ESAG(mu_i, gamma_i) with eta_i = B [1, x_i]."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, B.shape[1] - 1))
    with np.errstate(all="ignore"):  # spurious Accelerate warnings in numpy 2.0
        eta = np.column_stack([np.ones(n), x]) @ B.T
    d = next(k for k in range(2, 20) if k + (k - 2) * (k + 1) // 2 == B.shape[0])
    y = np.vstack([esag.Distribution(list(e[:d]), list(e[d:])).sample(1, seed=1000 * seed + i)
                   for i, e in enumerate(eta)])
    return x, y


def b_true():
    B = np.zeros((D + 2, P + 1))
    B[:D, 0] = [1.5, 0.5, 1.0]
    B[:D, 1] = [-0.6, 1.2, 0.3]
    B[:D, 2] = [0.4, 0.0, -0.5]
    B[D:, 0] = [1.0, -0.6]
    B[D:, 1] = [0.5, 1.0]
    return B  # third covariate ("noise") has no effect


@pytest.fixture(scope="module")
def data():
    x, y = simulate(800, b_true(), 1)
    X = pd.DataFrame(x, columns=["temp", "ph", "noise"])
    Y = pd.DataFrame(y, columns=["east", "north", "up"])
    return X, Y


@pytest.fixture(scope="module")
def res(data):
    X, Y = data
    return esag.ESAG(Y, X).fit()


def test_fit_recovers_truth_and_labels(res):
    assert res.converged
    assert list(res.params.index) == ["const", "temp", "ph", "noise"]
    assert list(res.params.columns) == ["mu[1]", "mu[2]", "mu[3]", "gamma[1]", "gamma[2]"]
    err = np.abs(res.params.values.T - b_true())
    assert (err < 5 * res.bse.values.T).all(), err
    assert (res.bse.values > 0).all()
    np.testing.assert_allclose(res.tvalues.values, res.params.values / res.bse.values)
    assert ((res.pvalues.values >= 0) & (res.pvalues.values <= 1)).all()
    ci = res.conf_int(0.05)
    assert list(ci.columns) == ["lower", "upper"]
    assert (ci["lower"] < ci["upper"]).all()
    assert res.nobs == 800 and res.df_model == 20 and res.df_resid == 780
    assert np.isclose(res.aic, -2 * res.llf + 2 * 20)
    cov = res.cov_params()
    assert cov.shape == (20, 20) and "mu[1]:const" in cov.index
    for label, v in zip(cov.index, np.sqrt(np.diag(cov.values))):
        eq, ex = label.split(":")
        assert np.isclose(v, res.bse.loc[ex, eq], rtol=1e-10)


def test_summary_is_statsmodels_style(res):
    s = str(res.summary())
    for token in ["ESAG Regression Results", "Dep. Variable:", "east, north, up",
                  "No. Observations:", "Log-Likelihood:", "P>|z|", "[0.025", "temp", "gamma[2]"]:
        assert token in s, token
    assert "[0.050" in str(res.summary(alpha=0.1)) and "0.950]" in str(res.summary(alpha=0.1))
    assert res.summary()._repr_html_().startswith("<pre>")


def test_predict_and_diagnostics(res, data):
    X, _ = data
    m = res.predict()
    assert m.shape == (800, 3)
    np.testing.assert_allclose(np.linalg.norm(m.values, axis=1), 1.0)
    new = res.predict(X.iloc[:5])
    np.testing.assert_allclose(new.values, m.values[:5])
    assert res.linear_predictor().shape == (800, 5)
    assert np.isclose(res.loglike_obs().sum(), res.llf)
    r = res.resid
    assert list(r.columns) == ["Q", "T1"] and (r.values >= 0).all()
    sim = res.simulate(seed=3)
    np.testing.assert_allclose(np.linalg.norm(sim.values, axis=1), 1.0)


def test_numpy_input_and_add_constant(data):
    X, Y = data
    a = esag.ESAG(Y.values, X.values).fit()
    assert a.model.exog_names == ["const", "x1", "x2", "x3"]
    assert a.model.endog_names == ["y1", "y2", "y3"]
    with_const = np.column_stack([np.ones(len(X)), X.values])
    b = esag.ESAG(Y.values, with_const).fit()  # sm.add_constant(X)
    assert b.model.exog_names == ["const", "x2", "x3", "x4"]
    np.testing.assert_allclose(a.params, b.params, atol=1e-8)
    np.testing.assert_allclose(b.predict(with_const[:3]), a.predict(X.values[:3]), atol=1e-8)


def test_missing_drop(data):
    X, Y = data
    Xn = X.copy()
    Xn.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        esag.ESAG(Y, Xn)
    assert esag.ESAG(Y, Xn, missing="drop").nobs == 799


def test_bootstrap_tests(res):
    t0 = time.perf_counter()
    iso = res.test("isotropy", n_boot=200)
    noise = res.test("feature", "noise", n_boot=200)
    elapsed = time.perf_counter() - t0
    assert iso.pvalue["LR"] < 0.01 and iso.pvalue["RoC"] < 0.01
    assert min(noise.pvalue[k] for k in ["RoC", "D", "LR"]) > 0.01
    assert noise.df == 5 and noise.n_boot == 200 and noise.n_failed == 0
    assert set(noise.statistic) == {"RoC", "D", "M", "LR"}
    assert len(noise.bootstrap["LR"]) == 200
    assert list(noise.table.index) == ["RoC", "D", "M", "LR"]
    s = str(noise.summary())
    assert "ESAG Bootstrap Hypothesis Test" in s and "no effect of noise" in s
    assert res.test("mu", 2, n_boot=50).null == "mu free of noise"
    assert elapsed < 30


@pytest.mark.parametrize("penalty,kw", [
    ("lasso", {}),
    ("adaptive_lasso", {}),
    ("group_lasso", {}),
    ("adaptive_group_lasso", {"groups": "covariate_part"}),
    ("sparse_group_lasso", {"l1_ratio": 0.3}),
    ("adaptive_lasso", {"target": "mean"}),
    ("group_lasso", {"adaptive": True}),
])
def test_lasso_family_drops_noise(data, penalty, kw):
    X, Y = data
    reg = esag.ESAG(Y, X).fit_regularized(penalty, **kw)
    assert (reg.params.loc["noise"] == 0).all() or kw.get("target") == "mean"
    if kw.get("target") == "mean":
        assert (reg.params.loc["noise", ["mu[1]", "mu[2]", "mu[3]"]] == 0).all()
    assert (reg.params.loc["temp"] != 0).any()
    assert reg.relaxed is not None and reg.relaxed.converged
    path = reg.path
    assert path["selected"].sum() == 1
    assert np.isclose(reg.alpha, path.loc[path["selected"], "alpha"].iloc[0])
    s = str(reg.summary())
    assert "ESAG Regularized Regression Results" in s and "(dropped)" in s
    assert len(reg.params_path()) == len(path)


def test_parameter_groups(data):
    X, Y = data
    reg = esag.ESAG(Y, X).fit_regularized("group_lasso", groups="parameter")
    slopes = reg.params.drop(index="const")
    # each equation's slopes are kept or dropped together
    assert all((slopes[c] == 0).all() or (slopes[c] != 0).all() for c in slopes.columns)


def test_custom_groups_alpha_and_cv(data):
    X, Y = data
    m = esag.ESAG(Y, X)
    groups = [[("mu[1]", "noise"), ("mu[2]", "noise"), ("mu[3]", "noise")],
              [("gamma[1]", "noise"), ("gamma[2]", "noise")]]
    reg = m.fit_regularized("group_lasso", groups=groups, alpha=[0.0, 0.02, 0.05])
    assert len(reg.path) == 3
    one = m.fit_regularized("lasso", alpha=0.05)
    assert len(one.path) == 1 and np.isclose(one.alpha, 0.05)
    cv = m.fit_regularized("adaptive_lasso", select="cv", cv_folds=3)
    assert (cv.params.loc["temp"] != 0).any()


def test_iid_and_shape_penalties():
    mu = [2.0, 1.0, 0.5, 0.0]
    dist = esag.Distribution(mu, [0.0] * 5)
    y = dist.sample(1500, seed=4)
    m = esag.ESAG(y, shape="log-tangent")
    assert m.shape == "log-tangent"
    r = m.fit()
    assert r.converged and r.params.shape == (1, 9)
    iso = m.fit_regularized("isotropy")
    assert (iso.params.values[:, 4:] == 0).all()
    gaps = m.fit_regularized("eigen_gap")
    assert "eigen-gap" in gaps.method
    with pytest.raises(ValueError):
        esag.ESAG(y, shape="gamma").fit_regularized("eigen_gap")


def test_auto_shape():
    rng = np.random.default_rng(2)
    y = rng.normal(size=(50, 5))
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    assert esag.ESAG(y, rng.normal(size=(50, 2))).shape == "log-tangent"
    assert esag.ESAG(y).shape == "gamma"
    assert esag.ESAG(y[:, :3], rng.normal(size=(50, 2))).shape == "gamma"


def test_distribution():
    dist = esag.Distribution([1.0, -0.5, 2.0], [0.6, -0.3])
    y = dist.sample(2000, seed=1)
    np.testing.assert_allclose(np.linalg.norm(y, axis=1), 1.0)
    np.testing.assert_allclose(dist.second_moment, (y.T @ y) / len(y), atol=0.03)
    assert np.isclose(np.trace(dist.second_moment), 1.0)
    assert dist.logpdf(y[:4]).shape == (4,)
    np.testing.assert_allclose(np.linalg.det(dist.covariance), 1.0)
    assert esag.Distribution([1.0, 0.5, 0.2]).gamma == [0.0, 0.0]
    with pytest.raises(ValueError):
        esag.Distribution([1.0, 0.5])  # ESAG needs d >= 3


def test_json_roundtrip(res):
    js = res.to_json()
    assert json.loads(js)["d"] == 3
    back = esag.ESAGResults.from_json(res.model, js)
    np.testing.assert_allclose(back.params, res.params)
    assert back.test("feature", "noise", n_boot=20).n_boot == 20


def test_errors(data):
    X, Y = data
    with pytest.raises(ValueError):
        esag.ESAG(Y, X.iloc[:10])
    m = esag.ESAG(Y, X)
    with pytest.raises(ValueError, match="unknown penalty"):
        m.fit_regularized("ridge")
    with pytest.raises(ValueError):
        m.fit().test("mu")  # needs a covariate
    with pytest.raises(ValueError):
        m.fit().test("mu", "nope")
