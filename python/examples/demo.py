"""ESAG regression from Python: fit, bootstrap tests, LASSO family.

    python examples/demo.py            # d = 3
    python examples/demo.py 5          # d = 5 (log-tangent shape, chosen automatically)

n = 2000 observations, 5 covariates (+ const); the last covariate is noise.
"""

import sys
import time

import numpy as np
import pandas as pd

import esag

d = int(sys.argv[1]) if len(sys.argv) > 1 else 3
n, p, n_boot = 2000, 5, 500
rng = np.random.default_rng(7)

# true model: the mean direction depends on x1..x4, the shape on x1 only
X = pd.DataFrame(rng.normal(size=(n, p)), columns=["temp", "ph", "depth", "salt", "noise"])
g = (d - 2) * (d + 1) // 2
B = np.zeros((d + g, p + 1))
B[:d, 0] = np.linspace(1.5, 0.5, d)
B[:d, 1:5] = rng.normal(scale=0.5, size=(d, 4))
B[d:, 0] = np.linspace(1.0, -0.6, g)
B[d:, 1] = 0.6
with np.errstate(all="ignore"):
    eta = np.column_stack([np.ones(n), X.values]) @ B.T
Y = np.vstack([esag.Distribution(list(e[:d]), list(e[d:])).sample(1, seed=i) for i, e in enumerate(eta)])
Y = pd.DataFrame(Y, columns=[f"y{k + 1}" for k in range(d)])

t = time.perf_counter()
model = esag.ESAG(Y, X)
res = model.fit()
print(res.summary())
print(f"fit: {time.perf_counter() - t:.3f} s\n")

print(f"Bootstrap tests, B = {n_boot} (RoC, D, M, LR in one run each):")
for null, exog in [("isotropy", None), ("mu", "noise"), ("shape", "noise"),
                   ("feature", "noise"), ("feature", "temp")]:
    t = time.perf_counter()
    test = res.test(null, exog, n_boot=n_boot)
    el = time.perf_counter() - t
    pv = "  ".join(f"{k} {v:.3f}" for k, v in test.pvalue.items() if k != "LR_chi2")
    print(f"  {test.null:<24} {el:5.2f} s   p: {pv}")
print()
print(test.summary())

print("LASSO family (BIC selection):")
for penalty, kw in [("lasso", {}), ("adaptive_lasso", {}), ("group_lasso", {}),
                    ("adaptive_group_lasso", {"groups": "covariate_part"}),
                    ("sparse_group_lasso", {"l1_ratio": 0.5})]:
    t = time.perf_counter()
    reg = model.fit_regularized(penalty, **kw)
    el = time.perf_counter() - t
    dropped = [c for c in reg.params.index if (reg.params.loc[c] == 0).all()]
    print(f"  {reg.method:<26} {el:5.2f} s   alpha {reg.alpha:.4g}   covariates dropped: {dropped}")
print()
print(reg.summary())
