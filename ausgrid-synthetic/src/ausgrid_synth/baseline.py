"""Joint conditional PCA/Ridge/Gaussian baseline; no raw-day resampling."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def fit(data: dict, path: Path, seed: int = 2026) -> None:
    available = np.flatnonzero(data["split"] == 0)
    indices = available
    x = data["x"][indices] / data["scale"][None, :, None]
    target = np.log1p(x).reshape(len(indices), 96)
    pca = PCA(n_components=16, svd_solver="randomized", random_state=seed)
    scores = pca.fit_transform(target)
    reg = Ridge(alpha=20.0).fit(data["cond"][indices], scores)
    residual = scores - reg.predict(data["cond"][indices])
    cov = LedoitWolf().fit(residual).covariance_.astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump({"pca": pca, "reg": reg, "cov": cov, "scale": data["scale"],
                 "train_days": len(indices), "seed": seed}, tmp)
    tmp.replace(path)


def sample(path: Path, cond: np.ndarray, seed: int) -> np.ndarray:
    model = joblib.load(path)
    rng = np.random.default_rng(seed)
    scores = model["reg"].predict(cond) + rng.multivariate_normal(np.zeros(16), model["cov"], size=len(cond))
    flat = model["pca"].inverse_transform(scores).reshape(-1, 2, 48)
    return (np.expm1(np.maximum(flat, 0)) * model["scale"][None, :, None]).astype("float32")
