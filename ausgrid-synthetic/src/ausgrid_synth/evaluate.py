from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.neighbors import NearestNeighbors


def _sample(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return z["x"], z["idx"]


def _cor(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-8 and np.std(b) > 1e-8 else None


def fidelity(real: np.ndarray, synth: np.ndarray, night: np.ndarray) -> dict:
    assert real.shape == synth.shape and night.shape == (len(real), 48)
    metrics = {}
    for c, label in enumerate(("total_consumption", "gross_generation")):
        for reducer, suffix in ((lambda y: y.sum(-1), "daily_kwh"), (lambda y: y.max(-1), "peak_interval_kwh")):
            a, b = reducer(real[:, c]), reducer(synth[:, c])
            metrics[f"{label}_{suffix}_wasserstein"] = float(wasserstein_distance(a, b))
            metrics[f"{label}_{suffix}_real_q95"] = float(np.quantile(a, .95))
            metrics[f"{label}_{suffix}_synthetic_q95"] = float(np.quantile(b, .95))
        metrics[f"{label}_median_profile_mae"] = float(np.abs(np.median(real[:, c], 0) - np.median(synth[:, c], 0)).mean())
        metrics[f"{label}_q90_profile_mae"] = float(np.abs(np.quantile(real[:, c], .9, axis=0) - np.quantile(synth[:, c], .9, axis=0)).mean())
    r_co = _cor(real[:, 0].sum(-1), real[:, 1].sum(-1))
    s_co = _cor(synth[:, 0].sum(-1), synth[:, 1].sum(-1))
    metrics["gccl_gg_daily_total_correlation_real"] = r_co
    metrics["gccl_gg_daily_total_correlation_synthetic"] = s_co
    metrics["negative_values"] = int((synth < 0).sum())
    metrics["night_solar_gt_0p02_intervals"] = int((synth[:, 1][night] > .02).sum())
    metrics["night_solar_kwh"] = float(synth[:, 1][night].sum())
    metrics["night_intervals"] = int(night.sum())
    metrics["real_night_solar_gt_0p02_intervals"] = int((real[:, 1][night] > .02).sum())
    metrics["real_night_solar_max_kwh"] = float(real[:, 1][night].max())
    return metrics


def utility(data: dict, synthetic_train: np.ndarray, train_idx: np.ndarray,
            test_idx: np.ndarray, seed: int = 2026) -> dict:
    # Same contexts and same number of examples for real and synthetic training.
    def inputs(x, indices):
        return np.column_stack((x[:, 0, :24], data["cond"][indices]))
    def targets(x): return x[:, 0, 24:]
    real_train, real_test = data["x"][train_idx], data["x"][test_idx]
    real = Ridge(alpha=100).fit(inputs(real_train, train_idx), targets(real_train))
    synth = Ridge(alpha=100).fit(inputs(synthetic_train, train_idx), targets(synthetic_train))
    xt, yt = inputs(real_test, test_idx), targets(real_test)
    real_loss = np.abs(real.predict(xt) - yt).mean(axis=1)
    synth_loss = np.abs(synth.predict(xt) - yt).mean(axis=1)
    naive_loss = np.abs(real_test[:, 0, :24].mean(axis=1, keepdims=True) - yt).mean(axis=1)
    rng = np.random.default_rng(seed)
    unique = np.unique(data["customer"][test_idx])
    house = data["customer"][test_idx]
    # Equal household weight, then household bootstrap on the paired loss difference.
    house_diff = np.array([(synth_loss[house == h] - real_loss[house == h]).mean() for h in unique])
    boot = np.array([rng.choice(house_diff, len(unique), replace=True).mean() for _ in range(300)])
    return {"synthetic_train_test_real_mae": float(synth_loss.mean()),
            "real_train_test_real_mae": float(real_loss.mean()),
            "naive_test_real_mae": float(naive_loss.mean()),
            "paired_mae_difference_household_mean": float(house_diff.mean()),
            "paired_mae_difference_95pct_ci": np.quantile(boot, [.025, .975]).tolist(),
            "train_days_each": len(train_idx), "test_days": len(test_idx)}


def privacy(data: dict, synthetic: np.ndarray, seed: int = 2026) -> dict:
    """A bounded household membership inference probe, not a privacy guarantee."""
    rng = np.random.default_rng(seed)
    candidate = np.flatnonzero(np.isin(data["split"], [0, 2]))
    ids = data["customer"][candidate]
    picked = np.concatenate([rng.choice(candidate[ids == h], size=min(16, sum(ids == h)), replace=False)
                             for h in np.unique(ids)])
    real = (data["x"][picked] / data["scale"][None, :, None]).reshape(len(picked), 96)
    generated = (synthetic / data["scale"][None, :, None]).reshape(len(synthetic), 96)
    fit_idx = rng.choice(np.flatnonzero(data["split"] == 0), size=8000, replace=False)
    train = (data["x"][fit_idx] / data["scale"][None, :, None]).reshape(len(fit_idx), 96)
    projection = PCA(n_components=16, svd_solver="randomized", random_state=seed).fit(train)
    nn = NearestNeighbors(n_neighbors=1).fit(projection.transform(generated))
    distances = nn.kneighbors(projection.transform(real), return_distance=True)[0].ravel()
    pids = data["customer"][picked]
    unique = np.unique(pids)
    scores = np.array([-np.median(distances[pids == h]) for h in unique])
    labels = np.array([int(data["split"][picked[pids == h][0]] == 0) for h in unique])
    auc = roc_auc_score(labels, scores)
    boots = []
    for _ in range(300):
        draw = np.concatenate([rng.choice(np.flatnonzero(labels == c), sum(labels == c), replace=True) for c in (0, 1)])
        boots.append(roc_auc_score(labels[draw], scores[draw]))
    return {"attack": "nearest synthetic profile, 16 dates per candidate, PCA-16",
            "household_membership_auc": float(auc), "auc_95pct_bootstrap_ci": np.quantile(boots, [.025, .975]).tolist(),
            "member_households": int(labels.sum()), "nonmember_households": int((1 - labels).sum()),
            "interpretation": "Exploratory attack on released daily samples; no privacy guarantee."}


def evaluate(data: dict, sample_dir: Path, out_dir: Path, arm: str, seed: int):
    def locate(context): return sample_dir / f"{arm}_seed{seed}_{context}.npz"
    test, test_idx = _sample(locate("test"))
    train, train_idx = _sample(locate("train"))
    attack, _ = _sample(locate("privacy"))
    if not np.all(data["split"][test_idx] == 2) or not np.all(data["split"][train_idx] == 0):
        raise ValueError("Sample contexts do not match the locked split")
    report = {"arm": arm, "seed": seed,
              "fidelity_and_physics": fidelity(data["x"][test_idx], test, data["night"][test_idx]),
              "utility": utility(data, train, train_idx, test_idx, seed),
              "disclosure_probe": privacy(data, attack, seed)}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{arm}_seed{seed}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(path)
    print(f"Saved {path}")
    return report
