from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import baseline, data as dataset, evaluate


def main():
    p = argparse.ArgumentParser(description="Ausgrid paired-daily synthesis, with stage checkpoints")
    p.add_argument("stage", choices=["prepare", "baseline", "train", "sample", "evaluate"])
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--arm", choices=["statistical", "vae", "vae_post", "vae_daylight"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--context", choices=["train", "test", "privacy"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()
    root = args.root.resolve()
    prepared = root / "data" / "prepared" / "prepared.npz"
    models, samples, results = (root / "outputs" / name for name in ("checkpoints", "samples", "reports"))
    if args.stage == "prepare":
        report = dataset.prepare(root / "data" / "raw", prepared.parent)
        print(json.dumps(report, indent=2)); return
    d = dataset.load_prepared(prepared)
    if args.stage == "baseline":
        baseline.fit(d, models / "statistical.joblib"); print("Saved statistical baseline"); return
    if args.arm is None: p.error("--arm is required")
    if args.stage == "train":
        if args.arm not in ("vae", "vae_daylight"): p.error("train requires vae or vae_daylight")
        from .train import fit
        fit(d, models / f"{args.arm}_seed{args.seed}.pt", args.arm == "vae_daylight", args.seed,
            epochs=args.epochs, batch_size=args.batch_size)
        return
    if args.stage == "sample":
        if args.context is None: p.error("sample requires --context")
        rng = np.random.default_rng(2026)
        if args.context == "train":
            available = np.flatnonzero(d["split"] == 0)
            idx = rng.choice(available, size=min(30000, len(available)), replace=False)
        elif args.context == "test": idx = np.flatnonzero(d["split"] == 2)
        else:
            # Common contexts across models. These reveal only calendar/capacity, not load profiles.
            available = np.flatnonzero(np.isin(d["split"], [0, 2]))
            idx = rng.choice(available, size=min(8000, len(available)), replace=False)
        if args.arm == "statistical":
            x = baseline.sample(models / "statistical.joblib", d["cond"][idx], 2026 + args.seed)
        else:
            from .train import sample
            source_arm = "vae" if args.arm == "vae_post" else args.arm
            x = sample(d, models / f"{source_arm}_seed{args.seed}.pt", idx, 2026 + args.seed,
                       post_mask=args.arm == "vae_post")
        samples.mkdir(parents=True, exist_ok=True)
        path = samples / f"{args.arm}_seed{args.seed}_{args.context}.npz"
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as stream: np.savez_compressed(stream, x=x, idx=idx)
        tmp.replace(path)
        print(f"Saved {len(idx)} paired synthetic days to {path}"); return
    evaluate.evaluate(d, samples, results, args.arm, args.seed)


if __name__ == "__main__": main()
