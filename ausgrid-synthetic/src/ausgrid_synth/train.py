from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import ConditionalVAE


def _dataset(data: dict, code: int):
    sel = data["split"] == code
    x = (data["x"][sel] / data["scale"][None, :, None]).astype("float32")
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(data["cond"][sel].astype("float32")),
                         torch.from_numpy(data["night"][sel]))


def fit(data: dict, checkpoint: Path, constrained: bool, seed: int, epochs: int = 30,
        batch_size: int = 512, patience: int = 5, lr: float = 1e-3):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set = _dataset(data, 0), _dataset(data, 1)
    generator = torch.Generator().manual_seed(seed)
    loaders = (DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator),
               DataLoader(val_set, batch_size=batch_size, shuffle=False))
    model = ConditionalVAE().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    start, best, stale = 0, float("inf"), 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    latest = checkpoint.with_name(checkpoint.stem + "_latest.pt")
    if latest.exists():
        # Keep CPU RNG and DataLoader generator states on CPU when resuming on GPU.
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["seed"] != seed or state["constrained"] != constrained:
            raise ValueError("Checkpoint configuration mismatch")
        model.load_state_dict(state["model"]); optim.load_state_dict(state["optimizer"])
        for slot in optim.state.values():
            for name, value in slot.items():
                if isinstance(value, torch.Tensor): slot[name] = value.to(device)
        start, best, stale = state["epoch"] + 1, state["best"], state["stale"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        generator.set_state(state["loader_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None: torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resuming epoch {start + 1} from {latest}", flush=True)
    if start >= epochs:
        print(f"Checkpoint already completed {start} epochs; no additional training requested.")
        return
    for epoch in range(start, epochs):
        results = []
        for is_train, loader in zip((True, False), loaders):
            model.train(is_train)
            loss_sum = 0.0; n = 0
            for x, cond, night in loader:
                x, cond, night = x.to(device), cond.to(device), night.to(device)
                with torch.set_grad_enabled(is_train):
                    pred, mean, logvar = model(x, cond, night if constrained else None)
                    recon = ((pred - x) ** 2).mean()
                    kl = (-0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(dim=1)).mean()
                    loss = recon + 0.001 * kl
                    if is_train:
                        optim.zero_grad(); loss.backward(); optim.step()
                loss_sum += loss.item() * len(x); n += len(x)
            results.append(loss_sum / n)
        train_loss, val_loss = results
        improved = val_loss < best - 1e-6
        if improved:
            best, stale = val_loss, 0
            best_tmp = checkpoint.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "seed": seed, "constrained": constrained,
                        "best_val_loss": best, "scale": data["scale"].tolist()}, best_tmp)
            best_tmp.replace(checkpoint)
        else: stale += 1
        state = {"model": model.state_dict(), "optimizer": optim.state_dict(), "epoch": epoch,
                 "seed": seed, "constrained": constrained, "best": best, "stale": stale,
                 "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                 "python_rng": random.getstate(), "loader_rng": generator.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        latest_tmp = latest.with_suffix(".tmp")
        torch.save(state, latest_tmp)
        latest_tmp.replace(latest)
        print(f"epoch={epoch+1} train={train_loss:.5f} val={val_loss:.5f} best={best:.5f}", flush=True)
        if stale >= patience: break
    (checkpoint.parent / (checkpoint.stem + "_status.json")).write_text(json.dumps({
        "seed": seed, "constrained": constrained, "best_validation_loss": best,
        "last_epoch": epoch + 1, "target_epochs": epochs,
        "early_stopped": stale >= patience, "device": str(device)}, indent=2))


@torch.no_grad()
def sample(data: dict, checkpoint: Path, indices: np.ndarray, seed: int, post_mask: bool = False,
           batch_size: int = 4096) -> np.ndarray:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ConditionalVAE().to(device)
    model.load_state_dict(state["model"]); model.eval()
    out = []
    for chunk in np.array_split(indices, max(1, int(np.ceil(len(indices) / batch_size)))):
        cond = torch.from_numpy(data["cond"][chunk].astype("float32")).to(device)
        night = torch.from_numpy(data["night"][chunk]).to(device)
        z = torch.randn(len(chunk), model.latent_dim, device=device)
        result = model.decode(z, cond, night if state["constrained"] else None)
        if post_mask: result[:, 1, :] *= (~night).float()
        out.append(result.cpu().numpy() * data["scale"][None, :, None])
    return np.concatenate(out).astype("float32")
