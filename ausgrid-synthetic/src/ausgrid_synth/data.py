from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SLOTS = [f"{(i + 1) // 2 % 24}:{'30' if i % 2 == 0 else '00'}" for i in range(48)]
SYDNEY = ZoneInfo("Australia/Sydney")


def _files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("Solar home 20*.csv"))
    if len(files) != 3 or not all(y in " ".join(f.name for f in files) for y in ("2010-2011", "2011-2012", "2012-2013")):
        raise FileNotFoundError("Put the three original 'Solar home YYYY-YYYY.csv' files in data/raw")
    return files


def _read(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, skiprows=1, keep_default_na=False, low_memory=False)
    expected = ["Customer", "Generator Capacity", "Consumption Category", "date", *SLOTS]
    if any(c not in d.columns for c in expected):
        raise ValueError(f"Unexpected headers in {path.name}")
    fmt = "%d-%b-%y" if "2010-2011" in path.name else "%d/%m/%Y"
    d["date"] = pd.to_datetime(d["date"], format=fmt, errors="raise")
    d["Customer"] = pd.to_numeric(d["Customer"], errors="raise").astype("int16")
    d["Generator Capacity"] = pd.to_numeric(d["Generator Capacity"], errors="raise").astype("float32")
    d[SLOTS] = d[SLOTS].apply(pd.to_numeric, errors="coerce").astype("float32")
    d["Consumption Category"] = d["Consumption Category"].str.strip()
    if d.duplicated(["Customer", "date", "Consumption Category"]).any():
        raise ValueError(f"Duplicate customer/date/category rows in {path.name}")
    if not set(d["Consumption Category"]).issubset({"GC", "CL", "GG"}):
        raise ValueError(f"Unknown category in {path.name}")
    d["valid_row"] = np.isfinite(d[SLOTS].to_numpy()).all(axis=1) & (d[SLOTS].to_numpy() >= 0).all(axis=1)
    if "Row Quality" in d:
        d["valid_row"] &= d["Row Quality"].astype(str).str.strip().eq("")
    return d


def _clock_change(date: pd.Timestamp) -> bool:
    day = date.to_pydatetime()
    return day.replace(hour=0, tzinfo=SYDNEY).utcoffset() != day.replace(hour=23, tzinfo=SYDNEY).utcoffset()


def night_mask(dates: np.ndarray, latitude: float = -33.87, longitude: float = 151.21,
               threshold_degrees: float = -9.0) -> np.ndarray:
    """Conservative astronomical-night mask at approximate Sydney location.

    Columns are half-hour END labels; final 0:00 represents the previous day's
    23:30–00:00 interval. Exclude DST-change dates before using this function.
    """
    unique, inverse = np.unique(dates.astype("datetime64[D]"), return_inverse=True)
    masks = np.empty((len(unique), 48), dtype=bool)
    lat = np.deg2rad(latitude)
    for j, value in enumerate(unique):
        day = datetime.strptime(str(value), "%Y-%m-%d")
        offset = day.replace(hour=12, tzinfo=SYDNEY).utcoffset().total_seconds() / 3600
        doy = day.timetuple().tm_yday
        midhour = (np.arange(48) + 0.5) / 2
        gamma = 2 * np.pi / 365 * (doy - 1 + (midhour - 12) / 24)
        eq = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
        decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
                - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
                - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
        hour_angle = np.deg2rad((midhour * 60 + eq + 4 * longitude - 60 * offset) / 4 - 180)
        elev = np.rad2deg(np.arcsin(np.clip(np.sin(lat) * np.sin(decl)
                                                + np.cos(lat) * np.cos(decl) * np.cos(hour_angle), -1, 1)))
        masks[j] = elev < threshold_degrees
    return masks[inverse]


def _features(dates: np.ndarray, capacities: np.ndarray) -> np.ndarray:
    d = pd.DatetimeIndex(dates.astype("datetime64[D]"))
    doy = d.dayofyear.to_numpy()
    weekday = d.dayofweek.to_numpy()
    return np.column_stack((np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
                            (weekday >= 5).astype(float), np.log(capacities))).astype("float32")


def prepare(raw_dir: Path, out_dir: Path, seed: int = 2026) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [_read(p) for p in _files(raw_dir)]
    ever_cl = set(pd.concat([f.loc[f["Consumption Category"].eq("CL"), "Customer"] for f in frames]).unique())
    all_x, all_ids, all_dates, all_caps = [], [], [], []
    audit = {"source_files": [p.name for p in _files(raw_dir)], "ever_cl_households": len(ever_cl), "years": []}
    capacities = {}
    for raw, path in zip(frames, _files(raw_dir)):
        idx = ["Customer", "date"]
        cats = {c: raw.loc[raw["Consumption Category"].eq(c)].set_index(idx) for c in ("GC", "CL", "GG")}
        gc, gg, cl = cats["GC"], cats["GG"], cats["CL"]
        if not gc.index.is_unique or not gg.index.is_unique or not cl.index.is_unique:
            raise ValueError("Non-unique category index")
        paired = gc.index.intersection(gg.index)
        gc, gg = gc.loc[paired], gg.loc[paired]
        customers = paired.get_level_values("Customer").to_numpy()
        dates = paired.get_level_values("date")
        has_cl = np.isin(customers, list(ever_cl))
        missing_cl = has_cl & ~paired.isin(cl.index)
        dst = np.array([_clock_change(d) for d in dates.unique()])
        dst_dates = set(dates.unique()[dst])
        dst_mask = dates.isin(dst_dates)
        common_cl = cl.reindex(paired)
        valid = gc["valid_row"].to_numpy() & gg["valid_row"].to_numpy()
        valid &= ~missing_cl
        cl_valid = common_cl["valid_row"].eq(True).to_numpy()
        valid &= np.where(has_cl, cl_valid, True)
        valid &= ~dst_mask
        chosen = np.flatnonzero(valid)
        load = gc[SLOTS].to_numpy()[chosen].copy()
        cl_values = common_cl[SLOTS].to_numpy()[chosen]
        load += np.nan_to_num(cl_values, nan=0.0)
        solar = gg[SLOTS].to_numpy()[chosen]
        x = np.stack((load, solar), axis=1).astype("float32")
        cs = customers[chosen]
        cap = gc["Generator Capacity"].to_numpy()[chosen]
        for house, value in zip(cs, cap):
            if house in capacities and not np.isclose(capacities[house], value):
                raise ValueError(f"Generator capacity changes for customer {house}")
            capacities[house] = float(value)
        all_x.append(x); all_ids.append(cs.astype("int16")); all_dates.append(dates.to_numpy()[chosen]); all_caps.append(cap)
        audit["years"].append({"file": path.name, "rows": len(raw), "gc_days": len(gc),
                               "gg_days": len(gg), "cl_days": len(cl), "paired_days": len(paired),
                               "missing_cl_for_cl_households": int(missing_cl.sum()),
                               "excluded_dst_after_cl": int((dst_mask & ~missing_cl).sum()),
                               "excluded_other_quality_days": int((~valid & ~missing_cl & ~dst_mask).sum()),
                               "usable_days": len(chosen)})
        print(path.name, audit["years"][-1], flush=True)
    del frames
    x = np.concatenate(all_x)
    ids = np.concatenate(all_ids)
    dates = np.concatenate(all_dates).astype("datetime64[D]")
    caps = np.concatenate(all_caps)
    rng = np.random.default_rng(seed)
    bycap = sorted(capacities, key=lambda h: (capacities[h], h))
    bins = np.array_split(bycap, 5)
    split_ids = {"train": [], "val": [], "test": []}
    for b in bins:
        b = rng.permutation(b)
        split_ids["train"].extend(int(i) for i in b[:36])
        split_ids["val"].extend(int(i) for i in b[36:48])
        split_ids["test"].extend(int(i) for i in b[48:])
    split = np.zeros(len(ids), dtype="uint8")
    split[np.isin(ids, split_ids["val"])] = 1
    split[np.isin(ids, split_ids["test"])] = 2
    tr = split == 0
    scale = np.quantile(x[tr], 0.95, axis=(0, 2)).astype("float32")
    cond = _features(dates, caps)
    mean, std = cond[tr].mean(0), cond[tr].std(0).clip(min=1e-6)
    cond = (cond - mean) / std
    night = night_mask(dates)
    audit.update({"total_usable_days": len(x), "split_households": {k: len(v) for k, v in split_ids.items()},
                  "scale_kwh": scale.tolist(), "measured_night_nonzero_gt_0p005_train": int((x[tr, 1][night[tr]] > 0.005).sum()),
                  "measured_night_gt_0p02_train": int((x[tr, 1][night[tr]] > 0.02).sum()),
                  "measured_night_max_kwh_train": float(x[tr, 1][night[tr]].max()),
                  "masked_slots_train": int(night[tr].sum()), "seed": seed})
    tmp = out_dir / "prepared.tmp"
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, x=x, customer=ids, date=dates,
                            capacity=caps, cond=cond, split=split, night=night, scale=scale)
    tmp.replace(out_dir / "prepared.npz")
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2))
    (out_dir / "split.json").write_text(json.dumps({"seed": seed, "households": split_ids,
                                                       "feature_mean": mean.tolist(), "feature_std": std.tolist()}, indent=2))
    return audit


def load_prepared(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
