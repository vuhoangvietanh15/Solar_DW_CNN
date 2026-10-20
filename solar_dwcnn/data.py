"""Strict CSV ingestion, day-level split, fixed training-only normalization."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone, timedelta
import numpy as np

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "Sun_Observering_Interferometry" / "Signal_Processing"
DEFAULT_TRAIN = ["2025_03_20", "2025_03_21", "2025_03_22"]
DEFAULT_VAL = ["2025_03_23"]
DEFAULT_TEST = ["2025_03_24"]


def stamp(path):
    # Files use a single decimal digit for fractions of seconds; local UTC+07.
    parts = path.stem.removesuffix("_PSD").split("_")
    if len(parts) != 7:
        raise ValueError(f"Unsupported timestamp filename: {path.name}")
    dt = datetime.strptime("_".join(parts[:6]), "%Y_%m_%d_%H_%M_%S")
    return dt.replace(tzinfo=timezone(timedelta(hours=7))).timestamp() + float("0." + parts[6])


def read_day(root, day):
    paths = sorted((Path(root) / day).glob("*_PSD.csv"), key=stamp)
    if not paths:
        raise ValueError(f"No *_PSD.csv in {Path(root) / day}")
    rows, hashes, reference = [], [], None
    for p in paths:
        a = np.loadtxt(p, delimiter=",", dtype=np.float64, ndmin=2)
        if a.shape[1] != 2 or a.shape[0] < 16 or a.shape[0] % 4 or not np.isfinite(a).all():
            raise ValueError(f"Invalid shape/nonfinite data in {p}: {a.shape}")
        f = a[:, 0]
        if not np.all(np.diff(f) > 0):
            raise ValueError(f"Frequency column must increase: {p}")
        if reference is None:
            reference = f
        if f.shape != reference.shape or not np.allclose(f, reference, rtol=0, atol=1e-6):
            raise ValueError(f"Frequency grid changed within {day}: {p}")
        rows.append(a[:, 1].astype(np.float32))
        hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
    return {"values": np.stack(rows), "frequency": reference,
            "files": np.array([str(p.resolve()) for p in paths]),
            "times": np.array([stamp(p) for p in paths]), "hashes": hashes}


def prepare(root, output, train_days, val_days, test_days):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new preparation directory: {output}")
    groups = {"train": train_days, "val": val_days, "test": test_days}
    flat = [d for ds in groups.values() for d in ds]
    if len(flat) != len(set(flat)) or any(not ds for ds in groups.values()):
        raise ValueError("Train/val/test days must be nonempty and disjoint")
    datasets, metadata, ref, seen = {}, {}, None, {}
    for split, days in groups.items():
        batches = []
        for day in days:
            b = read_day(root, day)
            if ref is None:
                ref = b['frequency']
            if b['frequency'].shape != ref.shape or not np.allclose(b['frequency'], ref, rtol=0, atol=1e-6):
                raise ValueError(f"Incompatible frequency grid in {day}; do not mix observing setups")
            for h in b['hashes']:
                if h in seen and seen[h] != split:
                    raise ValueError(f"Identical CSV content across splits: {day} and {seen[h]}")
                seen[h] = split
            metadata[day] = {"split": split, "count": len(b['values']),
                             "sha256": b['hashes']}
            batches.append(b)
        datasets[split] = {k: np.concatenate([b[k] for b in batches]) for k in ['values', 'files', 'times']}
    train = datasets['train']['values'].astype(np.float64)
    center, scale = float(train.mean()), float(train.std())
    if scale < 1e-8:
        raise ValueError("Degenerate training data")
    # Fit once on training days. Never use a masked target's per-frame statistics.
    meta = {"schema": 1, "source": str(Path(root).resolve()), "days": metadata,
            "normalization": {"center_db": center, "scale_db": scale},
            "length": len(ref), "frequency_axis_verified": False,
            "units": "recorded dB, not calibrated dBm or flux", "clean_target_available": False,
            "splits": groups}
    output.mkdir(parents=True)
    for split, b in datasets.items():
        np.savez_compressed(output / f"{split}.npz", values_db=b['values'],
                            frequency_mhz=ref, files=b['files'], timestamps=b['times'])
    (output / 'metadata.json').write_text(json.dumps(meta, indent=2))
    return meta


def normalize(x, meta):
    n = meta['normalization']
    return ((x - n['center_db']) / n['scale_db']).astype(np.float32)


def mask_values(values, phases, period=8):
    """Replace phase-selected bins with mean of nearest VISIBLE neighbors.

    With period >= 3, both neighbors are visible. Boundaries replicate the one
    visible neighbor; no wraparound. The held-out value cannot enter the input.
    """
    if period < 3:
        raise ValueError("period must be >=3")
    x = np.asarray(values, dtype=np.float32)
    phases = np.asarray(phases).reshape(-1, 1)
    if period > x.shape[1] or len(phases) != len(x) or np.any((phases < 0) | (phases >= period)):
        raise ValueError('Invalid mask period or phases')
    mask = (np.arange(x.shape[1])[None, :] % period == phases).astype(np.float32)
    left = np.concatenate([x[:, 1:2], x[:, :-1]], axis=1)
    right = np.concatenate([x[:, 1:], x[:, -2:-1]], axis=1)
    filled = np.where(mask.astype(bool), (left + right) * .5, x)
    return np.stack([filled, mask], -1), np.stack([x, mask], -1)
