"""Readers for the data formats the analysis can consume. None of the MD formats is present in the repository at
the time of writing (see inspect_data.py); the readers are provided so the pipeline runs unchanged once data exist."""
from pathlib import Path

import numpy as np


def read_lammps_dump(path, columns=None):
    """Yield frames of a LAMMPS text dump ('dump custom'). Each frame: dict(step, box (3,2), cols{name: array}).
    Rows are sorted by atom id when an 'id' column exists."""
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline())
            fh.readline()
            n = int(fh.readline())
            fh.readline()
            box = np.array([[float(x) for x in fh.readline().split()[:2]] for _ in range(3)])
            names = fh.readline().split()[2:]
            data = np.loadtxt([fh.readline() for _ in range(n)], ndmin=2)
            cols = {nm: data[:, k] for k, nm in enumerate(names) if columns is None or nm in columns or nm == "id"}
            if "id" in cols:
                order = np.argsort(cols["id"])
                cols = {k: v[order] for k, v in cols.items()}
            yield dict(step=step, box=box, cols=cols)


def stack_dump(path, pos=("x", "y", "z"), vel=("vx", "vy", "vz"), force=("fx", "fy", "fz")):
    """Whole dump -> arrays x (T,n,3), v, f (None if absent), mol, type, box lengths (T,3)."""
    X, V, F, Lb = [], [], [], []
    mol = typ = None
    for fr in read_lammps_dump(path):
        c = fr["cols"]
        X.append(np.stack([c[k] for k in pos], axis=1))
        V.append(np.stack([c[k] for k in vel], axis=1) if all(k in c for k in vel) else None)
        F.append(np.stack([c[k] for k in force], axis=1) if all(k in c for k in force) else None)
        Lb.append(fr["box"][:, 1] - fr["box"][:, 0])
        mol = c.get("mol", mol)
        typ = c.get("type", typ)
    arr = lambda a: None if a[0] is None else np.array(a)              # noqa: E731
    return dict(x=np.array(X), v=arr(V), f=arr(F), box=np.array(Lb), mol=mol, type=typ)


def load_npz_trajectory(path):
    """Generic npz trajectory with keys q|x, p|v, optional f, L, dt, m."""
    d = np.load(path)
    q = d["q"] if "q" in d.files else d["x"]
    m = float(d["m"]) if "m" in d.files else 1.0
    v = d["v"] if "v" in d.files else d["p"] / m
    return dict(q=q, v=v, p=v * m, f=d["f"] if "f" in d.files else None, L=float(d["L"]),
                dt=float(d["dt"]) if "dt" in d.files else np.nan, m=m)


def own_model_trajectories(project_dir, pattern="traj_N512_rank40_seed*.npz"):
    """Trajectories produced by this project's OWN CG model (ewald_longitudinal/test_dynamic_correlations.py).
    These are model output with a known input kernel - usable for known-answer pipeline tests only."""
    files = sorted((Path(project_dir) / "dynamics_results" / "correlations_raw").glob(pattern))
    return [load_npz_trajectory(f) | {"file": str(f)} for f in files]
