"""Recursive inventory of the repository: which MD / coarse-graining / memory data exist, and which analysis case
(1: friction or memory tensor available, 2: constrained-dynamics unresolved forces available, 3: ordinary MD
positions/velocities only, 0: no MD data at all) is scientifically possible."""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

PATTERNS = {
    "lammps_input": re.compile(r"(^in\..*|\.lmp$|\.in$|\.lammps$)", re.I),
    "md_trajectory": re.compile(r"\.(lammpstrj|dump|xyz|dcd|xtc|trr|gro|pdb|nc|h5md)$", re.I),
    "topology": re.compile(r"\.(data|psf|top|itp|prmtop|mol2)$", re.I),
    "hdf5": re.compile(r"\.(h5|hdf5)$", re.I),
    "numpy": re.compile(r"\.(npy|npz)$", re.I),
    "table": re.compile(r"\.(csv|tsv)$", re.I),
    "notebook": re.compile(r"\.ipynb$", re.I),
    "python": re.compile(r"\.py$", re.I),
    "document": re.compile(r"\.(md|pdf|txt)$", re.I),
}
KEYWORDS = {
    "mori_zwanzig": re.compile(r"mori|zwanzig|memory kernel|orthogonal dynamics", re.I),
    "constrained_dynamics": re.compile(r"constrained dynamics|unresolved force|fluctuating force|rattle|shake", re.I),
    "lammps": re.compile(r"\bLAMMPS\b|pair_style|fix nvt|read_data", re.I),
    "molecule_mapping": re.compile(r"\bmol(ecule)?[_ ]?id\b|atom-to-molecule|mol2cg", re.I),
    "friction_tensor": re.compile(r"friction tensor|Gamma_ij|pair friction|graph Laplacian", re.I),
    "vacf_vccf": re.compile(r"\bVACF\b|\bVCCF\b|velocity autocorrelation", re.I),
}
TRAJ_KEYS = {"positions": {"q", "x", "pos", "positions", "Q"}, "velocities": {"v", "vel", "velocities"},
             "momenta": {"p", "P", "momenta"}, "forces": {"f", "F", "force", "forces"},
             "molecule": {"mol", "molecule", "mol_id"}, "memory": {"K", "memory", "kernel", "Gamma"}}


def scan(root=ROOT):
    files = [f for f in root.rglob("*") if f.is_file() and ".git" not in f.parts and "__pycache__" not in f.parts
             and "analysis_friction_kernel" not in f.parts]
    inv = {k: [] for k in PATTERNS}
    hits = {k: [] for k in KEYWORDS}
    arrays = []
    for f in files:
        rel = str(f.relative_to(root))
        for k, pat in PATTERNS.items():
            if pat.search(f.name):
                inv[k].append(rel)
        if f.suffix in (".py", ".md", ".txt", ".ipynb", ".in") and f.stat().st_size < 5e6:
            txt = f.read_text(errors="ignore")
            for k, pat in KEYWORDS.items():
                if pat.search(txt):
                    hits[k].append(rel)
        if f.suffix == ".npz":
            try:
                with np.load(f) as d:
                    keys = set(d.files)
                    shapes = {k: list(d[k].shape) for k in list(d.files)[:12]}
            except Exception as e:                                   # noqa: BLE001
                keys, shapes = set(), {"error": str(e)}
            kinds = sorted(c for c, names in TRAJ_KEYS.items() if keys & names)
            arrays.append(dict(file=rel, size_mb=round(f.stat().st_size / 1e6, 2), kinds=kinds, shapes=shapes))
    return files, inv, hits, arrays


def classify(inv, arrays):
    """Decide what can be done. Only genuine MD data count; arrays produced by this project's own CG model
    (ewald_longitudinal/) are flagged as model output, never as MD."""
    md_traj = inv["md_trajectory"] + [a["file"] for a in arrays
                                      if "positions" in a["kinds"] and not a["file"].startswith("ewald_longitudinal/")]
    md_forces = [a["file"] for a in arrays if "forces" in a["kinds"] and not a["file"].startswith("ewald_longitudinal/")]
    md_memory = [a["file"] for a in arrays if "memory" in a["kinds"] and not a["file"].startswith("ewald_longitudinal/")]
    model_traj = [a["file"] for a in arrays if a["file"].startswith("ewald_longitudinal/") and "positions" in a["kinds"]]
    if md_memory:
        case = 1
    elif md_forces:
        case = 2
    elif md_traj:
        case = 3
    else:
        case = 0
    return dict(case=case, md_trajectories=md_traj, md_forces=md_forces, md_memory=md_memory,
                model_trajectories=model_traj)


def report(root=ROOT):
    files, inv, hits, arrays = scan(root)
    cls = classify(inv, arrays)
    out = dict(n_files=len(files), by_type={k: v for k, v in inv.items()}, keyword_hits=hits, npz=arrays,
               classification=cls)
    lines = [f"Scanned {len(files)} files under {root} (excluding .git and this directory)."]
    for k in ("lammps_input", "md_trajectory", "topology", "hdf5"):
        lines.append(f"  {k:16s}: {len(inv[k])} {inv[k][:5] if inv[k] else ''}")
    lines.append(f"  numpy archives  : {len(inv['numpy'])} (keys inspected below)")
    kinds = {}
    for a in arrays:
        kinds.setdefault(tuple(a["kinds"]), []).append(a["file"])
    for k, v in kinds.items():
        lines.append(f"      kinds {list(k) or ['observables only']}: {len(v)} files, e.g. {v[0]}")
    lines.append(f"  notebooks       : {inv['notebook']}")
    for k, v in hits.items():
        lines.append(f"  text mentions '{k}': {len(v)} {v[:4]}")
    lines.append(f"Classification: CASE {cls['case']} "
                 f"(MD trajectories {len(cls['md_trajectories'])}, MD forces {len(cls['md_forces'])}, "
                 f"MD memory/friction {len(cls['md_memory'])}; own-model trajectories {len(cls['model_trajectories'])})")
    return out, "\n".join(lines)


if __name__ == "__main__":
    out, txt = report()
    print(txt)
