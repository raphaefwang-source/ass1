"""Read-only audit of legacy inputs; writes only audit_initial.json beside this file.

Run before new numerical work. No operator, reference, sweep, or timing is rerun.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True
import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent
BENCH = VALIDATION.parent
REPO = BENCH.parents[1]
B2 = VALIDATION / "positive_mesh_b2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def number(row, key):
    val = row.get(key, "")
    return float(val) if val not in (None, "", "None") else None


def command(*args):
    return subprocess.run(args, cwd=REPO, check=True, text=True,
                          capture_output=True).stdout.strip()


def check_manifest(path):
    data = json.loads(path.read_text())
    bad = []
    for rel, expected in data.items():
        file = REPO / rel
        if not file.exists() or sha(file) != expected:
            bad.append(rel)
    return dict(manifest=str(path.relative_to(REPO)), checked=len(data),
                mismatches=bad)


def csv_summary(path):
    rows = read_csv(path)
    keys = ["sweep", "setting", "value", "config"]
    duplicates = len(rows) - len({tuple(r.get(k) for k in keys) for r in rows})
    return dict(path=str(path.relative_to(REPO)), sha256=sha(path), rows=len(rows),
                sweep_counts=dict(Counter(r.get("sweep", "") for r in rows)),
                unique_parameter_points=len({tuple(r.get(k) for k in keys[:3]) for r in rows}),
                n_configs=len({r.get("config") for r in rows}),
                duplicate_sweep_config_keys=duplicates,
                missing_operator_errors=sum(number(r, "E_G") is None for r in rows),
                mesh_invalid=sum(r.get("mesh_valid") == "False" for r in rows),
                min_translation_complement_eigenvalue=min(
                    (number(r, "lam_min_h") for r in rows if number(r, "lam_min_h") is not None),
                    default=None),
                max_matrixfree_dense_residual=max(
                    (number(r, "mf_vs_dense") for r in rows if number(r, "mf_vs_dense") is not None),
                    default=None))


def audit_references():
    out = []
    for row in read_csv(B2 / "results/references.csv"):
        filename = row["file"]
        paths = [BENCH / "raw" / sub / filename
                 for sub in ("exp_refs_c1", "positive_mesh_b2_refs")]
        existing = [p for p in paths if p.exists()]
        if not existing:
            out.append(dict(file=filename, missing=True))
            continue
        path = existing[0]
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        actual = sha(path)
        base_path = VALIDATION / "configs" / f'{row["family"]}_N256_s{row["seed"]}.npz'
        with np.load(base_path) as data:
            q = np.mod(data["q"].astype(float), float(data["L"]))
            L0 = float(data["L"])
        rho = float(row["rho"])
        L = (256 / rho) ** (1 / 3)
        if rho != .75:
            q = np.mod(q * (L / L0), L)
        pos_hash = hashlib.sha256(np.ascontiguousarray(
            np.concatenate([q.ravel(), [L]]), dtype=np.float64).tobytes()).hexdigest()[:12]
        meta_hash = meta.get("position_hash", meta.get("hash"))
        expected_name = (f'{row["config"]}_kap{float(row["kappa"]):g}_c1_'
                         f'L{L:.10f}_{pos_hash}.npz')
        out.append(dict(file=filename, path=str(path.relative_to(REPO)), sha256=actual,
                        sha256_matches_record=actual == row["sha256"],
                        metadata_sha256=sha(meta_path),
                        position_hash=pos_hash, metadata_position_hash=meta_hash,
                        position_hash_matches=(pos_hash == row["position_hash"]
                                               and meta_hash in (None, pos_hash)
                                               and filename == expected_name),
                        metadata_physical_keys_present="kappa" in meta and meta_hash is not None,
                        physical_keys_match=(float(meta.get("kappa", row["kappa"])) == float(row["kappa"])
                                             and float(meta.get("c", 1)) == 1
                                             and meta.get("config", row["config"]) == row["config"]
                                             and abs(float(meta.get("L", L)) - L) < 1e-12),
                        source=meta["source"], cert_bound_rel=float(meta["cert_bound_rel"]),
                        nbytes=path.stat().st_size))
    return out


def main():
    refs = audit_references()
    mismatches = [r["file"] for r in refs if r.get("missing") or not all(
        r.get(k, False) for k in ("sha256_matches_record", "position_hash_matches", "physical_keys_match"))]
    tracked = command("git", "ls-files").splitlines()
    initial_hashes = {rel: sha(REPO / rel) for rel in tracked if (REPO / rel).is_file()}
    manifest = check_manifest(B2 / "results/historical_artifact_hashes.json")
    b2_source = json.loads((B2 / "results/final_audit.json").read_text())["source_sha256"]
    b2_source_mismatches = [rel for rel, expected in b2_source.items()
                            if sha(REPO / rel) != expected]
    legacy = json.loads((B2 / "results/original_reference_cache_hashes.json").read_text())
    legacy_ref_bad = [row["file"] for row in legacy
                      if sha(BENCH / "raw/exp_refs_c1" / row["file"]) != row["sha256"]]
    result = dict(
        timestamp=datetime.now(ZoneInfo("America/Detroit")).isoformat(),
        purpose="Read-only snapshot and validation of pre-existing experiment artifacts",
        git=dict(branch=command("git", "branch", "--show-current"),
                 commit=command("git", "rev-parse", "HEAD"),
                 status_porcelain=command("git", "status", "--porcelain")),
        environment=dict(python=platform.python_version(), executable=sys.executable,
                         numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                         cpu_count=os.cpu_count(),
                         memory_bytes=int(command("sysctl", "-n", "hw.memsize"))),
        original_artifact_manifest=manifest,
        b2_source_manifest=dict(checked=len(b2_source), mismatches=b2_source_mismatches),
        original_reference_manifest=dict(checked=len(legacy), mismatches=legacy_ref_bad),
        old_sweeps=[csv_summary(VALIDATION / "accuracy_sweeps/results/sweeps.csv"),
                    csv_summary(B2 / "results/sweeps.csv")],
        references=dict(count=len(refs), mismatches=mismatches,
                        original_count=sum("/exp_refs_c1/" in r.get("path", "") for r in refs),
                        b2_isolated_count=sum("/positive_mesh_b2_refs/" in r.get("path", "") for r in refs),
                        total_bytes=sum(r.get("nbytes", 0) for r in refs),
                        max_cert_bound_rel=max(r.get("cert_bound_rel", 0) for r in refs),
                        rows=refs),
        b2_best=json.loads((B2 / "results/best_configuration.json").read_text()),
        original_best=json.loads((VALIDATION / "theory_guided_joint_optimization/results/best_configuration.json")
                                  .read_text())["best_free"],
        b2_search=dict(screen_rows=len(read_csv(B2 / "results/search.csv")),
                       validation_rows=len(read_csv(B2 / "results/validation.csv")),
                       validation_summary_rows=len(read_csv(B2 / "results/validation_summary.csv"))),
        reuse_policy=[
            "Reuse legacy accuracy CSV and 72 direct-reference NPZ files read-only after hash checks.",
            "Store every new source, checkpoint, plot, report and timing pass in deconv_psd_comparison.",
            "Original sF and B2 sT are distinct cutoff parameters; label their axes and units explicitly.",
            "Use identical Gaussian probes, configurations, physical amplitude c=1 and spectral-norm error.",
            "Unify mesh-only convergence definition against G_D minus far-short when fitting orders.",
            "Measure all three methods serially together after numerical workers exit; retain old timings.",
            "Record parameter-table and configuration setup separately from complete cached matvec.",
            "Bound mesh caches on the 8 GiB host; high M full-tensor eigensolves and copies multiply memory.",
            "Original influence Gr requires IFFT/h^3 to get physical grid tensors; B2 Gr=rFFT(table).",
        ],
        initial_tracked_file_sha256=initial_hashes,
    )
    assert not (mismatches or manifest["mismatches"] or b2_source_mismatches or legacy_ref_bad)
    target = HERE / "audit_initial.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(audit=str(target), references=len(refs),
                          reused_sweep_rows=[r["rows"] for r in result["old_sweeps"]],
                          original_files_checked=manifest["checked"],
                          b2_sources_checked=len(b2_source), all_hashes_match=True), indent=2))


if __name__ == "__main__":
    main()
