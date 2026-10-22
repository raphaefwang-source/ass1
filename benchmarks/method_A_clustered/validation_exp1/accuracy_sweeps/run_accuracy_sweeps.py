#!/usr/bin/env python3
"""
Experiment 1 (new): accuracy of the Candidate 1 (theta = 0, existing positive sharp Laplace) Ewald-PPPM operator
for the c = 1 screened longitudinal kernel, with six one-at-a-time parameter sweeps around a common baseline.

Model, splitting, references: see ../ewald_exp_lib.py. Baseline: N = 256, rho = 0.75, kappa = 0.7, xi = 0.85,
s_R = xi r_c = 6, s_F = k_c/(2 xi) = 6 (k_c snapped to the largest reciprocal shell), M = 80, p = 9. Secondary
moderate-accuracy setting: s_R = s_F = 4, M = 40, p = 7 (sweeps M, p, rho, kappa), so the parameter trends are
visible above the truncation level and away from the reference / round-off floor.
Configurations: uniform, moderate, strong x seeds 1, 2, 3 (configs/*_N256_s*.npz); other densities by isotropic
rescaling of the same positions.

Sweeps (only the named parameter changes):
    sR     s_R = 2.0 ... 6.5 (0.25)                 baseline
    sF     s_F = 2.0 ... 6.5 (0.25), shell-snapped  baseline
    M      FFT size, smooth M, Nyquist/eta/shell-validity checked            baseline, moderate
    p      B-spline order 3 ... 9                                           baseline, moderate
    rho    0.3, 0.5, 0.75, 1.0 (L = (N/rho)^{1/3}, positions rescaled)      baseline, moderate
    kappa  0.2, 0.5, 0.7, 1.0, 2.0 (c = 1)                                  baseline, moderate

Stages:  smoke | configs | refs | sweeps | timing | fits | plots | tables | all (= configs refs sweeps timing fits
plots tables). Results in results/, figures in figures/, dense references in ../../raw/exp_refs_c1/ (git-ignored).

    python3 run_accuracy_sweeps.py smoke
    python3 run_accuracy_sweeps.py all --workers 4
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import ewald_exp_lib as E                # noqa: E402  (thread settings before numpy)
import common as C                       # noqa: E402
import numpy as np                       # noqa: E402

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
S_GRID = [float(s) for s in np.round(np.arange(2.0, 6.5001, 0.25), 3)]
M_BASE = (24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80)
M_MOD = (16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 54, 60, 64, 72, 80)
P_LIST = tuple(range(3, 10))
RHO_LIST = (0.3, 0.5, 0.75, 1.0)
KAPPA_LIST = (0.2, 0.5, 0.7, 1.0, 2.0)
SETTINGS = {"baseline": E.BASE, "moderate": E.MODERATE}


def sweep_points():
    """(sweep, setting, value, parameter dict) for every one-at-a-time point."""
    pts = []
    for s in S_GRID:
        pts.append(("sR", "baseline", s, dict(E.BASE, sR=s)))
        pts.append(("sF", "baseline", s, dict(E.BASE, sF=s)))
    for st in ("baseline", "moderate"):
        B = SETTINGS[st]
        for M in (M_BASE if st == "baseline" else M_MOD):
            pts.append(("M", st, M, dict(B, M=M)))
        for p in P_LIST:
            pts.append(("p", st, p, dict(B, p=p)))
        for r in RHO_LIST:
            pts.append(("rho", st, r, dict(B, rho=r)))
        for k in KAPPA_LIST:
            pts.append(("kappa", st, k, dict(B, kappa=k)))
    return pts


def ref_jobs():
    """Every (family, seed, rho, kappa) needed by Experiment 1 (Experiment 2 uses a subset)."""
    jobs = {(f, s, 0.75, k) for f in E.FAMS for s in E.SEEDS for k in KAPPA_LIST}
    jobs |= {(f, s, r, 0.7) for f in E.FAMS for s in E.SEEDS for r in RHO_LIST}
    return sorted(jobs, key=lambda j: (j[3], -j[2], j[0], j[1]))     # slowest (small kappa) first


# ============================================================================ references
def _ref_worker(job):
    fam, seed, rho, kappa = job
    cfg = E.config(fam, seed, rho)
    if E.ref_path(cfg, kappa).exists():
        return None
    _, _, rec, conv = E.compute_reference(cfg, kappa, say=lambda m: C.log(m))
    return rec, conv


def stage_configs(args):
    rows = []
    for fam in E.FAMS:
        for s in E.SEEDS:
            for r in RHO_LIST:
                cfg = E.config(fam, s, r)
                rows.append(E.config_record(cfg))
    C.write_csv(RESULTS / "configurations.csv", rows)
    C.write_json(RESULTS / "configurations.json", dict(
        rows=rows, note="rho != 0.75: the committed rho = 0.75 positions (configs/*_N256_s*.npz) rescaled "
                        "isotropically by L(rho)/L(0.75); geometry in box units identical, d_min scales with L.",
        provenance=E.provenance()))
    C.log(f"configs: {len(rows)} (d_min {min(r['d_min'] for r in rows):.4f} ... {max(r['d_min'] for r in rows):.4f})")


def stage_refs(args, jobs=None):
    jobs = jobs or ref_jobs()
    todo = [j for j in jobs if not E.ref_path(E.config(*j[:3]), j[3]).exists()]
    C.log(f"refs: {len(jobs)} needed, {len(todo)} to compute, workers {args.workers}")
    old = C.read_json(RESULTS / "references.json", {"rows": {}, "conv": {}})
    with Pool(args.workers) as pool:
        for out in pool.imap_unordered(_ref_worker, todo):
            if out is None:
                continue
            rec, conv = out
            old["rows"][rec["key"]] = rec
            old["conv"][rec["key"]] = conv
            C.write_json(RESULTS / "references.json", dict(old, provenance=E.provenance()))
    missing = [j for j in jobs if E.ref_key(E.config(*j[:3]), j[3]) not in old["rows"]]
    if missing:
        C.log(f"refs: WARNING {len(missing)} cached matrices without metadata in references.json: {missing[:3]}")
    rows = sorted(old["rows"].values(), key=lambda r: (r["kappa"], r["rho"], r["config"]))
    C.write_csv(RESULTS / "references.csv", rows)
    conv_rows = [dict(key=k, config=old["rows"][k]["config"], kappa=old["rows"][k]["kappa"],
                      rho=old["rows"][k]["rho"], R_D=old["rows"][k]["R_D"], **c)
                 for k, cs in old["conv"].items() for c in cs]
    C.write_csv(RESULTS / "reference_convergence.csv", conv_rows)
    C.write_json(RESULTS / "references.json", dict(old, provenance=E.provenance()))


# ============================================================================ sweeps
def _sweep_worker(task):
    (fam, seed, rho, kappa), items, opts = task
    cfg = E.config(fam, seed, rho)
    G, GLF = E.load_reference(cfg, kappa)
    probes, spec = E.probes_for(G)
    meshes = E.MeshCache()
    done, rows = {}, []
    for sweep, st, val, par in items:
        key = tuple(sorted(par.items()))
        if key not in done:
            done[key] = E.evaluate(cfg, G, GLF, probes, par, meshes, timing=opts["timing"],
                                   meshfree=opts["meshfree"])
        rows.append(dict(sweep=sweep, setting=st, value=val, **done[key]))
    C.log(f"sweeps {cfg['name']} kappa {kappa:g}: {len(rows)} rows ({len(done)} distinct)")
    return rows


def sweep_tasks(points, fams=E.FAMS, seeds=E.SEEDS):
    groups = {}
    for sw, st, val, par in points:
        for f in fams:
            for s in seeds:
                groups.setdefault((f, s, par["rho"], par["kappa"]), []).append((sw, st, val, par))
    return groups


def stage_sweeps(args, points=None, out=None, fams=E.FAMS, seeds=E.SEEDS, meshfree=True):
    points = points or sweep_points()
    out = Path(out) if out is not None else RESULTS / "sweeps.csv"
    groups = sweep_tasks(points, fams, seeds)
    opts = dict(timing=True, meshfree=meshfree)
    t0 = time.perf_counter()
    rows = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(_sweep_worker, [(k, v, opts) for k, v in sorted(groups.items())]):
            rows += r
    rows.sort(key=lambda r: (r["sweep"], r["setting"], r["family"], r["seed"], r["value"]))
    C.write_csv(out, rows)
    C.log(f"sweeps: {len(rows)} rows in {time.perf_counter() - t0:.0f} s -> {out}")
    summarize(rows, out.with_name(out.stem + "_summary.csv"))
    return rows


def summarize(rows, out):
    keys = ("E_G", "E_GX", "E_G_vsLF", "E_GX_vsLF", "E_GX_gauss_max", "E_GX_slow_max", "E_GX_fast", "ref_floor_G",
            "ref_floor_GX", "E_G_meshfree", "E_G_mesh_only", "sym_resid", "mom_opnorm_max", "lam_min_h",
            "lam_min_ref", "mf_vs_dense", "t_setup_s", "t_matvec_s", "sF_eff", "n_images")
    grp = {}
    for r in rows:
        if not r.get("mesh_valid"):
            continue
        grp.setdefault((r["sweep"], r["setting"], r["family"], r["value"]), []).append(r)
    out_rows = []
    for (sw, st, fam, val), rs in sorted(grp.items()):
        rec = dict(sweep=sw, setting=st, family=fam, value=val, n_seeds=len(rs))
        for k in keys:
            v = np.array([r[k] for r in rs if r.get(k) is not None], float)
            if v.size:
                rec[f"{k}_median"], rec[f"{k}_min"], rec[f"{k}_max"] = float(np.median(v)), float(v.min()), float(v.max())
        lam = np.array([(r["lam_min_h"] - r["lam_min_ref"]) / r["lam_min_ref"] for r in rs])
        rec["lam_min_rel_dev_maxabs"] = float(np.abs(lam).max())
        out_rows.append(rec)
    C.write_csv(out, out_rows)


# ============================================================================ timing (serial, quiet machine)
def stage_timing(args):
    rows = []
    for st, par in SETTINGS.items():
        for fam in E.FAMS:
            for s in E.SEEDS:
                cfg = E.config(fam, s, par["rho"])
                G, GLF = E.load_reference(cfg, par["kappa"])
                pr, _ = E.probes_for(G)
                meshes = E.MeshCache()
                for theta in (0.0, 0.5):
                    r = E.evaluate(cfg, G, GLF, pr, dict(par, theta=theta), meshes, dense=False)
                    rows.append(dict(setting=st, **r))
        C.log(f"timing {st}: done")
    C.write_csv(RESULTS / "timing_serial.csv", rows)


# ============================================================================ fits
def stage_fits(args):
    rows = E.read_csv(RESULTS / "sweeps.csv")
    fits = []
    for fam in E.FAMS + ("all",):
        for sw in ("sR", "sF"):
            sel = [r for r in rows if r["sweep"] == sw and (fam == "all" or r["family"] == fam) and r.get("E_G")]
            for metric in ("E_G", "E_GX"):
                xs = np.array([r["sF_eff"] if sw == "sF" else r["sR"] for r in sel])
                ys = np.array([r[metric] for r in sel])
                floor = np.median([r["ref_floor_G"] for r in sel])
                keep = ys > 1e3 * max(floor, 1e-15)
                if keep.sum() < 4:
                    continue
                A = np.stack([np.ones(keep.sum()), -xs[keep] ** 2, np.log(xs[keep])], 1)
                coef, *_ = np.linalg.lstsq(A, np.log(ys[keep]), rcond=None)
                A2 = A[:, :2]
                coef2, *_ = np.linalg.lstsq(A2, np.log(ys[keep]), rcond=None)
                fits.append(dict(family=fam, sweep=sw, metric=metric, model="ln E = a - c s^2", a=coef2[0], c=coef2[1],
                                 c_with_log_term=coef[1], b_log_term=coef[2], s_lo=float(xs[keep].min()),
                                 s_hi=float(xs[keep].max()), n=int(keep.sum()), floor=float(floor)))
        for st in ("baseline", "moderate"):
            sel = [r for r in rows if r["sweep"] == "M" and r["setting"] == st and (fam == "all" or r["family"] == fam)
                   and r.get("E_G_mesh_only")]
            for metric in ("E_G_mesh_only", "E_GX"):
                xs = np.array([r["M"] for r in sel], float)
                ys = np.array([r[metric] for r in sel])
                sat = np.median([r[metric] for r in sel if r["M"] == max(xs)])
                keep = ys > 30 * max(sat, 1e-15)
                if keep.sum() < 4:
                    continue
                b, a = np.polyfit(np.log(xs[keep]), np.log(ys[keep]), 1)
                fits.append(dict(family=fam, sweep="M", setting=st, metric=metric, model="ln E = a + b ln M", a=a, b=b,
                                 M_lo=int(xs[keep].min()), M_hi=int(xs[keep].max()), n=int(keep.sum()), floor=float(sat)))
    C.write_csv(RESULTS / "fits.csv", fits)
    C.log(f"fits: {len(fits)} rows")


# ============================================================================ plots
def stage_plots(args):
    """Paper-style figures (../paper_figures.py) from the saved CSV files."""
    import paper_figures
    paper_figures.exp1(RESULTS, FIGURES, E.read_csv)
    C.log("plots written")


# ============================================================================ tables
def g3(v):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.2e}"


def stage_tables(args):
    rows = E.read_csv(RESULTS / "sweeps.csv")
    refs = E.read_csv(RESULTS / "references.csv")
    out = ["# Experiment 1 tables (generated by run_accuracy_sweeps.py tables)", ""]
    out += ["## References", "", "| kappa | rho | R_D | images | cert. bound | ‖Γ_D(0.95R_D)−Γ_D‖ | LF 1e-10 | LF 1e-12 | "
            "LF 1e-14 | t_direct [s] |", "|---|---|---|---|---|---|---|---|---|---|"]
    for kap in KAPPA_LIST:
        for rho in RHO_LIST:
            sel = [r for r in refs if r["kappa"] == kap and r["rho"] == rho]
            if not sel:
                continue
            mx = lambda k: max(r[k] for r in sel)
            out.append(f"| {kap:g} | {rho:g} | {sel[0]['R_D']:g}–{mx('R_D'):g} | {mx('n_images_direct'):.2e} | "
                       f"{g3(mx('cert_bound_rel'))} | {g3(mx('conv_R_0p95_rel'))} | {g3(mx('LF_1e-10_vs_D_rel'))} | "
                       f"{g3(mx('LF_1e-12_vs_D_rel'))} | {g3(mx('LF_1e-14_vs_D_rel'))} | {mx('t_direct_s'):.0f} |")
    out += ["", "(maximum over the 9 configurations; relative spectral norms)", ""]
    for sw, st in (("sR", "baseline"), ("sF", "baseline"), ("M", "baseline"), ("M", "moderate"), ("p", "baseline"),
                   ("p", "moderate"), ("rho", "baseline"), ("rho", "moderate"), ("kappa", "baseline"),
                   ("kappa", "moderate")):
        sel = [r for r in rows if r["sweep"] == sw and r["setting"] == st]
        out += [f"## Sweep {sw} ({st})", "",
                "| value | family | E_Γ med [min, max] | E_ΓX med [min, max] | E_ΓX slow max | ref floor | "
                "λmin_h/λmin_ref−1 max | mom. | sym. | t_matvec [ms] |", "|---|---|---|---|---|---|---|---|---|---|"]
        for val in sorted({r["value"] for r in sel}):
            for fam in E.FAMS:
                rs = [r for r in sel if r["value"] == val and r["family"] == fam]
                if not rs:
                    continue
                if not rs[0].get("mesh_valid"):
                    out.append(f"| {val:g} | {fam} | invalid mesh: {rs[0].get('mesh_reason')} | | | | | | | |")
                    continue
                e = np.array([r["E_G"] for r in rs])
                ex = np.array([r["E_GX"] for r in rs])
                lam = max(abs(r["lam_min_h"] / r["lam_min_ref"] - 1) for r in rs)
                vs = f"{val:g}" + (f" ({rs[0]['sF_eff']:.3f})" if sw == "sF" else "")
                out.append(f"| {vs} | {fam} | {np.median(e):.2e} [{e.min():.1e}, {e.max():.1e}] | "
                           f"{np.median(ex):.2e} [{ex.min():.1e}, {ex.max():.1e}] | "
                           f"{max(r['E_GX_slow_max'] for r in rs):.1e} | {max(r['ref_floor_G'] for r in rs):.1e} | "
                           f"{lam:.1e} | {max(r['mom_opnorm_max'] for r in rs):.1e} | {max(r['sym_resid'] for r in rs):.1e} | "
                           f"{1e3 * np.median([r['t_matvec_s'] for r in rs]):.1f} |")
        out.append("")
    (RESULTS / "tables.md").write_text("\n".join(out) + "\n")
    C.log("tables written")


# ============================================================================ smoke
def stage_smoke(args):
    """Fast end-to-end check (one kappa = 2 reference, coarse sweeps on one configuration, small-system dense /
    matrix-free agreement, c = 1 audit). Writes results/smoke/ only."""
    out = RESULTS / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(amplitude=E.amplitude_checks())
    cfg = E.config("strong", 1, 0.75)
    kap = 2.0
    if not E.ref_path(cfg, kap).exists():
        E.compute_reference(cfg, kap, say=C.log)
    G, GLF = E.load_reference(cfg, kap)
    # DirectPairs (the existing unsplit direct operator) vs the annulus accumulation used for the cache
    import direct_periodic as DP
    with np.load(E.ref_path(cfg, kap)) as d:
        R_D = float(d["conv"][-1, 0])
    t0 = time.perf_counter()
    Dp = DP.DirectPairs(cfg["q"], cfg["L"], R_D, g=E.g_phys(kap))
    GDp = Dp.dense()
    rec["DirectPairs_vs_annulus_rel"] = E.norm2_sym(GDp - G)[0] / E.norm2_sym(G)[0]
    rec["DirectPairs_t_s"] = time.perf_counter() - t0
    rng = np.random.default_rng(5)
    X = rng.standard_normal((cfg["N"], 3))
    rec["DirectPairs_action_vs_dense_rel"] = float(np.linalg.norm(Dp.apply(X).ravel() - G @ X.ravel())
                                                   / np.linalg.norm(G @ X.ravel()))
    # small system: dense-by-basis of the matrix-free operator vs the independent dense assembly
    with np.load(HERE.parent / "configs" / "uniform_N128_s1.npz") as d:
        q, L = np.mod(d["q"], float(d["L"])), float(d["L"])
    small = {}
    for theta in (0.0, 0.5):
        K = E.kernel(0.7, 0.85, theta)
        rc, kc, _ = E.cutoffs(dict(E.MODERATE), L)
        mesh, info = E.check_mesh(L, K, kc, 32, 7)
        op = E.ThetaOp(q, L, K, rc, mesh)
        Gb = op.dense_by_basis()
        Gd = E.real_dense(op) + E.lap_full(E.mode_pair_matrix_fast(mesh, op.x))
        small[f"theta{theta:g}"] = dict(basis_vs_dense_rel=E.norm2_sym(Gb - Gd)[0] / E.norm2_sym(Gd)[0],
                                        basis_sym=float(np.linalg.norm(Gb - Gb.T) / np.linalg.norm(Gb)), N=len(q))
    rec["small_system_dense_vs_matrix_free"] = small
    pts = [(sw, st, v, dict(par, kappa=kap)) for sw, st, v, par in sweep_points()
           if sw in ("sR", "sF", "M", "p") and st == "baseline" and v in (2.0, 4.0, 6.0, 24, 48, 80, 3, 6, 9)]
    meshes = E.MeshCache()
    pr, _ = E.probes_for(G)
    rows = [dict(sweep=sw, setting=st, value=v, **E.evaluate(cfg, G, GLF, pr, par, meshes, meshfree=True))
            for sw, st, v, par in pts]
    C.write_csv(out / "smoke_sweeps.csv", rows)
    rec["n_sweep_rows"] = len(rows)
    rec["max_mf_vs_dense"] = max(r["mf_vs_dense"] for r in rows if r.get("mf_vs_dense") is not None)
    C.write_json(out / "smoke_checks.json", dict(rec, provenance=E.provenance()))
    for r in rows:
        C.log(f"smoke {r['sweep']}={r['value']}: E_G {r.get('E_G', float('nan')):.2e} E_GX {r.get('E_GX', float('nan')):.2e} "
              f"mf/dense {r.get('mf_vs_dense', float('nan')):.1e}")
    C.log(f"smoke: DirectPairs vs annulus {rec['DirectPairs_vs_annulus_rel']:.1e}; small-system basis vs dense "
          f"{max(v['basis_vs_dense_rel'] for v in small.values()):.1e}; c audit ok")
    assert rec["DirectPairs_vs_annulus_rel"] < 1e-14
    assert max(v["basis_vs_dense_rel"] for v in small.values()) < 1e-12
    for v in rec["amplitude"].values():
        assert v["RadialEwald_amp"] == 1.0 and v["theta_full_max_rel"] == 0.0 and v["direct_g_max_rel"] == 0.0
        assert v["lattice_friction_g_max_rel"] < 1e-14


STAGES = dict(smoke=stage_smoke, configs=stage_configs, refs=stage_refs, sweeps=stage_sweeps, timing=stage_timing,
              fits=stage_fits, plots=stage_plots, tables=stage_tables)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=list(STAGES) + ["all"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    E.log_cmd(RESULTS)
    stages = ["configs", "refs", "sweeps", "timing", "fits", "plots", "tables"] if "all" in args.stages else args.stages
    for st in stages:
        t0 = time.perf_counter()
        C.log(f"=== stage {st}")
        STAGES[st](args)
        C.log(f"=== stage {st} done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
