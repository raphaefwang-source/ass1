#!/usr/bin/env python3
"""
Time-step comparison of four LJ-A dynamical observables at the lj075 state point (rho 0.75, kT 1, N 256, law A),
re-using the existing coupled trajectories (no new long trajectories):
  VACF; VCCF grouped by the initial pair distance; longitudinal / transverse current correlations C_L(k,t), C_T(k,t);
  distinct van Hove function g_d(r,t).

Data: lj075_results/raw/dt_study/
  A_rep*_L4_T10.npz         16 coupled 4-level chains (dt 0.00125, 0.0025, 0.005, 0.01), t_end 10, frames every step
  A_rep*_L2_T100_F0.01.npz  11 coupled 2-level chains (dt 0.005, 0.01), t_end 100, frames every 0.01
Both: unwrapped positions, velocity v = p/m after the final half kick, float32 frames; ranks 12/5 (coupling 4),
PPPM p 8 (configuration lj_rho0.75_kT1.0_costopt), not the later recommended 14/5.

Stages (outputs in lj075_results/observable_dt/, per-level caches in lj075_results/raw/observable_dt_cache/):
    python3 lj075_observable_dt.py inventory      # data manifest and missing items
    python3 lj075_observable_dt.py protocol       # write protocol.json (refuses to overwrite an existing one)
    python3 lj075_observable_dt.py compute        # per replica and level observables (cached)
    python3 lj075_observable_dt.py analyze        # paired differences, bounds, verdicts, tables, figures
    python3 lj075_observable_dt.py timing         # measured production-step cost at dt 0.005 and 0.01
The protocol fixes every analysis choice before any step-size difference of these observables is computed.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lj075_common as C  # noqa: E402

RAWDIR = C.RAW / "dt_study"
OUT = C.OUT / "observable_dt"
CACHE = C.RAW / "observable_dt_cache"
PROTOCOL_F = OUT / "protocol.json"
CODE_VERSION = "1"
N = 256
L = C.box_length(N, 0.75)
LEVELS4 = (0.00125, 0.0025, 0.005, 0.01)

PROTOCOL = dict(
    version=1,
    scope=dict(potential="lj", law="A", N=N, density=0.75, kT=1.0, L=L,
               config_of_trajectories="lj_rho0.75_kT1.0_costopt (Lanczos ranks noise/damp 12/5, coupling 4, PPPM p 8); "
                                      "NOT the later recommended 14/5",
               note="supplementary analysis; the earlier dt-study verdicts (REPORT.md section 3) are kept unchanged"),
    comparisons=dict(
        primary=dict(data="16 four-level chains (A_rep*_L4_T10)", reference=0.00125,
                     pairs=[[0.01, 0.00125], [0.005, 0.00125]],
                     reference_sensitivity=[[0.0025, 0.00125]],
                     note="0.00125 is a numerical reference, not the exact solution"),
        secondary=dict(data="11 two-level chains (A_rep*_L2_T100_F0.01)", pairs=[[0.01, 0.005]],
                       note="longer trajectories, no 0.00125 level: tells whether 0.01 differs from 0.005 only")),
    grid=dict(dtau=0.01, tau_max=2.0, note="common physical-time grid; finer levels are read at exact integer strides "
                                           "of their stored frames (no interpolation of positions or velocities)"),
    origins=dict(primary=dict(start=1.0, stop=8.0, step=0.05),
                 secondary=dict(start=10.0, stop=98.0, step=0.25),
                 note="identical physical origin times for all levels of a chain; origins are averaged within a "
                      "replica and are NOT statistical units"),
    observables=dict(
        vacf=dict(definition="Z(t) = <v_i(t0).v_i(t0+t)> / <|v_i(t0)|^2>, average over particles and origins "
                             "(project convention, prl_toy_models.compute_observables)",
                  also="unnormalized <v_i(t0).v_i(t0+t)> (C(0) = 3 kT_kin/m) for scalars"),
        vccf=dict(definition="C_b(t) = <v_i(t0).v_j(t0+t)> over ordered pairs i != j with minimum-image r_ij(t0) in bin "
                             "b, divided by <|v(t0)|^2> (project convention vccf_normalized)",
                  bins=[0.0, 1.59, 2.55, L / 2],
                  bins_note="first and second minima of the reference RDF at this state point (1.59, 2.55; "
                            "lj075_results/dt_study.json reference curve), last bin to L/2 (minimum image)",
                  zero_momentum_background="P_total = 0 exactly, so sum over all j != i of v_i(t0).v_j(t0+t) = "
                                           "-v_i(t0).v_i(t0+t): a uniform background of -Z(t)/(N-1) per pair "
                                           "(reported; not subtracted)"),
        currents=dict(definition="j(k,t) = (1/N) sum_i v_i e^{i k.r_i}; C_L(k,t) = Re<j_L(t0+t) j_L*(t0)>, "
                                 "C_T(k,t) = Re<j_T(t0+t).j_T*(t0)>/2, averaged over the modes of a shell and the "
                                 "origins, divided by the same at t = 0 (project convention hydro_L / hydro_T)",
                      shells={"k1": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              "k2": [[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]],
                              "k4": [[2, 0, 0], [0, 2, 0], [0, 0, 2]]},
                      k_values={"k1": 2 * np.pi / L, "k2": 2 * np.pi / L * np.sqrt(2), "k4": 4 * np.pi / L},
                      note="k = 0 is identically zero (P_total = 0) and excluded"),
        van_hove=dict(definition="g_d(r,t) = V / (n_origins N (N-1)) * sum over ordered pairs i != j of "
                                 "delta(r - |r_j(t0+t) - r_i(t0)|_min-image) / (4 pi r^2 dr)",
                      times=[0.0, 0.1, 0.5, 1.0, 2.0], r_bins=dict(n=60, r_max=L / 2),
                      finite_N="N (N-1) ordered pairs: g_d(r,0) = g(r) and g_d -> 1 for uncorrelated pairs; "
                               "the total-momentum constraint adds no shift to positions-based g_d")),
    windows=dict(time={"W1": [0.0, 0.2], "W2": [0.2, 0.5], "W3": [0.5, 1.0], "W4": [1.0, 2.0]},
                 time_note="half-open (a, b]; t = 0 is excluded for the normalized curves (identical by construction)",
                 radius={"R1": [0.0, 1.59], "R2": [1.59, L / 2]}),
    metrics=dict(
        signal="deviation from the uncorrelated value: Z, C_b, C_L, C_T -> 0; g_d -> 1",
        E_abs="sup over the window of |mean paired difference| in the units of the displayed quantity "
              "(VACF / C_L / C_T: fraction of the t = 0 value; VCCF: fraction of <|v|^2>; g_d: g units)",
        E_sig="E_abs / S_W with S_W = sup over the same window of |reference signal| (weak signals are judged "
              "against their own size, not against C(0))",
        E_glob="E_abs / S_glob with S_glob = sup over the whole t in (0, 2] (or all r at fixed t) of |reference "
               "signal| (plot-level view; secondary)",
        bounds="simultaneous 95% band over the grid points of the window: studentized multiplier (Rademacher) "
               "bootstrap over replicas, B = 4000, seed 20261010; the critical value is multiplied by "
               "t_{0.975,n-1}/z_{0.975} for small-n conservativeness. upper = sup(|mean| + c se), "
               "lower = sup max(|mean| - c se, 0). Pointwise 95% t-intervals are shown in figures only; they are NOT "
               "used to claim that a whole curve passes",
        zero_crossings="no pointwise relative error is computed anywhere; relative measures use window sup-norms"),
    tolerance=dict(
        curves="PASS if the simultaneous upper bound of E_sig <= 0.01 and the reference signal is resolved; FAIL if "
               "the simultaneous lower bound of E_sig > 0.01; otherwise INCONCLUSIVE",
        signal_resolved="t_{0.975,n-1} se_ref(t*) / S_W <= 0.1 at the window maximum t* of the reference signal; "
                        "otherwise INCONCLUSIVE (signal not resolved)",
        glob_secondary="the same rule with E_glob (reported as 'plot-level'; weaker; not used for weak features)",
        scalars="paired Student-t 95% CI of the relative difference (x_dt - x_ref)/|mean x_ref|: PASS if inside "
                "+-1%, FAIL if entirely outside, else INCONCLUSIVE; an interval containing 0 is not equivalence",
        resolved_deviation="a deviation is 'resolved' if the simultaneous lower bound of E_abs > 0 (curves) or the "
                           "CI excludes 0 (scalars), whatever its size"),
    scalars=dict(
        vacf=["C0 = <|v|^2> (relative; equals the T_kin ratio)", "Z_min (value at the first minimum)",
              "t_min (quadratic interpolation)", "t_zero (first zero crossing, linear interpolation)",
              "D2 = (1/3) int_0^2 <v(0).v(t)> dt (trapezoid on the 0.01 grid)"],
        vccf=["per bin: extremum of C_b on (0, 2] (signed value; max or min chosen from the reference mean) and "
              "its time; int_0^2 C_b dt"],
        currents=["per shell: C_L first minimum value and time, C_L first zero; C_T time to 1/2 and int_0^2 C_T dt"],
        van_hove=["per time t > 0: first-peak height and position of g_d(r,t) (quadratic interpolation)"]),
    statistics=dict(unit="independent replica (16 primary, 11 secondary); frames, particles, origins and distance "
                         "bins are never treated as independent samples",
                    pairing="differences are formed per replica between levels of the same chain (same initial "
                            "state and noise stream), also after the paths have separated",
                    multiplicity="bands are simultaneous within a window only; there are many windows, observables "
                                 "and comparisons, so isolated verdicts near the threshold are expected by chance"),
    decision=dict(display_support="dt = 0.01 supports showing an observable in a window if the primary comparison "
                                  "0.01 vs 0.00125 PASSes there; the reference is considered adequate there only if "
                                  "0.0025 vs 0.00125 also PASSes; the secondary 0.01 vs 0.005 is a consistency check",
                  long_time_D="the long-time D plateau (REPORT.md section 5) is not addressed by t <= 2 correlation "
                              "functions and stays unconfirmed"),
)


def proto_hash(p):
    return hashlib.sha256(json.dumps(p, sort_keys=True, default=float).encode()).hexdigest()[:16]


def load_protocol():
    if not PROTOCOL_F.exists():
        raise SystemExit("protocol.json missing: run the 'protocol' stage first")
    p = json.loads(PROTOCOL_F.read_text())
    if p["hash"] != proto_hash(p["protocol"]):
        raise SystemExit("protocol.json was modified after it was written")
    return p["protocol"], p["hash"]


# ----------------------------------------------------------------------------------------------------- data
def chains():
    out = []
    for f in sorted(RAWDIR.glob("A_rep*_L4_T10.json")) + sorted(RAWDIR.glob("A_rep*_L2_T100_F0.01.json")):
        m = json.loads(f.read_text())
        if m.get("complete"):
            out.append((m, f.with_suffix(".npz"), "primary" if "_L4_" in f.name else "secondary"))
    return out


def frame_dt(meta, dt):
    fe = meta.get("frame_every", "every step")
    return dt if fe in (None, "every step") else float(fe)


def stage_inventory():
    rows, missing = [], []
    for m, f, kind in chains():
        with np.load(f) as z:
            lev = {}
            for dt in m["levels"]:
                Q, V, S = z[f"Q_{dt}"], z[f"V_{dt}"], z[f"S_{dt}"]
                fd = frame_dt(m, dt)
                jumps = bool(np.any(np.abs(np.diff(Q[:, :, :], axis=0)) > L / 2))
                lev[str(dt)] = dict(frames=int(len(Q)), frame_interval=fd, t_last=float((len(Q) - 1) * fd),
                                    dtype=str(Q.dtype), position_jumps_over_L2=jumps,
                                    max_abs_P_total=float(S[:, 4].max()),
                                    T_kin_mean=float(S[S[:, 0] >= 1.0, 1].mean()))
        rows.append(dict(file=f.name, kind=kind, replica=m["replica"], init_state=m["init_state"], levels=m["levels"],
                         t_end=m["t_end"], frame_every=m.get("frame_every", "every step (field absent: early run)"),
                         seed=m["seed"], ranks=[m["resolved"]["rank_noise"], m["resolved"]["rank_damp"]],
                         rank_couple=m["rank_couple"], pppm=m["resolved"]["pppm"], config=m["resolved"]["config"],
                         levels_detail=lev, sha256_16=C.sha256_file(f)[:16] if hasattr(C, "sha256_file") else None))
    prim = [r for r in rows if r["kind"] == "primary"]
    sec = [r for r in rows if r["kind"] == "secondary"]
    if len(prim) != 16:
        missing.append(f"expected 16 primary chains, found {len(prim)}")
    if len(sec) != 11:
        missing.append(f"expected 11 secondary chains, found {len(sec)}")
    missing += [
        "secondary chains have no 0.0025 / 0.00125 levels: they compare 0.01 with 0.005 only",
        "no frame finer than the 0.01 grid is used in the comparison (the 0.01 level has no sub-step data); "
        "features faster than 0.01 are not resolved by any level on the common grid",
        "smallest wave number is 2 pi / L = 0.899 (N = 256); smaller k are not available without a larger box",
        "primary chains are 10 time units long: correlation lags up to 2 with origins in [1, 8] only",
        "trajectories were run with ranks 12/5 (coupling 4), not with the recommended production ranks 14/5"]
    jumps = [r["file"] for r in rows for v in r["levels_detail"].values() if v["position_jumps_over_L2"]]
    res = dict(chains=rows, n_primary=len(prim), n_secondary=len(sec),
               positions="unwrapped" if not jumps else f"WRAPPED or jumping in {sorted(set(jumps))}",
               velocity_definition="v = p/m after the final half kick of the BAOAB step (full-step velocity), "
                                   "lj075_dt_study.py record()",
               missing_or_limited=missing, provenance=C.provenance())
    OUT.mkdir(parents=True, exist_ok=True)
    C.write_json(OUT / "inventory.json", res)
    print(f"inventory: {len(prim)} primary, {len(sec)} secondary chains; positions {res['positions']}")
    for x in missing:
        print("  -", x)


def stage_protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    if PROTOCOL_F.exists():
        raise SystemExit(f"{PROTOCOL_F} exists; it is not overwritten (delete it deliberately to change the protocol)")
    p = json.loads(json.dumps(PROTOCOL, default=float))
    C.write_json(PROTOCOL_F, dict(protocol=p, hash=proto_hash(p), written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                  written_before_any_difference_was_computed=True, provenance=C.provenance()))
    print(f"wrote {PROTOCOL_F} (hash {proto_hash(p)})")


# ------------------------------------------------------------------------------------------- observables
def min_image(d):
    return d - L * np.round(d / L)


def level_observables(Q, V, fd, origins, P):
    """All observables of one level on the common grid. Q, V: stored frames (unwrapped), fd: frame interval."""
    g = P["grid"]
    stride = int(round(g["dtau"] / fd))
    if abs(stride * fd - g["dtau"]) > 1e-12:
        raise ValueError("frame interval does not divide the grid spacing")
    nl = int(round(g["tau_max"] / g["dtau"])) + 1
    o_idx = np.rint(origins / fd).astype(int)
    lag_idx = np.arange(nl) * stride
    if o_idx.max() + lag_idx[-1] >= len(Q):
        raise ValueError("trajectory too short for the requested origins and lags")
    if np.any(np.abs(o_idx * fd - origins) > 1e-9):
        raise ValueError("origins are not on stored frames")
    # frames needed: union of origin + lag frames, all on the 0.01 grid
    need = np.unique((o_idx[:, None] + lag_idx[None, :]).ravel())
    pos = {f: i for i, f in enumerate(need)}
    Qs = Q[need].astype(float)
    Vs = V[need].astype(float)
    oi = np.array([pos[f] for f in o_idx])
    li = np.array([[pos[o + l] for l in lag_idx] for o in o_idx])            # (n_orig, nl)
    # VACF
    num = np.zeros(nl)
    v2 = 0.0
    for a, row in zip(oi, li):
        num += np.einsum("nc,tnc->t", Vs[a], Vs[row]) / N
        v2 += np.sum(Vs[a] ** 2) / N
    num /= len(oi)
    v2 /= len(oi)
    vacf_un = num                                                      # <v(t0).v(t0+t)> per particle
    # VCCF
    edges = np.asarray(P["observables"]["vccf"]["bins"], float)
    nb = len(edges) - 1
    cross = np.zeros((nb, nl))
    counts = np.zeros(nb)
    eye = np.eye(N, dtype=bool)
    for a, row in zip(oi, li):
        r0 = np.linalg.norm(min_image(Qs[a][:, None, :] - Qs[a][None, :, :]), axis=-1)
        lab = np.searchsorted(edges, r0, side="right") - 1
        lab[eye] = -1
        lab[r0 >= edges[-1]] = -1
        for b in range(nb):
            M = (lab == b)                                             # M[i, j]: pair (i at t0, j at t0 + t)
            counts[b] += M.sum()
            W = M.T.astype(float) @ Vs[a]                             # W_j = sum_i M_ij v_i(t0)
            cross[b] += np.einsum("jc,tjc->t", W, Vs[row])
    vccf = cross / counts[:, None]
    # currents
    shells = P["observables"]["currents"]["shells"]
    cur = {}
    for name, modes in shells.items():
        kv = 2 * np.pi / L * np.asarray(modes, float)
        kh = kv / np.linalg.norm(kv, axis=1)[:, None]
        ph = np.exp(1j * np.einsum("fnc,kc->fnk", Qs, kv))            # (frames, N, modes)
        j = np.einsum("fnc,fnk->fkc", Vs, ph) / N                      # (frames, modes, 3)
        jl = np.einsum("fkc,kc->fk", j, kh)
        jt = j - jl[..., None] * kh[None]
        cl = np.zeros(nl)
        ct = np.zeros(nl)
        for a, row in zip(oi, li):
            cl += np.mean((jl[row] * jl[a].conj()[None]).real, axis=1)
            ct += np.mean(np.sum((jt[row] * jt[a].conj()[None]).real, axis=-1) / 2, axis=1)
        cur[name] = (cl / len(oi), ct / len(oi))
    # van Hove
    vh = P["observables"]["van_hove"]
    rb = np.linspace(0, vh["r_bins"]["r_max"], vh["r_bins"]["n"] + 1)
    shell_vol = 4 * np.pi / 3 * np.diff(rb ** 3)
    gd = []
    for tt in vh["times"]:
        k = int(round(tt / g["dtau"]))
        hist = np.zeros(len(rb) - 1)
        for a, row in zip(oi, li):
            d = np.linalg.norm(min_image(Qs[row[k]][None, :, :] - Qs[a][:, None, :]), axis=-1)[~eye]
            hist += np.histogram(d, bins=rb)[0]
        gd.append(hist * L ** 3 / (len(oi) * N * (N - 1) * shell_vol))
    return dict(tau=np.arange(nl) * g["dtau"], vacf_un=vacf_un, v2=np.array(v2), vacf=vacf_un / v2,
                vccf=vccf / v2, vccf_counts=counts, vccf_background=-(vacf_un / v2) / (N - 1),
                **{f"CL_{k}": v[0] / v[0][0] for k, v in cur.items()},
                **{f"CT_{k}": v[1] / v[1][0] for k, v in cur.items()},
                **{f"CL0_{k}": np.array(v[0][0]) for k, v in cur.items()},
                **{f"CT0_{k}": np.array(v[1][0]) for k, v in cur.items()},
                gd=np.array(gd), r_edges=rb, r_centers=0.5 * (rb[1:] + rb[:-1]))


def origins_for(P, kind):
    o = P["origins"][kind]
    return np.round(np.arange(o["start"], o["stop"] + 1e-9, o["step"]), 10)


def stage_compute(P, ph):
    CACHE.mkdir(parents=True, exist_ok=True)
    cpu0, wall0 = C.cpu_seconds(), time.perf_counter()
    done = 0
    for m, f, kind in chains():
        origins = origins_for(P, kind)
        for dt in m["levels"]:
            out = CACHE / f"{f.stem}_dt{dt:g}.npz"
            key = f"{ph}:{CODE_VERSION}:{f.stat().st_size}:{int(f.stat().st_mtime)}"
            if out.exists():
                with np.load(out) as z:
                    if str(z["key"]) == key:
                        continue
            t0 = time.perf_counter()
            with np.load(f) as z:
                obs = level_observables(z[f"Q_{dt}"], z[f"V_{dt}"], frame_dt(m, dt), origins, P)
            np.savez(out, key=np.array(key), **obs)
            done += 1
            print(f"{f.stem} dt {dt}: {time.perf_counter() - t0:.1f} s", flush=True)
    t = dict(stage="compute", new_level_files=done, cpu_s=C.cpu_seconds() - cpu0, wall_s=time.perf_counter() - wall0)
    tf = OUT / "postprocessing_time.json"
    db = json.loads(tf.read_text()) if tf.exists() else {}
    db.setdefault("runs", []).append(t)
    C.write_json(tf, db)
    print(t)


# ---------------------------------------------------------------------------------------------- analysis
def window_mask(tau, w, exclude_zero=True):
    a, b = w
    m = (tau > a + 1e-12) & (tau <= b + 1e-12)
    if not exclude_zero and a <= 0:
        m |= np.abs(tau) < 1e-12
    return m


def sim_crit(D, rng, B=4000):
    """Studentized multiplier-bootstrap critical value for sup over columns of |mean| / se (rows = replicas)."""
    n = len(D)
    m = D.mean(0)
    se = D.std(0, ddof=1) / np.sqrt(n)
    ok = se > 0
    R = (D - m)[:, ok]
    eps = rng.choice([-1.0, 1.0], size=(B, n))
    T = np.abs(eps @ R / n) / se[ok]
    c = float(np.quantile(T.max(axis=1), 0.95)) if ok.any() else 0.0
    return c * stats.t.ppf(0.975, n - 1) / stats.norm.ppf(0.975)


def curve_cell(Dw, refw, rng, tol=0.01):
    """Dw: paired differences (replicas x points), refw: reference curves (replicas x points), signal deviations."""
    n = len(Dw)
    m = Dw.mean(0)
    se = Dw.std(0, ddof=1) / np.sqrt(n)
    c = sim_crit(Dw, rng)
    up = float(np.max(np.abs(m) + c * se))
    lo = float(np.max(np.maximum(np.abs(m) - c * se, 0)))
    rm = refw.mean(0)
    rse = refw.std(0, ddof=1) / np.sqrt(n)
    i = int(np.argmax(np.abs(rm)))
    S = float(np.abs(rm[i]))
    rel_unc = float(stats.t.ppf(0.975, n - 1) * rse[i] / S) if S > 0 else np.inf
    resolved = rel_unc <= 0.1
    est = float(np.max(np.abs(m)))
    out = dict(E_abs=est, E_abs_upper=up, E_abs_lower=lo, crit=c, S_W=S, signal_rel_unc=rel_unc,
               signal_resolved=bool(resolved), E_sig=est / S if S > 0 else np.inf,
               E_sig_upper=up / S if S > 0 else np.inf, E_sig_lower=lo / S if S > 0 else np.inf,
               deviation_resolved=bool(lo > 0))
    if out["E_sig_lower"] > tol:
        v = "FAIL"
    elif out["E_sig_upper"] <= tol and resolved:
        v = "PASS"
    else:
        v = "INCONCLUSIVE" + ("" if resolved else " (signal not resolved)")
    out["verdict"] = v
    return out


def glob_verdict(cell, S_glob, tol=0.01):
    up, lo = cell["E_abs_upper"] / S_glob, cell["E_abs_lower"] / S_glob
    return dict(E_glob=cell["E_abs"] / S_glob, E_glob_upper=up, E_glob_lower=lo, S_glob=S_glob,
                verdict="PASS" if up <= tol else ("FAIL" if lo > tol else "INCONCLUSIVE"))


def interp_extremum(t, y, sign):
    """Extremum (sign=+1 max, -1 min) on (0, t_max] with quadratic interpolation; returns (value, time)."""
    s = sign * y
    i = int(np.argmax(s[1:])) + 1
    if 1 <= i < len(y) - 1:
        y0, y1, y2 = y[i - 1], y[i], y[i + 1]
        den = y0 - 2 * y1 + y2
        x = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        x = float(np.clip(x, -1, 1))
        return float(y1 - 0.25 * (y0 - y2) * x), float(t[i] + x * (t[1] - t[0]))
    return float(y[i]), float(t[i])


def first_crossing(t, y, level=0.0):
    s = y - level
    k = np.where((s[:-1] > 0) & (s[1:] <= 0))[0]
    if len(k) == 0:
        return np.nan
    k = k[0]
    return float(t[k] + s[k] / (s[k] - s[k + 1]) * (t[k + 1] - t[k]))


def first_min_after_zero(t, y):
    """First local minimum of a curve that starts at 1 and decays (C_L): value and time (quadratic)."""
    d = np.diff(y)
    k = np.where((d[:-1] < 0) & (d[1:] >= 0))[0]
    if len(k) == 0:
        return np.nan, np.nan
    i = int(k[0] + 1)
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2 * y1 + y2
    x = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if den != 0 else 0.0
    return float(y1 - 0.25 * (y0 - y2) * x), float(t[i] + x * (t[1] - t[0]))


def scalars_of(o, P, vccf_signs):
    t = o["tau"]
    s = {}
    s["vacf_C0"] = float(o["v2"])
    s["vacf_Zmin"], s["vacf_tmin"] = interp_extremum(t, o["vacf"], -1)
    s["vacf_tzero"] = first_crossing(t, o["vacf"])
    s["vacf_D2"] = float(np.trapezoid(o["vacf_un"], t) / 3)
    for b in range(o["vccf"].shape[0]):
        y = o["vccf"][b]
        s[f"vccf_b{b + 1}_ext"], s[f"vccf_b{b + 1}_text"] = interp_extremum(t, y, vccf_signs[b])
        s[f"vccf_b{b + 1}_int"] = float(np.trapezoid(y, t))
    for k in P["observables"]["currents"]["shells"]:
        s[f"CL_{k}_min"], s[f"CL_{k}_tmin"] = first_min_after_zero(t, o[f"CL_{k}"])
        s[f"CL_{k}_tzero"] = first_crossing(t, o[f"CL_{k}"])
        s[f"CT_{k}_thalf"] = first_crossing(t, o[f"CT_{k}"], 0.5)
        s[f"CT_{k}_int"] = float(np.trapezoid(o[f"CT_{k}"], t))
    rc = o["r_centers"]
    for j, tt in enumerate(P["observables"]["van_hove"]["times"]):
        if tt == 0:
            continue
        g = o["gd"][j]
        i = int(np.argmax(g))
        y0, y1, y2 = g[i - 1], g[i], g[i + 1]
        den = y0 - 2 * y1 + y2
        x = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if den != 0 else 0.0
        s[f"vh_t{tt:g}_peak"] = float(y1 - 0.25 * (y0 - y2) * x)
        s[f"vh_t{tt:g}_rpeak"] = float(rc[i] + x * (rc[1] - rc[0]))
    return s


def load_level(f, dt):
    with np.load(CACHE / f"{f.stem}_dt{dt:g}.npz") as z:
        return {k: z[k] for k in z.files if k != "key"}


def curve_specs(P):
    specs = [("vacf", "VACF Z(t)", lambda o: o["vacf"])]
    for b in range(len(P["observables"]["vccf"]["bins"]) - 1):
        specs.append((f"vccf_b{b + 1}", f"VCCF bin {b + 1}", lambda o, b=b: o["vccf"][b]))
    for k in P["observables"]["currents"]["shells"]:
        specs.append((f"CL_{k}", f"C_L {k}", lambda o, k=k: o[f"CL_{k}"]))
        specs.append((f"CT_{k}", f"C_T {k}", lambda o, k=k: o[f"CT_{k}"]))
    return specs


def analyze_group(P, kind, pairs, rng):
    ch = [(m, f) for m, f, k in chains() if k == kind]
    reps = [m["replica"] for m, _ in ch]
    levels = sorted({dt for p in pairs for dt in p})
    obs = {dt: [load_level(f, dt) for _, f in ch] for dt in levels}
    tau = obs[levels[0]][0]["tau"]
    res = dict(n_replicas=len(ch), replicas=reps, curves={}, van_hove={}, scalars={}, background={})
    # curves
    for key, label, get in curve_specs(P):
        Y = {dt: np.array([get(o) for o in obs[dt]]) for dt in levels}
        res["curves"][key] = {}
        for a, b in pairs:
            D = Y[a] - Y[b]
            S_glob = float(np.max(np.abs(Y[b].mean(0)[1:])))
            cells = {}
            for wn, w in P["windows"]["time"].items():
                msk = window_mask(tau, w)
                cell = curve_cell(D[:, msk], Y[b][:, msk], rng)
                cell["plot_level"] = glob_verdict(cell, S_glob)
                cells[wn] = cell
            res["curves"][key][f"{a}_vs_{b}"] = cells
        res["curves"][key]["_mean"] = {str(dt): Y[dt].mean(0).tolist() for dt in levels}
        res["curves"][key]["_diff"] = {f"{a}_vs_{b}": dict(mean=(Y[a] - Y[b]).mean(0).tolist(),
                                                           se=((Y[a] - Y[b]).std(0, ddof=1) / np.sqrt(len(ch))).tolist(),
                                                           crit_full=sim_crit((Y[a] - Y[b])[:, 1:], rng))
                                       for a, b in pairs}
    # zero-momentum background of VCCF (reference level)
    ref = pairs[0][1]
    bg = np.array([o["vccf_background"] for o in obs[ref]]).mean(0)
    res["background"] = dict(vccf_background_ref=bg.tolist(),
                             ratio_to_signal={f"b{b + 1}": float(np.max(np.abs(bg[1:])) /
                                                                 np.max(np.abs(np.array([o["vccf"][b] for o in obs[ref]]).mean(0)[1:])))
                                              for b in range(len(P["observables"]["vccf"]["bins"]) - 1)},
                             pair_origin_counts=np.array([o["vccf_counts"] for o in obs[ref]]).mean(0).tolist())
    # van Hove
    rc = obs[levels[0]][0]["r_centers"]
    for j, tt in enumerate(P["observables"]["van_hove"]["times"]):
        G = {dt: np.array([o["gd"][j] for o in obs[dt]]) for dt in levels}
        entry = {}
        for a, b in pairs:
            D = G[a] - G[b]
            sigref = G[b] - 1.0
            S_glob = float(np.max(np.abs(sigref.mean(0))))
            cells = {}
            for rn, w in P["windows"]["radius"].items():
                msk = (rc >= w[0]) & (rc < w[1])
                cell = curve_cell(D[:, msk], sigref[:, msk], rng)
                cell["plot_level"] = glob_verdict(cell, S_glob)
                cells[rn] = cell
            entry[f"{a}_vs_{b}"] = cells
        entry["_mean"] = {str(dt): G[dt].mean(0).tolist() for dt in levels}
        entry["_diff"] = {f"{a}_vs_{b}": dict(mean=(G[a] - G[b]).mean(0).tolist(),
                                              se=((G[a] - G[b]).std(0, ddof=1) / np.sqrt(len(ch))).tolist())
                          for a, b in pairs}
        res["van_hove"][f"t{tt:g}"] = entry
    res["r_centers"] = rc.tolist()
    res["tau"] = tau.tolist()
    # scalars
    # sign of the VCCF extremum per bin from the reference mean (identical for all levels and replicas)
    vm = np.array([o["vccf"] for o in obs[ref]]).mean(0)
    signs = [1 if np.max(vm[b, 1:]) >= -np.min(vm[b, 1:]) else -1 for b in range(vm.shape[0])]
    res["vccf_extremum_sign"] = signs
    sc = {dt: [scalars_of(o, P, signs) for o in obs[dt]] for dt in levels}
    per_rep = {str(r): {str(dt): sc[dt][i] for dt in levels} for i, r in enumerate(reps)}
    res["per_replica_scalars"] = per_rep
    for a, b in pairs:
        tab = {}
        for name in sc[b][0]:
            xa = np.array([s[name] for s in sc[a]])
            xb = np.array([s[name] for s in sc[b]])
            ok = np.isfinite(xa) & np.isfinite(xb)
            if ok.sum() < 3:
                tab[name] = dict(n=int(ok.sum()), verdict="INCONCLUSIVE (undefined in too many replicas)")
                continue
            ref_mean = float(np.mean(xb[ok]))
            d = (xa[ok] - xb[ok]) / abs(ref_mean)
            m, lo, hi = C.t_ci(d)
            v = "PASS" if (-0.01 <= lo and hi <= 0.01) else ("FAIL" if (lo > 0.01 or hi < -0.01) else "INCONCLUSIVE")
            tab[name] = dict(n=int(ok.sum()), ref_mean=ref_mean, dt_mean=float(np.mean(xa[ok])), rel_diff=m,
                             ci95=[lo, hi], verdict=v, resolved=bool(lo > 0 or hi < 0))
        res["scalars"][f"{a}_vs_{b}"] = tab
    return res


def figures(P, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"0.00125": "#222222", "0.0025": "#1baf7a", "0.005": "#2a78d6", "0.01": "#eb6834"}
    pr = R["primary"]
    tau = np.array(pr["tau"])
    groups = [("vacf", ["vacf"]), ("vccf", [k for k in pr["curves"] if k.startswith("vccf")]),
              ("currents", [k for k in pr["curves"] if k.startswith(("CL_", "CT_"))])]
    for name, keys in groups:
        fig, ax = plt.subplots(2, len(keys), figsize=(4.2 * len(keys), 7), squeeze=False)
        for j, k in enumerate(keys):
            cur = pr["curves"][k]
            for dt, y in cur["_mean"].items():
                ax[0, j].plot(tau, y, color=cols[dt], lw=1.2, label=f"dt {dt}")
            if k.startswith("vccf"):
                ax[0, j].plot(tau, pr["background"]["vccf_background_ref"], color="#999999", ls=":", lw=1,
                              label="-Z/(N-1) background")
            ax[0, j].axhline(0, color="#bbbbbb", lw=0.6)
            ax[0, j].set_title(k, fontsize=9)
            for c, d in cur["_diff"].items():
                m, se = np.array(d["mean"]), np.array(d["se"])
                dt = c.split("_vs_")[0]
                hp = stats.t.ppf(0.975, pr["n_replicas"] - 1) * se
                ax[1, j].plot(tau, m, color=cols[dt], lw=1.0, label=f"{c.replace('_vs_', ' - ')}")
                ax[1, j].fill_between(tau, m - hp, m + hp, color=cols[dt], alpha=0.18, lw=0)
                ax[1, j].plot(tau, m + d["crit_full"] * se, color=cols[dt], lw=0.6, ls="--")
                ax[1, j].plot(tau, m - d["crit_full"] * se, color=cols[dt], lw=0.6, ls="--")
            ax[1, j].axhline(0, color="#bbbbbb", lw=0.6)
            ax[1, j].set_xlabel("t")
        ax[0, 0].legend(fontsize=7)
        ax[1, 0].legend(fontsize=7)
        ax[1, 0].set_ylabel("paired difference (shaded: pointwise 95% t;\ndashed: simultaneous 95% over (0,2])")
        fig.suptitle(f"lj075 law A, {name}: dt comparison, {pr['n_replicas']} paired replicas (ranks 12/5)", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"{name}_dt.png", dpi=110)
        plt.close(fig)
    rc = np.array(pr["r_centers"])
    vh = pr["van_hove"]
    ts = [k for k in vh]
    fig, ax = plt.subplots(2, len(ts), figsize=(3.6 * len(ts), 6.5), squeeze=False)
    for j, t in enumerate(ts):
        for dt, y in vh[t]["_mean"].items():
            ax[0, j].plot(rc, y, color=cols[dt], lw=1.1, label=f"dt {dt}")
        ax[0, j].axhline(1, color="#bbbbbb", lw=0.6)
        ax[0, j].set_title(f"g_d(r, {t[1:]})", fontsize=9)
        for c, d in vh[t]["_diff"].items():
            m, se = np.array(d["mean"]), np.array(d["se"])
            dt = c.split("_vs_")[0]
            hp = stats.t.ppf(0.975, pr["n_replicas"] - 1) * se
            ax[1, j].plot(rc, m, color=cols[dt], lw=1.0, label=c.replace("_vs_", " - "))
            ax[1, j].fill_between(rc, m - hp, m + hp, color=cols[dt], alpha=0.18, lw=0)
        ax[1, j].axhline(0, color="#bbbbbb", lw=0.6)
        ax[1, j].set_xlabel("r")
    ax[0, 0].legend(fontsize=7)
    ax[1, 0].legend(fontsize=7)
    ax[1, 0].set_ylabel("paired difference (pointwise 95% t)")
    fig.suptitle(f"lj075 law A, distinct van Hove: dt comparison, {pr['n_replicas']} paired replicas", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "van_hove_dt.png", dpi=110)
    plt.close(fig)


def conclusions(P, R):
    lines = ["# lj075 law A: dt comparison of VACF, VCCF, C_L / C_T and distinct van Hove (generated; see protocol.json)",
             "",
             "Verdict of the window error E_sig (sup |paired difference| / sup |reference signal| in the window, "
             "simultaneous 95% bound, tolerance 1%). P = PASS, F = FAIL, I = INCONCLUSIVE, I* = reference signal not "
             "resolved. In brackets: estimate [lower, upper] of E_sig in %. 'pl' = plot-level verdict (E_glob, scale = "
             "global signal maximum; secondary).", ""]
    for kind in ("primary", "secondary"):
        g = R[kind]
        pairs = [c for c in g["curves"]["vacf"] if not c.startswith("_")]
        lines.append(f"## {kind} ({g['n_replicas']} replicas)")
        lines.append("")
        head = "| observable | window | " + " | ".join(p.replace("_vs_", " vs ") for p in pairs) + " |"
        lines += [head, "|" + "---|" * (2 + len(pairs))]
        for key, cells in g["curves"].items():
            for wn, w in P["windows"]["time"].items():
                row = [key, f"{wn} ({w[0]:g}, {w[1]:g}]"]
                for p in pairs:
                    c = cells[p][wn]
                    v = c["verdict"]
                    tag = "P" if v == "PASS" else "F" if v == "FAIL" else ("I*" if "signal" in v else "I")
                    row.append(f"{tag} {100 * c['E_sig']:.2f} [{100 * c['E_sig_lower']:.2f}, "
                               f"{100 * c['E_sig_upper']:.2f}]; pl {c['plot_level']['verdict'][0]}")
                lines.append("| " + " | ".join(row) + " |")
        for t, e in g["van_hove"].items():
            for rn, w in P["windows"]["radius"].items():
                row = [f"g_d {t}", f"{rn} r in [{w[0]:.2f}, {w[1]:.2f})"]
                for p in pairs:
                    c = e[p][rn]
                    v = c["verdict"]
                    tag = "P" if v == "PASS" else "F" if v == "FAIL" else ("I*" if "signal" in v else "I")
                    row.append(f"{tag} {100 * c['E_sig']:.2f} [{100 * c['E_sig_lower']:.2f}, "
                               f"{100 * c['E_sig_upper']:.2f}]; pl {c['plot_level']['verdict'][0]}")
                lines.append("| " + " | ".join(row) + " |")
        lines += ["", f"### scalars ({kind}): relative paired difference, 95% t-CI, tolerance 1%", ""]
        sp = list(g["scalars"])
        lines += ["| scalar | reference mean | " + " | ".join(p.replace("_vs_", " vs ") for p in sp) + " |",
                  "|" + "---|" * (2 + len(sp))]
        for name in g["scalars"][sp[0]]:
            row = [name, f"{g['scalars'][sp[0]][name].get('ref_mean', float('nan')):.5g}"]
            for p in sp:
                x = g["scalars"][p][name]
                if "ci95" in x:
                    row.append(f"{100 * x['rel_diff']:+.2f}% [{100 * x['ci95'][0]:+.2f}, {100 * x['ci95'][1]:+.2f}] "
                               f"{x['verdict'][0]}")
                else:
                    row.append(x["verdict"])
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    (OUT / "conclusions_table.md").write_text("\n".join(lines) + "\n")


def stage_analyze(P, ph):
    cpu0, wall0 = C.cpu_seconds(), time.perf_counter()
    rng = np.random.default_rng(20261010)
    pp = P["comparisons"]["primary"]
    prim_pairs = [tuple(x) for x in pp["pairs"] + pp["reference_sensitivity"]]
    R = dict(protocol_hash=ph, provenance=C.provenance(),
             primary=analyze_group(P, "primary", prim_pairs, rng),
             secondary=analyze_group(P, "secondary", [tuple(x) for x in P["comparisons"]["secondary"]["pairs"]], rng))
    C.write_json(OUT / "results.json", R)
    # per-replica observables (compact): all curves per replica and level
    arrays = {}
    for m, f, kind in chains():
        for dt in m["levels"]:
            o = load_level(f, dt)
            for k in ("vacf", "vccf", "gd") + tuple(x for x in o if x.startswith(("CL_", "CT_")) and x[2] == "_"):
                arrays[f"{kind}_rep{m['replica']}_dt{dt:g}_{k}"] = o[k].astype(np.float32)
    arrays["tau"] = R["primary"]["tau"]
    arrays["r_centers"] = R["primary"]["r_centers"]
    np.savez_compressed(OUT / "per_replica_observables.npz", **arrays)
    figures(P, R)
    conclusions(P, R)
    t = dict(stage="analyze", cpu_s=C.cpu_seconds() - cpu0, wall_s=time.perf_counter() - wall0)
    tf = OUT / "postprocessing_time.json"
    db = json.loads(tf.read_text()) if tf.exists() else {}
    db.setdefault("runs", []).append(t)
    C.write_json(tf, db)
    print(t)


# ------------------------------------------------------------------------------------------------ timing
def stage_timing(rounds=3, steps=300):
    """Measured production-runner step cost at dt 0.005 and 0.01 (single level, no coupling), interleaved."""
    tmp = C.RAW / "observable_dt_timing"
    tmp.mkdir(parents=True, exist_ok=True)
    state = C.RAW / "states" / "dtinit_s120.npz"
    cases = [("lj_rho0.75_kT1.0_costopt", 0.005), ("lj_rho0.75_kT1.0_costopt", 0.01),
             ("lj_rho0.75_kT1.0_A_prod", 0.005), ("lj_rho0.75_kT1.0_A_prod", 0.01)]
    res = {f"{c}_dt{dt:g}": [] for c, dt in cases}
    for rd in range(rounds):
        for c, dt in (cases if rd % 2 == 0 else cases[::-1]):
            out = tmp / f"{c}_dt{dt:g}_r{rd}"
            if out.exists():
                subprocess.run(["rm", "-rf", str(out)], check=True)
            cmd = [sys.executable, str(HERE / "toy_run.py"), "--config", c, "--potential", "lj", "--kernel", "A",
                   "--N", "256", "--seed", "11", "--dt", str(dt), "--allow-unverified", "--production-steps",
                   str(steps), "--save-every-steps", str(steps), "--monitor-every-steps", "0", "--quiet",
                   "--from-state", str(state), "--out", str(out)]
            subprocess.run(cmd, check=True, capture_output=True, cwd=HERE)
            st = json.loads((out / "status.json").read_text())
            seg = st["segments"][-1]
            res[f"{c}_dt{dt:g}"].append(dict(cpu_s=seg["cpu_s"], wall_s=seg["wall_s"],
                                             median_step_wall_s=seg.get("median_step_wall_s"), steps=steps))
            print(c, dt, rd, seg["cpu_s"], seg.get("median_step_wall_s"), flush=True)
    summ = {}
    for k, v in res.items():
        med = float(np.median([x["median_step_wall_s"] for x in v]))
        cpu = float(np.median([x["cpu_s"] / x["steps"] for x in v]))
        dt = float(k.split("_dt")[1])
        summ[k] = dict(median_step_wall_s=med, cpu_s_per_step_incl_setup=cpu, dt=dt,
                       core_h_per_time_unit_from_step_wall=med / dt / 3600, rounds=v)
    for c in ("lj_rho0.75_kT1.0_costopt", "lj_rho0.75_kT1.0_A_prod"):
        a, b = summ[f"{c}_dt0.005"], summ[f"{c}_dt0.01"]
        summ[f"{c}_saving_0.01_vs_0.005"] = dict(
            measured_step_cost_ratio=b["median_step_wall_s"] / a["median_step_wall_s"],
            measured_time_per_physical_time_ratio=(b["median_step_wall_s"] / 0.01) / (a["median_step_wall_s"] / 0.005),
            measured_saving=1 - (b["median_step_wall_s"] / 0.01) / (a["median_step_wall_s"] / 0.005),
            step_count_estimate_saving=0.5)
    C.write_json(OUT / "timing.json", dict(summary=summ, steps_per_run=steps, rounds=rounds,
                                           note="median per-step wall time of the production runner (single level, "
                                                "1 thread, no frame output); load from other jobs affects both dt alike "
                                                "(interleaved)", provenance=C.provenance()))
    print(json.dumps({k: v.get("measured_saving", v.get("median_step_wall_s")) for k, v in summ.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["inventory", "protocol", "compute", "analyze", "timing"])
    args = ap.parse_args()
    if args.stage == "inventory":
        stage_inventory()
    elif args.stage == "protocol":
        stage_protocol()
    elif args.stage == "timing":
        stage_timing()
    else:
        P, ph = load_protocol()
        (stage_compute if args.stage == "compute" else stage_analyze)(P, ph)


if __name__ == "__main__":
    main()
