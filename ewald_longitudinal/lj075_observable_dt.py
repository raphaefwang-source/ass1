#!/usr/bin/env python3
"""
Time-step comparison of four LJ-A dynamical observables at the lj075 state point (rho 0.75, kT 1, N 256, law A),
re-using the existing coupled trajectories (no new long trajectories):
  VACF; VCCF grouped by the initial pair distance; longitudinal / transverse current correlations C_L(k,t), C_T(k,t);
  distinct van Hove function g_d(r,t).

Data: lj075_results/raw/dt_study/
  A_rep*_L4_T10.npz         16 coupled 4-level chains (dt 0.00125, 0.0025, 0.005, 0.01), t_end 10, frames every step
  A_rep*_L2_T100_F0.01.npz  11 coupled 2-level chains (dt 0.005, 0.01), t_end 100, frames every 0.01
Both: unwrapped positions, velocity v = p/m after the final half kick, float32 frames; Lanczos ranks 12/5 (coupling 4),
PPPM p 8 (configuration lj_rho0.75_kT1.0_costopt), NOT the later recommended 14/5.

Stages (outputs in lj075_results/observable_dt/, per-level caches in lj075_results/raw/observable_dt_cache/):
    python3 lj075_observable_dt.py inventory   # data manifest, hashes, missing items
    python3 lj075_observable_dt.py compute     # per replica and level observables (cached; no step-size differences)
    python3 lj075_observable_dt.py selftest    # coverage of the simultaneous band on synthetic Gaussian data
    python3 lj075_observable_dt.py protocol    # derive reference-only settings and freeze protocol.json (no overwrite)
    python3 lj075_observable_dt.py analyze     # paired differences, simultaneous bounds, verdicts, tables, figures
    python3 lj075_observable_dt.py timing      # measured production-step cost at dt 0.005 and 0.01
The protocol stage reads only the reference level of each group (0.00125 primary, 0.005 secondary) and refuses to run
if any step-size difference of these observables has been computed (results.json present). analyze refuses to run if
this script or lj075_common.py changed after the freeze, unless --allow-deviation "<reason>" is given (logged).
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
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
N = 256
L = C.box_length(N, 0.75)
REF = {"primary": 0.00125, "secondary": 0.005}
LEVELS_PLAN = (0.00125, 0.0025, 0.005, 0.01)

# --------------------------------------------------------------------------------------------------------------------
# BASE: a-priori settings that the per-level computation needs (fixed before anything was computed)
BASE = dict(
    version=2,
    scope=dict(potential="lj", law="A", N=N, density=0.75, kT=1.0, L=L,
               trajectories="lj_rho0.75_kT1.0_costopt: Lanczos ranks noise/damp 12/5, coupling 4, PPPM p 8; NOT the "
                            "later recommended 14/5",
               note="supplementary analysis; every earlier dt-study verdict (REPORT.md section 3: e.g. RDF norm "
                    "INCONCLUSIVE, MSD(5) FAIL at 0.0025, no dt validated) stands unchanged"),
    comparisons=dict(
        primary=dict(data="16 four-level chains (A_rep*_L4_T10)", reference=0.00125,
                     pairs=[[0.01, 0.00125], [0.005, 0.00125]], reference_sensitivity=[[0.0025, 0.00125]],
                     note="0.00125 is a numerical reference, not the exact solution; verdicts are 'vs 0.00125'"),
        secondary=dict(data="11 two-level chains (A_rep*_L2_T100_F0.01)", reference=0.005, pairs=[[0.01, 0.005]],
                       note="longer trajectories without the 0.0025 / 0.00125 levels: only 0.01 vs 0.005")),
    grid=dict(dtau=0.01, tau_max=2.0,
              note="common physical-time grid; finer levels are read at exact integer strides of their stored frames; "
                   "no interpolation of positions or velocities"),
    origins=dict(primary=dict(start=1.0, stop=8.0, step=0.05), secondary=dict(start=10.0, stop=98.0, step=0.25),
                 note="identical physical origin times for all levels of a chain; origins are averaged within a "
                      "replica and are not statistical units"),
    observables=dict(
        vacf=dict(definition="Z(t) = <v_i(t0).v_i(t0+t)> / <|v_i(t0)|^2> over particles and origins, each level "
                             "normalized by its own <|v|^2> (project convention, prl_toy_models.compute_observables)",
                  amplitude="unnormalized C(t) = <v_i(t0).v_i(t0+t)>; curve 'vacf_amp' = C(t) / mean reference C(0) "
                            "(the definition of the earlier E_V criterion; continuity only)"),
        vccf=dict(definition="C_b(t) = <v_i(t0).v_j(t0+t)> over ordered pairs i != j with minimum-image r_ij(t0) in bin "
                             "b, divided by the same level's <|v(t0)|^2> (project convention vccf_normalized)",
                  bins=[0.0, 1.59, 2.55, L / 2],
                  bins_note="first and second minima of the reference RDF at this state point (1.59, 2.55; from the "
                            "earlier dt-study reference g(r)), last bin to L/2 (minimum image)",
                  momentum_sum_rule="P_total = 0 (|P| < 2e-12): sum over j != i of v_i(t0).v_j(t0+t) = "
                                    "-v_i(t0).v_i(t0+t), so the mean over ALL ordered pairs is -Z(t)/(N-1). This is not "
                                    "a per-bin baseline: the bins cover only r < L/2, and at t = 0 every bin should "
                                    "be -1/(N-1) = -0.0039 (velocities independent of positions); C_b(0) is reported "
                                    "as a check. Nothing is subtracted."),
        currents=dict(definition="j(k,t) = (1/N) sum_i v_i exp(i k.r_i); for each mode C_L = Re<j_L(t0+t) j_L*(t0)> / "
                                 "<|j_L(t0)|^2>, C_T = Re<j_T(t0+t).j_T*(t0)> / <|j_T(t0)|^2> (origin averages, per "
                                 "level), then the mean over the modes of the shell (prl_toy_models.compute_observables "
                                 "hydro_L / hydro_T + ensemble.py mode mean)",
                      shells={"k1": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              "k2": [[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]],
                              "k4": [[2, 0, 0], [0, 2, 0], [0, 0, 2]]},
                      k_values={"k1": 2 * np.pi / L, "k2": 2 * np.pi / L * np.sqrt(2), "k4": 4 * np.pi / L},
                      amplitude="raw <|j_L|^2>, <|j_T|^2>/2 at t = 0 (shell mean) reported as scalars CL0, CT0",
                      note="k = 0 is identically zero (P_total = 0) and excluded"),
        van_hove=dict(definition="g_d(r,t) = V / (n_origins N (N-1)) * sum over ordered pairs i != j of "
                                 "delta(r - |r_j(t0+t) - r_i(t0)|_min-image) / (4 pi r^2 dr)",
                      times=[0.0, 0.1, 0.5, 1.0, 2.0], verdict_times=[0.1, 0.5, 1.0, 2.0],
                      r_bins=dict(n=60, r_max=L / 2),
                      finite_N="N (N-1) ordered pairs: g_d(r,0) = g(r) and g_d -> 1 for uncorrelated pairs",
                      t0_note="g_d(r,0) = g(r) is plotted only; the static RDF keeps its registered verdict "
                              "(REPORT.md 3.3: E_g bootstrap bound 1.68-1.82%, INCONCLUSIVE)")),
    windows=dict(time={"W1": [0.0, 0.2], "W2": [0.2, 0.5], "W3": [0.5, 1.0], "W4": [1.0, 2.0], "ALL": [0.0, 2.0]},
                 time_note="half-open (a, b]; t = 0 excluded (normalized curves are identical there). 'ALL' is the "
                           "whole-curve cell with its own simultaneous band",
                 radius={"R1a_core": [0.0, 0.9], "R1b_shell": [0.9, 1.59], "R2": [1.59, L / 2], "ALL": [0.0, L / 2]}),
)

RULES = dict(
    rules_version=3,
    amendments=[dict(
        version=3, made="after the v2 freeze (protocol_v2_frozen.json, hash a835d753e199f747) and BEFORE any step-size "
                        "difference of these observables was computed (results.json absent)",
        change="lobes: removed the rule 'from the first unresolved lobe on, the rest of the curve is one unresolved "
               "tail'; every sign lobe is judged on its own. Thresholds, bands and tolerances unchanged",
        reason="reference-only diagnostic: C_b(0) = -1/(N-1) makes the first lobe of every VCCF a tiny unresolved "
               "negative transient, which under v2 marked the whole VCCF (including its resolved positive peak, 0.029 "
               "at t = 0.1, relative half-width 0.046) as unresolved. Under v3 only that peak lobe changes status; the "
               "primary VACF dip (0.104 > 0.1) and the C_L / C_T negative lobes stay unresolved",
        reporting="results are reported under both v2 (results_v2.json) and v3 (results.json)")],
    tolerance=dict(tol=0.01, scale_resolution=0.1, count_min_per_replica=100),
    bootstrap=dict(method="re-studentized sign-flip max-t: for sign vectors e, Y_r = e_r (D_r - mean D); T* = max over "
                          "grid points of |mean Y| / se(Y) with se recomputed per draw; c_boot = quantile of T*. All "
                          "2^(n-1) sign vectors are enumerated (T* is invariant under e -> -e). Bonferroni-t "
                          "c_bonf = t_{1-alpha/(2p), n-1} over the p points; the band uses c = max(c_boot, c_bonf)",
                   level_each=0.975,
                   joint_note="the difference band (97.5%, simultaneous over the window) and the reference-scale band "
                              "(97.5%, simultaneous over the whole curve) hold jointly with >= 95% (Bonferroni)",
                   selftest="band_selftest.json: coverage on synthetic Gaussian data at n = 11, 16 and the window "
                            "sizes, independent and correlated columns, before freezing"),
    metrics=dict(
        signal="deviation from the uncorrelated value: Z, C_b, C_L, C_T -> 0; g_d -> 1",
        lobes="the reference mean curve (t in (0, 2], or r for g_d - 1) is cut into sign lobes; each lobe gets the "
              "scale S = sup over the lobe of |reference mean| with simultaneous bounds S_lo, S_hi; a lobe is resolved "
              "if (S_hi - S_lo)/(S_hi + S_lo) <= scale_resolution. Every lobe is judged on its own (amendment v3). "
              "Every grid point is judged against the scale of its own lobe, so weak lobes are judged against their own "
              "size and no relative error is formed at a zero crossing",
        E_abs="sup over the window of |mean paired difference| in the units of the displayed quantity (VACF / C_L / "
              "C_T: fraction of the same level's t = 0 value; VCCF: fraction of <|v|^2>; g_d: g units)",
        E_sig="sup over the window of |mean difference(t)| / S_lobe(t); upper bound sup (|m| + c se) / S_lo(t), lower "
              "bound sup max(|m| - c se, 0) / S_hi(t)",
        E_glob="E_abs / sup over the whole curve of |reference mean|: descriptive only, never a verdict",
        shape_only="VACF, VCCF, C_L, C_T are normalized by each level's own t = 0 value: curve verdicts are shape-only "
                   "and exclude the amplitude (kinetic-temperature) bias, which is judged by the scalars vacf_C0, "
                   "CL0_k, CT0_k",
        pointwise_vs_simultaneous="figures show pointwise 95% t-intervals (coverage 95% at each t separately) and the "
                                  "simultaneous band of the ALL cell (covers the whole curve at once); only "
                                  "simultaneous bounds enter verdicts"),
    verdicts=dict(
        curves="FAIL if E_sig lower > tol; PASS if E_sig upper <= tol and every lobe met by the window is resolved; "
               "otherwise INCONCLUSIVE ('unresolved lobe' when applicable). Verdicts are 'vs the reference level'",
        resolved_deviation="the simultaneous lower bound of E_abs > 0 (whatever its size)",
        scalars="per replica, paired. If the scalar is undefined (NaN) in any replica at either level: INCONCLUSIVE "
                "(undefined k/n), no replica is dropped. Absolute paired difference with Student-t 97.5% CI always "
                "reported. Scale s: |mean reference| for values, times and positions, int_0^2 |mean reference curve| "
                "dt for integrals, mean(peak) - 1 for van Hove peak heights; its uncertainty se_s (jackknife over "
                "replicas). If t se_s / s > scale_resolution: INCONCLUSIVE (reference not resolved). PASS if "
                "max(|lo|, |hi|) / (s - t se_s) <= tol; FAIL if the CI excludes the band, judged with s + t se_s",
        van_hove_bins="r bins whose mean reference pair count per replica is below count_min_per_replica are "
                      "excluded from bands and scales (listed)"),
    scalars=dict(
        windows="search windows, signs and lobes come from the reference mean only (frozen in 'derived'); an "
                "extremum at the edge of its search window is undefined (NaN); scalars whose reference feature is not "
                "resolved are dropped in advance",
        list=["vacf_C0 (= <|v|^2>, the T_kin ratio)", "vacf_Zmin, vacf_tmin (dip of Z)", "vacf_tzero (first zero)",
              "vacf_D2 = (1/3) int_0^2 C(t) dt", "vccf_b*_C0 (check vs -1/(N-1); information)",
              "vccf_b*_ext, vccf_b*_text (largest resolved interior extremum of the reference)", "vccf_b*_int",
              "CL_k*_min, CL_k*_tmin (first minimum after the first zero)", "CL_k*_tzero", "CT_k*_thalf",
              "CT_k*_int", "CL0_k*, CT0_k* (raw amplitudes)", "vh_t*_peak_excess (= peak - 1), vh_t*_rpeak"]),
    statistics=dict(unit="independent replica (16 primary, 11 secondary); frames, particles, origins and distance bins "
                         "are never statistical units",
                    pairing="paired per replica between levels of the same chain (same initial state and noise "
                            "stream), also after the paths have separated",
                    multiplicity="bands are simultaneous within a cell only; many cells are reported, so isolated "
                                 "verdicts near the threshold are expected by chance; FAIL and resolved deviations are "
                                 "reported as observed, not explained away"),
    decision=dict(
        display_support="per observable x window and dt in {0.01, 0.005}, only from E_sig curve verdicts and scalar "
                        "FAILs (E_glob never counts): NOT SUPPORTED if the primary verdict (dt vs 0.00125) is FAIL, or "
                        "(dt 0.01) the secondary 0.01 vs 0.005 is FAIL, or a shape scalar of that observable FAILs; "
                        "SUPPORTED if the primary verdict is PASS and 0.0025 vs 0.00125 is PASS in the same cell; "
                        "INCONCLUSIVE (reference not adequate) if the primary is PASS but 0.0025 vs 0.00125 is not; "
                        "otherwise INCONCLUSIVE with the primary reason",
        absolute_units="curve support is for the normalized (shape) display; a display in absolute units also needs "
                       "the amplitude scalars (vacf_C0, CL0, CT0) to PASS",
        exact_solution_bound="model-dependent qualification only (first order, p = 1): E_sig upper (dt vs 0.00125) + "
                             "E_sig upper (0.0025 vs 0.00125); never upgrades a verdict",
        long_time_D="the long-time D plateau (REPORT.md section 5) is not addressed by t <= 2 correlation functions and "
                    "stays unconfirmed"),
    prior_looks=dict(
        vacf="dt differences of the VACF (t <= 2, scaled by C(0)) were computed in the earlier dt study before this "
             "protocol (E_V 0.01 vs 0.00125: 0.52% [0.26, 0.78] PASS): NOT blind",
        van_hove_t0="g_d(r,0) = g(r): RDF differences seen earlier: NOT blind (no verdict here)",
        blind=["vccf", "currents C_L / C_T", "van Hove t > 0", "lobe-scaled VACF windows (new metric)"]),
)


def sha256(path, full=False):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest() if full else h.hexdigest()[:16]


def proto_hash(p):
    return hashlib.sha256(json.dumps(p, sort_keys=True, default=float).encode()).hexdigest()[:16]


BASE_HASH = proto_hash(json.loads(json.dumps(BASE, default=float)))
CODE_FILES = (Path(__file__).resolve(), HERE / "lj075_common.py")


# ----------------------------------------------------------------------------------------------------------- data
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
                Q, S = z[f"Q_{dt}"], z[f"S_{dt}"]
                fd = frame_dt(m, dt)
                lev[str(dt)] = dict(frames=int(len(Q)), frame_interval=fd, t_last=float((len(Q) - 1) * fd),
                                    dtype=str(Q.dtype),
                                    position_jumps_over_L2=bool(np.any(np.abs(np.diff(Q, axis=0)) > L / 2)),
                                    max_abs_P_total=float(S[:, 4].max()),
                                    T_kin_mean_after_1=float(S[S[:, 0] >= 1.0, 1].mean()))
        rows.append(dict(file=f.name, kind=kind, replica=m["replica"], init_state=m["init_state"], levels=m["levels"],
                         t_end=m["t_end"], frame_every=m.get("frame_every", "every step (field absent: early run)"),
                         seed=m["seed"], ranks=[m["resolved"]["rank_noise"], m["resolved"]["rank_damp"]],
                         rank_couple=m["rank_couple"], pppm=m["resolved"]["pppm"], config=m["resolved"]["config"],
                         levels_detail=lev, npz_sha256=sha256(f, full=True), json_sha256=sha256(f.with_suffix(".json"), full=True)))
    prim = [r for r in rows if r["kind"] == "primary"]
    sec = [r for r in rows if r["kind"] == "secondary"]
    if len(prim) != 16:
        missing.append(f"expected 16 primary chains, found {len(prim)}")
    if len(sec) != 11:
        missing.append(f"expected 11 secondary chains, found {len(sec)}")
    missing += [
        "secondary chains have no 0.0025 / 0.00125 levels: they compare 0.01 with 0.005 only",
        "the comparison grid is 0.01 (the 0.01 level has no sub-step frames): features faster than 0.01 are not "
        "resolved; the VACF half-decay time is 0.045 (about 4-5 grid points)",
        "smallest wave number is 2 pi / L = 0.899 (N = 256); smaller k need a larger box",
        "primary chains are 10 time units long: lags up to 2 with origins in [1, 8] only",
        "trajectories were run with ranks 12/5 (coupling 4), not with the recommended production ranks 14/5"]
    jumps = sorted({r["file"] for r in rows for v in r["levels_detail"].values() if v["position_jumps_over_L2"]})
    res = dict(chains=rows, n_primary=len(prim), n_secondary=len(sec),
               positions="unwrapped (no frame-to-frame jump > L/2)" if not jumps else f"WRAPPED or jumping in {jumps}",
               velocity_definition="v = p/m after the final half kick of the BAOAB step (full-step velocity), "
                                   "lj075_dt_study.py record()",
               missing_or_limited=missing, provenance=C.provenance())
    OUT.mkdir(parents=True, exist_ok=True)
    C.write_json(OUT / "inventory.json", res)
    print(f"inventory: {len(prim)} primary, {len(sec)} secondary chains; positions {res['positions']}")
    for x in missing:
        print("  -", x)


# ---------------------------------------------------------------------------------------------------- observables
def min_image(d):
    return d - L * np.round(d / L)


def origins_for(kind):
    o = BASE["origins"][kind]
    return np.round(np.arange(o["start"], o["stop"] + 1e-9, o["step"]), 10)


def level_observables(Q, V, fd, origins):
    """All observables of one level on the common grid (BASE settings). Q, V: stored frames (unwrapped)."""
    g = BASE["grid"]
    stride = int(round(g["dtau"] / fd))
    if abs(stride * fd - g["dtau"]) > 1e-12:
        raise ValueError("frame interval does not divide the grid spacing")
    nl = int(round(g["tau_max"] / g["dtau"])) + 1
    o_idx = np.rint(origins / fd).astype(int)
    if np.any(np.abs(o_idx * fd - origins) > 1e-9):
        raise ValueError("origins are not on stored frames")
    lag_idx = np.arange(nl) * stride
    if o_idx.max() + lag_idx[-1] >= len(Q):
        raise ValueError("trajectory too short for the requested origins and lags")
    need = np.unique((o_idx[:, None] + lag_idx[None, :]).ravel())
    pos = {f: i for i, f in enumerate(need)}
    Qs, Vs = Q[need].astype(float), V[need].astype(float)
    oi = np.array([pos[f] for f in o_idx])
    li = np.array([[pos[o + l] for l in lag_idx] for o in o_idx])
    no = len(oi)
    # VACF (unnormalized, per particle)
    num, v2 = np.zeros(nl), 0.0
    for a, row in zip(oi, li):
        num += np.einsum("nc,tnc->t", Vs[a], Vs[row]) / N
        v2 += np.sum(Vs[a] ** 2) / N
    vacf_un, v2 = num / no, v2 / no
    # VCCF: sum over pairs (i at t0 in bin with j) of v_i(t0).v_j(t0+t) = sum_j (M^T v(t0))_j . v_j(t0+t)
    edges = np.asarray(BASE["observables"]["vccf"]["bins"], float)
    nb = len(edges) - 1
    cross, counts = np.zeros((nb, nl)), np.zeros(nb)
    eye = np.eye(N, dtype=bool)
    for a, row in zip(oi, li):
        r0 = np.linalg.norm(min_image(Qs[a][:, None, :] - Qs[a][None, :, :]), axis=-1)
        lab = np.searchsorted(edges, r0, side="right") - 1
        lab[eye | (r0 >= edges[-1])] = -1
        for b in range(nb):
            M = (lab == b)
            counts[b] += M.sum()
            cross[b] += np.einsum("jc,tjc->t", M.T.astype(float) @ Vs[a], Vs[row])
    vccf = cross / counts[:, None] / v2
    # currents: per-mode normalization, then shell mean
    cur = {}
    for name, modes in BASE["observables"]["currents"]["shells"].items():
        kv = 2 * np.pi / L * np.asarray(modes, float)
        kh = kv / np.linalg.norm(kv, axis=1)[:, None]
        ph = np.exp(1j * np.einsum("fnc,kc->fnk", Qs, kv))
        j = np.einsum("fnc,fnk->fkc", Vs, ph) / N                       # (frames, modes, 3)
        jl = np.einsum("fkc,kc->fk", j, kh)
        jt = j - jl[..., None] * kh[None]
        clm = np.zeros((nl, len(modes)))
        ctm = np.zeros((nl, len(modes)))
        for a, row in zip(oi, li):
            clm += (jl[row] * jl[a].conj()[None]).real
            ctm += np.sum((jt[row] * jt[a].conj()[None]).real, axis=-1) / 2
        clm, ctm = clm / no, ctm / no
        cur[name] = (np.mean(clm / clm[0], axis=1), np.mean(ctm / ctm[0], axis=1), clm[0].mean(), ctm[0].mean())
    # van Hove (counts kept for the minimum-count rule)
    vh = BASE["observables"]["van_hove"]
    rb = np.linspace(0, vh["r_bins"]["r_max"], vh["r_bins"]["n"] + 1)
    shell_vol = 4 * np.pi / 3 * np.diff(rb ** 3)
    gd, gc = [], []
    for tt in vh["times"]:
        k = int(round(tt / g["dtau"]))
        hist = np.zeros(len(rb) - 1)
        for a, row in zip(oi, li):
            d = np.linalg.norm(min_image(Qs[row[k]][None, :, :] - Qs[a][:, None, :]), axis=-1)[~eye]
            hist += np.histogram(d, bins=rb)[0]
        gc.append(hist)
        gd.append(hist * L ** 3 / (no * N * (N - 1) * shell_vol))
    out = dict(tau=np.arange(nl) * g["dtau"], vacf_un=vacf_un, v2=np.array(v2), vacf=vacf_un / v2, vccf=vccf,
               vccf_counts=counts, gd=np.array(gd), gd_counts=np.array(gc), r_edges=rb,
               r_centers=0.5 * (rb[1:] + rb[:-1]), n_origins=np.array(no))
    for k, (cl, ct, cl0, ct0) in cur.items():
        out[f"CL_{k}"], out[f"CT_{k}"], out[f"CL0_{k}"], out[f"CT0_{k}"] = cl, ct, np.array(cl0), np.array(ct0)
    return out


def cache_file(f, dt):
    return CACHE / f"{f.stem}_dt{dt:g}.npz"


def compute_one(task):
    m, f, kind, dt = task
    out = cache_file(f, dt)
    key = f"{BASE_HASH}:{f.stat().st_size}:{int(f.stat().st_mtime)}"
    if out.exists():
        with np.load(out) as z:
            if str(z["key"]) == key:
                return 0, 0.0
    t0 = time.perf_counter()
    with np.load(f) as z:
        obs = level_observables(z[f"Q_{dt}"], z[f"V_{dt}"], frame_dt(m, dt), origins_for(kind))
    np.savez(out, key=np.array(key), **obs)
    print(f"{f.stem} dt {dt}: {time.perf_counter() - t0:.1f} s", flush=True)
    return 1, time.process_time()


def stage_compute(jobs=4):
    from concurrent.futures import ProcessPoolExecutor
    CACHE.mkdir(parents=True, exist_ok=True)
    cpu0, wall0 = C.cpu_seconds(), time.perf_counter()
    tasks = [(m, f, kind, dt) for m, f, kind in chains() for dt in m["levels"]]
    with ProcessPoolExecutor(jobs) as ex:
        done = sum(r[0] for r in ex.map(compute_one, tasks))
    log_time(dict(stage="compute", new_level_files=done, jobs=jobs, cpu_s=C.cpu_seconds() - cpu0,
                  wall_s=time.perf_counter() - wall0, note="cpu_s includes the worker processes (RUSAGE_CHILDREN)"))


def log_time(t):
    tf = OUT / "postprocessing_time.json"
    db = json.loads(tf.read_text()) if tf.exists() else {}
    db.setdefault("runs", []).append(dict(t, at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    C.write_json(tf, db)
    print(t)


def load_level(f, dt):
    with np.load(cache_file(f, dt)) as z:
        if not str(z["key"]).startswith(BASE_HASH):
            raise RuntimeError(f"cache {cache_file(f, dt).name} is from different BASE settings")
        return {k: z[k] for k in z.files if k != "key"}


# ------------------------------------------------------------------------------------------------------ statistics
def crit(D, level):
    """Simultaneous critical value over the columns of D (replicas x points): max(re-studentized sign-flip max-t,
    Bonferroni-t). Returns (c, c_boot, c_bonf, p)."""
    n = D.shape[0]
    se = D.std(0, ddof=1) / np.sqrt(n)
    ok = se > 0
    p = int(ok.sum())
    if p == 0:
        return 0.0, 0.0, 0.0, 0
    R = (D - D.mean(0))[:, ok]
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=n - 1)))
    signs = np.hstack([np.ones((len(signs), 1)), signs])
    T = []
    for s in range(0, len(signs), 2048):
        X = signs[s:s + 2048, :, None] * R[None]
        mb = X.mean(1)
        sb = X.std(1, ddof=1) / np.sqrt(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            T.append(np.nanmax(np.where(sb > 0, np.abs(mb) / sb, 0.0), axis=1))
    c_boot = float(np.quantile(np.concatenate(T), level))
    c_bonf = float(stats.t.ppf(1 - (1 - level) / (2 * p), n - 1))
    return max(c_boot, c_bonf), c_boot, c_bonf, p


def lobes(m, se_band, resolution):
    """Sign lobes of the reference mean m (points 1..len-1; point 0 excluded), with simultaneous half-widths se_band.
    Returns per-point arrays s_hat, s_lo, s_hi, resolved, lobe_id and the lobe list."""
    n = len(m)
    s_hat, s_lo, s_hi = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    res, lid = np.zeros(n, bool), np.full(n, -1)
    idx = np.where(np.isfinite(m))[0]
    idx = idx[idx > 0] if n > 1 else idx
    if len(idx) == 0:
        return s_hat, s_lo, s_hi, res, lid, []
    segs, cur = [], [idx[0]]
    for i in idx[1:]:
        if np.sign(m[i]) == np.sign(m[cur[-1]]) and i == cur[-1] + 1:
            cur.append(i)
        else:
            segs.append(cur)
            cur = [i]
    segs.append(cur)
    out = []
    for sg in segs:                       # amendment v3: every lobe is judged on its own (no tail merging)
        sg = np.array(sg)
        hat = float(np.max(np.abs(m[sg])))
        lo = float(np.max(np.maximum(np.abs(m[sg]) - se_band[sg], 0)))
        hi = float(np.max(np.abs(m[sg]) + se_band[sg]))
        ok = bool(lo > 0 and (hi - lo) / (hi + lo) <= resolution)
        s_hat[sg], s_lo[sg], s_hi[sg], res[sg], lid[sg] = hat, lo, hi, ok, len(out)
        out.append(dict(start=int(sg[0]), end=int(sg[-1]), sign=int(np.sign(m[sg[0]])), S=hat, S_lo=lo, S_hi=hi,
                        resolved=ok, rel_halfwidth=(hi - lo) / (hi + lo)))
    return s_hat, s_lo, s_hi, res, lid, out


# ------------------------------------------------------------------------------------------------------- curves
def curve_specs():
    specs = [("vacf", lambda o, c0: o["vacf"]), ("vacf_amp", lambda o, c0: o["vacf_un"] / c0)]
    for b in range(len(BASE["observables"]["vccf"]["bins"]) - 1):
        specs.append((f"vccf_b{b + 1}", lambda o, c0, b=b: o["vccf"][b]))
    for k in BASE["observables"]["currents"]["shells"]:
        specs.append((f"CL_{k}", lambda o, c0, k=k: o[f"CL_{k}"]))
        specs.append((f"CT_{k}", lambda o, c0, k=k: o[f"CT_{k}"]))
    return specs


def group_obs(kind, levels):
    ch = [(m, f) for m, f, k in chains() if k == kind]
    return [m["replica"] for m, _ in ch], {dt: [load_level(f, dt) for _, f in ch] for dt in levels}


def window_points(x, w, exclude_zero=True):
    a, b = w
    msk = (x > a + 1e-12) & (x <= b + 1e-12)
    if not exclude_zero:
        msk |= (x >= a - 1e-12) & (x <= b + 1e-12)
    return msk


def r_window(rc, w):
    return (rc >= w[0]) & (rc < w[1])


# -------------------------------------------------------------------------------------------------- scalar defs
def interior_extremum(t, y, sign, lo, hi):
    """Interior discrete extremum of sign*y with t in [lo, hi]; quadratic interpolation; NaN if at the window edge."""
    s = sign * y
    idx = np.where((t >= lo - 1e-12) & (t <= hi + 1e-12))[0]
    idx = idx[(idx >= 1) & (idx <= len(y) - 2)]
    if len(idx) < 3:
        return np.nan, np.nan
    i = idx[int(np.argmax(s[idx]))]
    if i == idx[0] or i == idx[-1] or not (s[i] >= s[i - 1] and s[i] >= s[i + 1]):
        return np.nan, np.nan
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2 * y1 + y2
    x = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return float(y1 + 0.5 * (y2 - y0) * x + 0.5 * den * x * x), float(t[i] + x * (t[1] - t[0]))


def crossing(t, y, level, lo, hi):
    """First downward crossing of y through level with both bracketing points in [lo, hi]; linear interpolation."""
    idx = np.where((t >= lo - 1e-12) & (t <= hi + 1e-12))[0]
    for a, b in zip(idx[:-1], idx[1:]):
        ya, yb = y[a] - level, y[b] - level
        if ya > 0 >= yb:
            return float(t[a] + ya / (ya - yb) * (t[b] - t[a]))
    return np.nan


def derive_scalar_windows(m_curves, tau, rc, m_gd, se_curves, n):
    """Search windows, signs and drop decisions from the reference mean curves (reference level only)."""
    t_ = BASE["grid"]["dtau"]
    out = {}
    z = m_curves["vacf"]
    zmin_t = tau[1 + int(np.argmin(z[1:]))]
    out["vacf_Zmin"] = dict(type="min", curve="vacf", lo=zmin_t - 0.1, hi=zmin_t + 0.1, ref_t=float(zmin_t))
    tz = crossing(tau, z, 0.0, t_, 2.0)
    out["vacf_tzero"] = dict(type="crossing", curve="vacf", level=0.0, lo=tz - 0.1, hi=tz + 0.1, ref_t=float(tz))
    for b in range(len(BASE["observables"]["vccf"]["bins"]) - 1):
        key = f"vccf_b{b + 1}"
        y, se = m_curves[key], se_curves[key]
        cands = []
        for i in range(2, len(y) - 2):
            for sg in (1, -1):
                if sg * y[i] >= sg * y[i - 1] and sg * y[i] >= sg * y[i + 1]:
                    cands.append((abs(y[i]), i, sg))
        if not cands:
            out[f"{key}_ext"] = dict(dropped="no interior extremum in the reference mean")
            continue
        val, i, sg = max(cands)
        resolved = stats.t.ppf(0.975, n - 1) * se[i] / val <= RULES["tolerance"]["scale_resolution"]
        if not resolved:
            out[f"{key}_ext"] = dict(dropped=f"reference extremum {y[i]:.2e} at t = {tau[i]:.2f} not resolved")
            continue
        out[f"{key}_ext"] = dict(type="max" if sg > 0 else "min", curve=key, lo=tau[i] - 0.1, hi=tau[i] + 0.1,
                                 ref_t=float(tau[i]), ref_value=float(y[i]))
    for k in BASE["observables"]["currents"]["shells"]:
        y = m_curves[f"CL_{k}"]
        t0 = crossing(tau, y, 0.0, t_, 2.0)
        out[f"CL_{k}_tzero"] = dict(type="crossing", curve=f"CL_{k}", level=0.0, lo=t0 - 0.1, hi=t0 + 0.1,
                                    ref_t=float(t0)) if np.isfinite(t0) else dict(dropped="no zero crossing")
        if np.isfinite(t0):
            idx = np.where(tau > t0)[0]
            mins = [i for i in idx[1:-1] if y[i] <= y[i - 1] and y[i] <= y[i + 1]]
            if mins:
                i = mins[0]
                out[f"CL_{k}_min"] = dict(type="min", curve=f"CL_{k}", lo=tau[i] - 0.1, hi=tau[i] + 0.1,
                                          ref_t=float(tau[i]), ref_value=float(y[i]))
            else:
                out[f"CL_{k}_min"] = dict(dropped="no minimum after the first zero")
        yt = m_curves[f"CT_{k}"]
        th = crossing(tau, yt, 0.5, t_, 2.0)
        out[f"CT_{k}_thalf"] = dict(type="crossing", curve=f"CT_{k}", level=0.5, lo=th - 0.2, hi=th + 0.2,
                                    ref_t=float(th)) if np.isfinite(th) else dict(dropped="C_T does not reach 1/2")
    for j, tt in enumerate(BASE["observables"]["van_hove"]["times"]):
        if tt not in BASE["observables"]["van_hove"]["verdict_times"]:
            continue
        g = m_gd[j]
        msk = r_window(rc, BASE["windows"]["radius"]["R1b_shell"])
        i = np.where(msk)[0][int(np.argmax(g[msk]))]
        out[f"vh_t{tt:g}_peak"] = dict(type="max", curve=f"vh{j}", lo=rc[i] - 0.15, hi=rc[i] + 0.15,
                                       ref_r=float(rc[i]), ref_value=float(g[i]))
    return out


def scalars_of(o, c0ref, sw):
    t = o["tau"]
    s = dict(vacf_C0=float(o["v2"]), vacf_D2=float(np.trapezoid(o["vacf_un"], t) / 3))
    for name, w in sw.items():
        if "dropped" in w:
            continue
        if w["curve"].startswith("vh"):
            j = int(w["curve"][2:])
            v, r = interior_extremum(o["r_centers"], o["gd"][j], 1, w["lo"], w["hi"])
            s[f"{name}_excess"], s[name.replace("_peak", "_rpeak")] = v - 1, r
            continue
        y = o[w["curve"]] if w["curve"] in o else None
        if w["curve"].startswith("vccf_b"):
            y = o["vccf"][int(w["curve"][6:]) - 1]
        if w["type"] in ("min", "max"):
            v, tt = interior_extremum(t, y, 1 if w["type"] == "max" else -1, w["lo"], w["hi"])
            base = name[:-4] if name.endswith(("_ext", "_min")) else name
            if name == "vacf_Zmin":
                s["vacf_Zmin"], s["vacf_tmin"] = v, tt
            elif name.endswith("_ext"):
                s[name], s[f"{base}_text"] = v, tt
            else:
                s[name], s[f"{base}_tmin"] = v, tt
        else:
            s[name] = crossing(t, y, w["level"], w["lo"], w["hi"])
    nb = len(BASE["observables"]["vccf"]["bins"]) - 1
    for b in range(nb):
        s[f"vccf_b{b + 1}_C0"] = float(o["vccf"][b][0])
        s[f"vccf_b{b + 1}_int"] = float(np.trapezoid(o["vccf"][b], t))
    for k in BASE["observables"]["currents"]["shells"]:
        s[f"CT_{k}_int"] = float(np.trapezoid(o[f"CT_{k}"], t))
        s[f"CL0_{k}"], s[f"CT0_{k}"] = float(o[f"CL0_{k}"]), float(o[f"CT0_{k}"])
    return s


INTEGRAL_SCALARS = {"vacf_D2": ("vacf_amp", 1 / 3)}


def scalar_scale(name, xb, ref_obs, c0ref):
    """Scale s of a scalar and its jackknife standard error (reference replicas)."""
    n = len(xb)
    if name.endswith("_int") or name == "vacf_D2":
        if name == "vacf_D2":
            Y = np.array([o["vacf_un"] for o in ref_obs]) / 3
        elif name.startswith("vccf_b"):
            Y = np.array([o["vccf"][int(name.split("_")[1][1:]) - 1] for o in ref_obs])
        else:
            Y = np.array([o[name[:-4]] for o in ref_obs])
        t = ref_obs[0]["tau"]
        f = lambda A: float(np.trapezoid(np.abs(A.mean(0)), t))         # noqa: E731
        s = f(Y)
        jk = np.array([f(np.delete(Y, i, axis=0)) for i in range(n)])
        se = float(np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2)))
        return s, se, "int_0^2 |mean reference curve| dt"
    if name.endswith("_C0") and name.startswith("vccf"):
        return 1.0 / (N - 1), 0.0, "1/(N-1) (information only)"
    return float(abs(np.mean(xb))), float(np.std(xb, ddof=1) / np.sqrt(n)), "|mean reference|"


def scalar_verdict(name, xa, xb, ref_obs, c0ref):
    n = len(xa)
    tol, res_thr = RULES["tolerance"]["tol"], RULES["tolerance"]["scale_resolution"]
    ka, kb = int(np.sum(~np.isfinite(xa))), int(np.sum(~np.isfinite(xb)))
    out = dict(n=n, undefined_dt=ka, undefined_ref=kb)
    if ka or kb:
        out["verdict"] = f"INCONCLUSIVE (undefined in {ka}/{n} at dt, {kb}/{n} at reference)"
        return out
    d = xa - xb
    tq = stats.t.ppf(1 - (1 - RULES["bootstrap"]["level_each"]) / 2, n - 1)
    m = float(d.mean())
    h = float(tq * d.std(ddof=1) / np.sqrt(n))
    s, se_s, sdef = scalar_scale(name, xb, ref_obs, c0ref)
    out.update(abs_diff=m, abs_ci=[m - h, m + h], ref_mean=float(xb.mean()), dt_mean=float(xa.mean()), scale=s,
               scale_def=sdef, scale_se=se_s, rel_diff=m / s if s else np.nan,
               rel_ci=[(m - h) / s, (m + h) / s] if s else [np.nan, np.nan],
               resolved_deviation=bool(m - h > 0 or m + h < 0))
    if name.startswith("vccf") and name.endswith("_C0"):
        out["verdict"] = "information"
        return out
    if s == 0 or tq * se_s / s > res_thr:
        out["verdict"] = "INCONCLUSIVE (reference not resolved)"
        return out
    s_lo, s_hi = s - tq * se_s, s + tq * se_s
    if max(abs(m - h), abs(m + h)) / s_lo <= tol:
        out["verdict"] = "PASS"
    elif (m - h > 0 and (m - h) / s_hi > tol) or (m + h < 0 and -(m + h) / s_hi > tol):
        out["verdict"] = "FAIL"
    else:
        out["verdict"] = "INCONCLUSIVE"
    return out


# --------------------------------------------------------------------------------------------------- self-test
def stage_selftest(reps=1000, seed=7):
    rng = np.random.default_rng(seed)
    res = {}
    for n in (11, 16):
        for p in (20, 30, 50, 100, 200):
            for corr in ("independent", "smooth_ell5"):
                if corr == "independent":
                    Lc = np.eye(p)
                else:
                    x = np.arange(p)
                    S = np.exp(-0.5 * ((x[:, None] - x[None]) / 5.0) ** 2) + 1e-6 * np.eye(p)
                    Lc = np.linalg.cholesky(S)
                cov_c = cov_boot = cov_bonf = 0
                for _ in range(reps):
                    D = rng.standard_normal((n, p)) @ Lc.T
                    m, se = D.mean(0), D.std(0, ddof=1) / np.sqrt(n)
                    T = np.max(np.abs(m) / se)
                    c, cb, cf, _ = crit(D, 0.95) if n <= 11 else crit_fast(D, 0.95, rng)
                    cov_c += T <= c
                    cov_boot += T <= cb
                    cov_bonf += T <= cf
                res[f"n{n}_p{p}_{corr}"] = dict(coverage_used=cov_c / reps, coverage_boot=cov_boot / reps,
                                                coverage_bonf=cov_bonf / reps, reps=reps)
                print(n, p, corr, res[f"n{n}_p{p}_{corr}"], flush=True)
    worst = min(v["coverage_used"] for v in res.values())
    C.write_json(OUT / "band_selftest.json", dict(results=res, worst_coverage_used=worst, nominal=0.95,
                                                  requirement=">= 0.93 for the band actually used (max of the two)",
                                                  passed=bool(worst >= 0.93),
                                                  note="n = 16 uses 4096 random sign vectors here (speed); the analysis "
                                                       "enumerates all 2^15", provenance=C.provenance()))
    print("worst coverage of the band used:", worst)


def crit_fast(D, level, rng, B=4096):
    n = D.shape[0]
    se = D.std(0, ddof=1) / np.sqrt(n)
    R = D - D.mean(0)
    eps = rng.choice([-1.0, 1.0], size=(B, n))
    X = eps[:, :, None] * R[None]
    T = np.max(np.abs(X.mean(1)) / (X.std(1, ddof=1) / np.sqrt(n)), axis=1)
    cb = float(np.quantile(T, level))
    cf = float(stats.t.ppf(1 - (1 - level) / (2 * D.shape[1]), n - 1))
    return max(cb, cf), cb, cf, D.shape[1]


# ---------------------------------------------------------------------------------------------------- protocol
def stage_protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    if PROTOCOL_F.exists():
        raise SystemExit(f"{PROTOCOL_F} exists; it is not overwritten (delete it deliberately to change the protocol)")
    if (OUT / "results.json").exists():
        raise SystemExit("results.json exists: differences were computed; the protocol can no longer be blind")
    st = OUT / "band_selftest.json"
    if not st.exists() or not json.loads(st.read_text())["passed"]:
        raise SystemExit("band self-test missing or failed")
    level = RULES["bootstrap"]["level_each"]
    derived = {}
    for kind, ref in REF.items():
        reps, obs = group_obs(kind, [ref])
        ro = obs[ref]
        n = len(ro)
        tau = ro[0]["tau"]
        c0ref = float(np.mean([o["v2"] for o in ro]))
        dv = dict(reference=ref, n=n, replicas=reps, c0_reference=c0ref, curves={}, van_hove={})
        m_curves, se_curves = {}, {}
        for key, get in curve_specs():
            Y = np.array([get(o, c0ref) for o in ro])
            m, se = Y.mean(0), Y.std(0, ddof=1) / np.sqrt(n)
            c = crit(Y[:, 1:] - 0, level)[0]
            s_hat, s_lo, s_hi, resv, lid, lob = lobes(m, c * se, RULES["tolerance"]["scale_resolution"])
            dv["curves"][key] = dict(c_ref=c, lobes=[dict(l, t_start=float(tau[l["start"]]), t_end=float(tau[l["end"]]))
                                                    for l in lob],
                                     s_hat=s_hat.tolist(), s_lo=s_lo.tolist(), s_hi=s_hi.tolist(),
                                     resolved=resv.tolist(), S_glob=float(np.max(np.abs(m[1:]))))
            m_curves[key], se_curves[key] = m, se
        rc = ro[0]["r_centers"]
        m_gd = np.array([o["gd"] for o in ro]).mean(0)
        for j, tt in enumerate(BASE["observables"]["van_hove"]["times"]):
            if tt not in BASE["observables"]["van_hove"]["verdict_times"]:
                continue
            G = np.array([o["gd"][j] for o in ro]) - 1.0
            cnt = np.array([o["gd_counts"][j] for o in ro]).mean(0)
            keep = cnt >= RULES["tolerance"]["count_min_per_replica"]
            m, se = G.mean(0), G.std(0, ddof=1) / np.sqrt(n)
            mk = np.where(keep, m, np.nan)
            c = crit(G[:, keep], level)[0]
            # lobes over r: reuse lobes() on an array with a dummy leading point
            arr = np.concatenate([[np.nan], mk])
            band = np.concatenate([[np.nan], c * se])
            s_hat, s_lo, s_hi, resv, lid, lob = lobes(arr, np.nan_to_num(band), RULES["tolerance"]["scale_resolution"])
            dv["van_hove"][f"t{tt:g}"] = dict(index=j, c_ref=c, excluded_bins=np.where(~keep)[0].tolist(),
                                             excluded_r_max=float(rc[~keep].max()) if (~keep).any() else None,
                                             s_hat=s_hat[1:].tolist(), s_lo=s_lo[1:].tolist(), s_hi=s_hi[1:].tolist(),
                                             resolved=resv[1:].tolist(),
                                             lobes=[dict(l, r_start=float(rc[l["start"] - 1]), r_end=float(rc[l["end"] - 1]))
                                                    for l in lob],
                                             S_glob=float(np.nanmax(np.abs(mk))))
        dv["scalar_windows"] = derive_scalar_windows(m_curves, tau, rc, m_gd, se_curves, n)
        derived[kind] = dv
    p = json.loads(json.dumps(dict(BASE=BASE, RULES=RULES, derived=derived), default=float))
    inv = json.loads((OUT / "inventory.json").read_text())
    rec = dict(protocol=p, hash=proto_hash(p), base_hash=BASE_HASH, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               blind_for=RULES["prior_looks"]["blind"], not_blind_for=["vacf (C(0)-scaled, earlier dt study)",
                                                                      "van Hove t = 0 (g(r))"],
               derived_from="reference levels only (0.00125 primary, 0.005 secondary); no step-size difference of "
                            "these observables had been computed (results.json absent)",
               code_sha256={f.name: sha256(f, full=True) for f in CODE_FILES},
               data_sha256={c["file"]: c["npz_sha256"] for c in inv["chains"]},
               band_selftest=json.loads(st.read_text())["worst_coverage_used"], provenance=C.provenance())
    C.write_json(PROTOCOL_F, rec)
    print(f"wrote {PROTOCOL_F} (hash {rec['hash']})")


def load_protocol(allow_deviation=None, pfile=None):
    pfile = pfile or PROTOCOL_F
    if not pfile.exists():
        raise SystemExit(f"{pfile.name} missing: run the 'protocol' stage first")
    rec = json.loads(pfile.read_text())
    if rec["hash"] != proto_hash(rec["protocol"]):
        raise SystemExit("protocol.json was modified after it was written")
    if rec["base_hash"] != BASE_HASH:
        raise SystemExit("BASE settings in the code differ from the frozen protocol")
    now = {f.name: sha256(f, full=True) for f in CODE_FILES}
    changed = [k for k in now if now[k] != rec["code_sha256"].get(k)]
    if changed:
        if not allow_deviation:
            raise SystemExit(f"code changed after the protocol was frozen: {changed}; rerun with --allow-deviation")
        f = OUT / "protocol_deviations.json"
        db = json.loads(f.read_text()) if f.exists() else {"deviations": []}
        db["deviations"].append(dict(at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), files=changed, reason=allow_deviation,
                                     frozen={k: rec["code_sha256"].get(k) for k in changed},
                                     current={k: now[k] for k in changed}))
        C.write_json(f, db)
    return rec["protocol"], rec["hash"]


# ----------------------------------------------------------------------------------------------------- analysis
def cell(D, s_hat, s_lo, s_hi, resolved, S_glob, level):
    tol = RULES["tolerance"]["tol"]
    n = D.shape[0]
    m, se = D.mean(0), D.std(0, ddof=1) / np.sqrt(n)
    c, cb, cf, p = crit(D, level)
    up_pt, lo_pt = np.abs(m) + c * se, np.maximum(np.abs(m) - c * se, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_up = np.where(s_lo > 0, up_pt / s_lo, np.inf)
        e_lo = np.where(s_hi > 0, lo_pt / s_hi, 0.0)
        e_hat = np.where(s_hat > 0, np.abs(m) / s_hat, np.inf)
    allres = bool(np.all(resolved))
    out = dict(points=int(len(m)), c=c, c_boot=cb, c_bonf=cf, E_abs=float(np.max(np.abs(m))),
               E_abs_upper=float(np.max(up_pt)), E_abs_lower=float(np.max(lo_pt)),
               E_sig=float(np.max(e_hat)), E_sig_upper=float(np.max(e_up)), E_sig_lower=float(np.max(e_lo)),
               all_lobes_resolved=allres, deviation_resolved=bool(np.max(lo_pt) > 0),
               E_glob=float(np.max(np.abs(m)) / S_glob), E_glob_upper=float(np.max(up_pt) / S_glob),
               t_of_E_sig_upper_index=int(np.argmax(e_up)))
    if out["E_sig_lower"] > tol:
        out["verdict"] = "FAIL"
    elif out["E_sig_upper"] <= tol and allres:
        out["verdict"] = "PASS"
    else:
        out["verdict"] = "INCONCLUSIVE" + ("" if allres else " (unresolved lobe)")
    return out


def analyze_group(P, kind, pairs):
    dv = P["derived"][kind]
    level = RULES["bootstrap"]["level_each"]
    levels = sorted({dt for p in pairs for dt in p})
    reps, obs = group_obs(kind, levels)
    if reps != dv["replicas"]:
        raise RuntimeError("replica set differs from the frozen protocol")
    ref = dv["reference"]
    c0ref = dv["c0_reference"]
    tau = obs[ref][0]["tau"]
    rc = obs[ref][0]["r_centers"]
    res = dict(n_replicas=len(reps), replicas=reps, reference=ref, curves={}, van_hove={}, scalars={},
               tau=tau.tolist(), r_centers=rc.tolist())
    for key, get in curve_specs():
        d = dv["curves"][key]
        s_hat, s_lo, s_hi = (np.array(d[x], float) for x in ("s_hat", "s_lo", "s_hi"))
        resv = np.array(d["resolved"], bool)
        Y = {dt: np.array([get(o, c0ref) for o in obs[dt]]) for dt in levels}
        entry = {}
        for a, b in pairs:
            D = Y[a] - Y[b]
            cells = {}
            for wn, w in BASE["windows"]["time"].items():
                msk = window_points(tau, w)
                cells[wn] = cell(D[:, msk], s_hat[msk], s_lo[msk], s_hi[msk], resv[msk], d["S_glob"], level)
                cells[wn]["t_of_E_sig_upper"] = float(tau[msk][cells[wn].pop("t_of_E_sig_upper_index")])
            entry[f"{a}_vs_{b}"] = cells
        entry["_mean"] = {str(dt): Y[dt].mean(0).tolist() for dt in levels}
        entry["_diff"] = {f"{a}_vs_{b}": dict(mean=(Y[a] - Y[b]).mean(0).tolist(),
                                              se=((Y[a] - Y[b]).std(0, ddof=1) / np.sqrt(len(reps))).tolist(),
                                              c_all=entry[f"{a}_vs_{b}"]["ALL"]["c"]) for a, b in pairs}
        res["curves"][key] = entry
    for tk, d in dv["van_hove"].items():
        j = d["index"]
        keep = np.ones(len(rc), bool)
        keep[d["excluded_bins"]] = False
        s_hat, s_lo, s_hi = (np.array(d[x], float) for x in ("s_hat", "s_lo", "s_hi"))
        resv = np.array(d["resolved"], bool)
        G = {dt: np.array([o["gd"][j] for o in obs[dt]]) for dt in levels}
        entry = {}
        for a, b in pairs:
            D = G[a] - G[b]
            cells = {}
            for rn, w in BASE["windows"]["radius"].items():
                msk = r_window(rc, w) & keep
                if msk.sum() == 0:
                    cells[rn] = dict(verdict="no bins (all below the count threshold)")
                    continue
                cells[rn] = cell(D[:, msk], s_hat[msk], s_lo[msk], s_hi[msk], resv[msk], d["S_glob"], level)
                cells[rn]["r_of_E_sig_upper"] = float(rc[msk][cells[rn].pop("t_of_E_sig_upper_index")])
            entry[f"{a}_vs_{b}"] = cells
        entry["_mean"] = {str(dt): G[dt].mean(0).tolist() for dt in levels}
        entry["_diff"] = {f"{a}_vs_{b}": dict(mean=(G[a] - G[b]).mean(0).tolist(),
                                              se=((G[a] - G[b]).std(0, ddof=1) / np.sqrt(len(reps))).tolist(),
                                              c_all=entry[f"{a}_vs_{b}"]["ALL"].get("c")) for a, b in pairs}
        res["van_hove"][tk] = entry
    # t = 0 van Hove (plot only)
    j0 = BASE["observables"]["van_hove"]["times"].index(0.0)
    res["van_hove_t0_plot"] = {str(dt): np.array([o["gd"][j0] for o in obs[dt]]).mean(0).tolist() for dt in levels}
    # scalars
    sw = dv["scalar_windows"]
    sc = {dt: [scalars_of(o, c0ref, sw) for o in obs[dt]] for dt in levels}
    res["per_replica_scalars"] = {str(r): {str(dt): sc[dt][i] for dt in levels} for i, r in enumerate(reps)}
    for a, b in pairs:
        tab = {}
        for name in sc[b][0]:
            xa = np.array([s[name] for s in sc[a]], float)
            xb = np.array([s[name] for s in sc[b]], float)
            tab[name] = scalar_verdict(name, xa, xb, obs[b], c0ref)
        res["scalars"][f"{a}_vs_{b}"] = tab
    res["vccf_allpair_mean_ref"] = (-np.array([o["vacf"] for o in obs[ref]]).mean(0) / (N - 1)).tolist()
    return res


SHAPE_SCALAR_PREFIX = {"vacf": ("vacf_Zmin", "vacf_tmin", "vacf_tzero", "vacf_D2"),
                       "vacf_amp": ("vacf_D2",)}


def scalar_conflicts(scal, key):
    """Shape scalars of an observable that FAIL (amplitude scalars C0 / CL0 / CT0 excluded)."""
    out = []
    for name, x in scal.items():
        if x.get("verdict") != "FAIL" or name.startswith(("vacf_C0", "CL0_", "CT0_")) or name.endswith("_C0"):
            continue
        if key in ("vacf", "vacf_amp"):
            if name in SHAPE_SCALAR_PREFIX[key]:
                out.append(name)
        elif key.startswith("vh_t"):
            if name.startswith(key + "_"):
                out.append(name)
        elif name.startswith(key + "_"):
            out.append(name)
    return out


def display_support(R):
    pr, se = R["primary"], R["secondary"]
    table = {}
    keys = list(pr["curves"]) + [f"vh_{t}" for t in pr["van_hove"]]
    for dt in (0.01, 0.005):
        for key in keys:
            if key.startswith("vh_"):
                tk = key[3:]
                P_ = pr["van_hove"][tk]
                S_ = se["van_hove"][tk] if dt == 0.01 else None
                windows = BASE["windows"]["radius"]
            else:
                P_ = pr["curves"][key]
                S_ = se["curves"][key] if dt == 0.01 else None
                windows = BASE["windows"]["time"]
            confl = scalar_conflicts(pr["scalars"][f"{dt}_vs_0.00125"], key.replace("vh_t", "vh_t"))
            if dt == 0.01:
                confl += [f"secondary:{x}" for x in scalar_conflicts(se["scalars"]["0.01_vs_0.005"], key)]
            for wn in windows:
                prim = P_[f"{dt}_vs_0.00125"][wn]["verdict"]
                refv = P_["0.0025_vs_0.00125"][wn]["verdict"]
                sec = S_["0.01_vs_0.005"][wn]["verdict"] if S_ else "n/a"
                if prim.startswith("FAIL") or sec.startswith("FAIL") or confl:
                    why = [w for w, v in (("primary", prim), ("secondary 0.01 vs 0.005", sec)) if v.startswith("FAIL")]
                    why += [f"scalar {c}" for c in confl]
                    v = ("NOT SUPPORTED (CONFLICT: " if prim == "PASS" else "NOT SUPPORTED (") + ", ".join(why) + ")"
                elif prim == "PASS" and refv == "PASS":
                    v = "SUPPORTED"
                elif prim == "PASS":
                    v = "INCONCLUSIVE (reference not adequate: 0.0025 vs 0.00125 " + refv + ")"
                else:
                    v = "INCONCLUSIVE (" + prim + ")"
                ex = None
                c1 = P_[f"{dt}_vs_0.00125"][wn]
                c2 = P_["0.0025_vs_0.00125"][wn]
                if "E_sig_upper" in c1 and "E_sig_upper" in c2:
                    ex = c1["E_sig_upper"] + c2["E_sig_upper"]
                table[f"{dt}|{key}|{wn}"] = dict(dt=dt, observable=key, window=wn, support=v, primary=prim,
                                                 reference_sensitivity=refv, secondary=sec,
                                                 exact_bound_p1_descriptive=ex)
    return table


def figures(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"0.00125": "#222222", "0.0025": "#1baf7a", "0.005": "#2a78d6", "0.01": "#eb6834"}
    pr = R["primary"]
    tau = np.array(pr["tau"])
    n = pr["n_replicas"]
    groups = [("vacf", ["vacf", "vacf_amp"]), ("vccf", [k for k in pr["curves"] if k.startswith("vccf")]),
              ("currents", [k for k in pr["curves"] if k.startswith(("CL_", "CT_"))])]
    for name, keys in groups:
        fig, ax = plt.subplots(2, len(keys), figsize=(3.9 * len(keys), 7), squeeze=False)
        for j, k in enumerate(keys):
            cur = pr["curves"][k]
            for dt, y in cur["_mean"].items():
                ax[0, j].plot(tau, y, color=cols[dt], lw=1.1, label=f"dt {dt}")
            if k.startswith("vccf"):
                ax[0, j].plot(tau, pr["vccf_allpair_mean_ref"], color="#999999", ls=":", lw=1,
                              label="all-pair mean -Z/(N-1)\n(not a per-bin baseline)")
                ax[0, j].axhline(-1 / (N - 1), color="#cccccc", lw=0.8, ls="--")
            ax[0, j].axhline(0, color="#bbbbbb", lw=0.6)
            ax[0, j].set_title(k, fontsize=9)
            for c, d in cur["_diff"].items():
                m, se = np.array(d["mean"]), np.array(d["se"])
                dt = c.split("_vs_")[0]
                hp = stats.t.ppf(0.975, n - 1) * se
                ax[1, j].plot(tau, m, color=cols[dt], lw=1.0, label=c.replace("_vs_", " - "))
                ax[1, j].fill_between(tau, m - hp, m + hp, color=cols[dt], alpha=0.16, lw=0)
                ax[1, j].plot(tau, m + d["c_all"] * se, color=cols[dt], lw=0.6, ls="--")
                ax[1, j].plot(tau, m - d["c_all"] * se, color=cols[dt], lw=0.6, ls="--")
            ax[1, j].axhline(0, color="#bbbbbb", lw=0.6)
            ax[1, j].set_xlabel("t")
        ax[0, 0].legend(fontsize=7)
        ax[1, 0].legend(fontsize=7)
        ax[1, 0].set_ylabel("paired difference\nshaded: pointwise 95% t; dashed: simultaneous over (0,2]")
        fig.suptitle(f"lj075 law A, {name}: dt vs 0.00125, {n} paired replicas (trajectories with ranks 12/5)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"{name}_dt.png", dpi=110)
        plt.close(fig)
    rc = np.array(pr["r_centers"])
    ts = list(pr["van_hove"])
    fig, ax = plt.subplots(2, len(ts) + 1, figsize=(3.4 * (len(ts) + 1), 6.5), squeeze=False)
    for dt, y in pr["van_hove_t0_plot"].items():
        ax[0, 0].plot(rc, y, color=cols[dt], lw=1.0, label=f"dt {dt}")
    ax[0, 0].set_title("g_d(r,0) = g(r) (plot only)", fontsize=9)
    ax[1, 0].axis("off")
    for j, t in enumerate(ts, start=1):
        e = pr["van_hove"][t]
        for dt, y in e["_mean"].items():
            ax[0, j].plot(rc, y, color=cols[dt], lw=1.0)
        ax[0, j].axhline(1, color="#bbbbbb", lw=0.6)
        ax[0, j].set_title(f"g_d(r, {t[1:]})", fontsize=9)
        for c, d in e["_diff"].items():
            m, se = np.array(d["mean"]), np.array(d["se"])
            dt = c.split("_vs_")[0]
            hp = stats.t.ppf(0.975, n - 1) * se
            ax[1, j].plot(rc, m, color=cols[dt], lw=1.0, label=c.replace("_vs_", " - "))
            ax[1, j].fill_between(rc, m - hp, m + hp, color=cols[dt], alpha=0.16, lw=0)
            if d.get("c_all"):
                ax[1, j].plot(rc, m + d["c_all"] * se, color=cols[dt], lw=0.6, ls="--")
                ax[1, j].plot(rc, m - d["c_all"] * se, color=cols[dt], lw=0.6, ls="--")
        ax[1, j].axhline(0, color="#bbbbbb", lw=0.6)
        ax[1, j].set_xlabel("r")
    ax[0, 0].legend(fontsize=7)
    ax[1, 1].legend(fontsize=7)
    ax[1, 1].set_ylabel("paired difference (g units)")
    fig.suptitle(f"lj075 law A, distinct van Hove: dt vs 0.00125, {n} paired replicas", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "van_hove_dt.png", dpi=110)
    plt.close(fig)


def tag(v):
    return "P" if v == "PASS" else "F" if v.startswith("FAIL") else ("I*" if "unresolved" in v else "I")


def conclusions(R, support, path=None):
    L_ = ["# lj075 law A: dt comparison of VACF, VCCF, C_L / C_T and distinct van Hove",
          "",
          "Generated by lj075_observable_dt.py from results.json (protocol.json frozen before any difference). "
          "Supplementary analysis: every earlier dt-study verdict (REPORT.md section 3) stands unchanged.",
          "",
          "Curve cells: verdict of E_sig (sup |paired mean difference| / scale of the reference sign lobe at each "
          "point, simultaneous bound at 97.5% for the difference x 97.5% for the scale), tolerance 1%. P = PASS, "
          "F = FAIL, I = INCONCLUSIVE, I* = an unresolved lobe in the window. Numbers: E_sig estimate [lower, upper] "
          "in %; E_abs = sup |difference| (fraction of the t = 0 value; VCCF: of <|v|^2>; g_d: g units). Verdicts "
          "are against the reference level (0.00125 primary, 0.005 secondary), not against the exact dynamics.", ""]
    for kind in ("primary", "secondary"):
        g = R[kind]
        pairs = [c for c in next(iter(g["curves"].values())) if not c.startswith("_")]
        L_ += [f"## {kind}: {g['n_replicas']} replicas, reference {g['reference']}", "",
               "| observable | window | " + " | ".join(p.replace("_vs_", " vs ") for p in pairs) + " |",
               "|" + "---|" * (2 + len(pairs))]
        rows = [(k, BASE["windows"]["time"], e) for k, e in g["curves"].items()] + \
               [(f"g_d {t}", BASE["windows"]["radius"], e) for t, e in g["van_hove"].items()]
        for key, wins, e in rows:
            for wn, w in wins.items():
                row = [key, f"{wn} ({w[0]:.2f}, {w[1]:.2f}]"]
                for p in pairs:
                    c = e[p][wn]
                    if "E_sig" not in c:
                        row.append(c["verdict"])
                        continue
                    row.append(f"{tag(c['verdict'])} {100 * c['E_sig']:.2f} [{100 * c['E_sig_lower']:.2f}, "
                               f"{100 * c['E_sig_upper']:.2f}]; E_abs {c['E_abs']:.1e}")
                L_.append("| " + " | ".join(row) + " |")
        L_ += ["", f"### scalars ({kind}): absolute paired difference [97.5% CI] and relative to the stated scale", ""]
        sp = list(g["scalars"])
        L_ += ["| scalar | scale | " + " | ".join(p.replace("_vs_", " vs ") for p in sp) + " |",
               "|" + "---|" * (2 + len(sp))]
        for name in g["scalars"][sp[0]]:
            x0 = g["scalars"][sp[0]][name]
            row = [name, f"{x0.get('scale', float('nan')):.4g} ({x0.get('scale_def', '-')})"]
            for p in sp:
                x = g["scalars"][p][name]
                if "abs_ci" in x:
                    row.append(f"{x['abs_diff']:+.3g} [{x['abs_ci'][0]:+.3g}, {x['abs_ci'][1]:+.3g}]; rel "
                               f"{100 * x['rel_diff']:+.2f}% [{100 * x['rel_ci'][0]:+.2f}, {100 * x['rel_ci'][1]:+.2f}] "
                               f"**{x['verdict']}**")
                else:
                    row.append(x["verdict"])
            L_.append("| " + " | ".join(row) + " |")
        L_.append("")
    L_ += ["## display support (shape-normalized display; E_sig verdicts and scalar FAILs only)", "",
           "| dt | observable | window | support | primary | 0.0025 vs 0.00125 | secondary 0.01 vs 0.005 | "
           "descriptive first-order bound vs exact (E_sig upper, %) |", "|---|---|---|---|---|---|---|---|"]
    for k, v in support.items():
        ex = v["exact_bound_p1_descriptive"]
        L_.append(f"| {v['dt']} | {v['observable']} | {v['window']} | {v['support']} | {v['primary']} | "
                  f"{v['reference_sensitivity']} | {v['secondary']} | {'-' if ex is None else f'{100 * ex:.2f}'} |")
    (path or OUT / "conclusions_table.md").write_text("\n".join(L_) + "\n")


def stage_analyze(allow_deviation=None, pfile=None, suffix=""):
    P, ph = load_protocol(allow_deviation, pfile)
    cpu0, wall0 = C.cpu_seconds(), time.perf_counter()
    pp = BASE["comparisons"]["primary"]
    R = dict(protocol_hash=ph, provenance=C.provenance(),
             primary=analyze_group(P, "primary", [tuple(x) for x in pp["pairs"] + pp["reference_sensitivity"]]),
             secondary=analyze_group(P, "secondary", [tuple(x) for x in BASE["comparisons"]["secondary"]["pairs"]]))
    R["display_support"] = display_support(R)
    R["protocol_file"] = (pfile or PROTOCOL_F).name
    C.write_json(OUT / f"results{suffix}.json", R)
    if suffix:
        conclusions(R, R["display_support"], OUT / f"conclusions_table{suffix}.md")
        log_time(dict(stage=f"analyze{suffix}", cpu_s=C.cpu_seconds() - cpu0, wall_s=time.perf_counter() - wall0))
        return
    arrays = {}
    for m, f, kind in chains():
        for dt in m["levels"]:
            o = load_level(f, dt)
            for k in ["vacf", "vacf_un", "vccf", "gd"] + [x for x in o if x[:3] in ("CL_", "CT_")]:
                arrays[f"{kind}_rep{m['replica']}_dt{dt:g}_{k}"] = np.asarray(o[k], np.float32)
    arrays["tau"] = np.array(R["primary"]["tau"])
    arrays["r_centers"] = np.array(R["primary"]["r_centers"])
    np.savez_compressed(OUT / "per_replica_observables.npz", **arrays)
    figures(R)
    conclusions(R, R["display_support"])
    log_time(dict(stage="analyze", cpu_s=C.cpu_seconds() - cpu0, wall_s=time.perf_counter() - wall0))


# ---------------------------------------------------------------------------------------------------- follow-up plan
PLAN_TARGETS = [("vacf", "head lobe", 0.01, 0.11), ("vacf", "dip lobe", 0.12, 0.5), ("vccf_b1", "first-shell peak", 0.02, 0.34),
                ("CL_k1", "first lobe", 0.01, 0.19), ("CT_k1", "first lobe", 0.01, 0.41),
                ("CL_k2", "first lobe", 0.01, 0.13), ("CT_k2", "first lobe", 0.01, 0.31),
                ("CL_k4", "first lobe", 0.01, 0.10), ("CT_k4", "first lobe", 0.01, 0.22),
                ("vacf", "t in (0.5, 1]", 0.51, 1.0), ("vacf", "t in (1, 2]", 1.01, 2.0)]


def stage_plan():
    """Planning only (post hoc, not a verdict): paired-difference spread of single-origin bursts (origin t0 = 0, the
    shared initial state of every coupled chain) versus the all-origin estimates of the analysis, and the number of
    independent bursts needed for a 1%-of-lobe decision."""
    rec = json.loads(PROTOCOL_F.read_text())["protocol"]["derived"]["primary"]
    files = [(m, f) for m, f, k in chains() if k == "primary"]
    lev = (0.00125, 0.005, 0.01)
    burst = {dt: [] for dt in lev}
    for m, f in files:
        with np.load(f) as z:
            for dt in lev:
                burst[dt].append(level_observables(z[f"Q_{dt}"], z[f"V_{dt}"], dt, np.array([0.0])))
    R = json.loads((OUT / "results.json").read_text())["primary"]
    tau = np.array(R["tau"])
    timing = json.loads((OUT / "timing.json").read_text())["summary"]
    per_step = np.mean([m["wall_s"] / sum(m["steps"]) for m, _ in files])           # coupled 4-level chain, per step
    out = dict(note=stage_plan.__doc__, chain_wall_s_per_step_mean=per_step, targets=[])
    for key, label, a, b in PLAN_TARGETS:
        get = dict(curve_specs())[key]
        msk = (tau >= a - 1e-9) & (tau <= b + 1e-9)
        S = float(np.max(np.abs(np.array(rec["curves"][key]["s_hat"])[msk])))
        for dt in (0.01, 0.005):
            Db = np.array([get(o1, 1.0) - get(o0, 1.0) for o1, o0 in zip(burst[dt], burst[0.00125])])[:, msk]
            sd_b = Db.std(0, ddof=1)
            m_b = np.abs(Db.mean(0))
            Da = np.array(R["curves"][key]["_diff"][f"{dt}_vs_0.00125"]["se"])[msk] * np.sqrt(R["n_replicas"])
            p = int(msk.sum())
            need = {}
            for lab, mu in (("if the true difference is 0", 0.0), ("at the burst mean difference", float(m_b.max()))):
                n_ok = None
                for n in range(3, 5001):
                    c = stats.t.ppf(1 - 0.025 / (2 * p), n - 1)
                    if mu + c * sd_b.max() / np.sqrt(n) <= 0.01 * S or (mu > 0.01 * S and mu - c * sd_b.max() / np.sqrt(n) > 0.01 * S):
                        n_ok = n
                        break
                need[lab] = n_ok
            T_b = b + 0.01
            steps = T_b * sum(1 / d for d in LEVELS_PLAN)
            cost_h = (steps * per_step + 13.0) / 3600
            out["targets"].append(dict(observable=key, feature=label, t_range=[a, b], lobe_scale=S, dt=dt,
                                       sd_paired_burst_max=float(sd_b.max()), sd_paired_all_origins_max=float(Da.max()),
                                       burst_mean_abs_diff_max=float(m_b.max()), tolerance_abs=0.01 * S,
                                       bursts_needed=need, burst_length=T_b, core_h_per_burst=cost_h,
                                       core_h_needed={k: (None if v is None else v * cost_h) for k, v in need.items()}))
            print(key, label, dt, "sd burst %.2e vs all-origins %.2e; tol %.2e; n %s" % (sd_b.max(), Da.max(), 0.01 * S, need))
    C.write_json(OUT / "followup_plan.json", out)


# -------------------------------------------------------------------------------------------------------- timing
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
    ap.add_argument("stage", choices=["inventory", "compute", "selftest", "protocol", "analyze", "timing", "plan"])
    ap.add_argument("--allow-deviation", default=None, help="reason (logged) for running analyze after a code change")
    ap.add_argument("--protocol-file", default=None, help="analyze with another frozen protocol (e.g. v2)")
    ap.add_argument("--suffix", default="", help="output suffix for results / conclusions (e.g. _v2)")
    args = ap.parse_args()
    if args.stage == "analyze":
        stage_analyze(args.allow_deviation, OUT / args.protocol_file if args.protocol_file else None, args.suffix)
    else:
        dict(inventory=stage_inventory, compute=stage_compute, selftest=stage_selftest, protocol=stage_protocol,
             timing=stage_timing, plan=stage_plan)[args.stage]()


if __name__ == "__main__":
    main()
