#!/usr/bin/env python3
"""
Static verification of the PPPM friction operator and its Lanczos matrix functions at the lj075 state point
(rho 0.75, kT 1.0, N = 256, L = 6.98864; laws A and B), on representative and actually visited configurations.

Error definitions (as in verify_production_configs.py, extended):
  operator   eps_op = ||Gamma_h - Gamma_ref||_2 / ||Gamma_ref||_2 (spectral norm, dense, N = 256), Gamma_ref the full
             periodic lattice sum v2.LatticeFriction (k = 0 retained); budget A 2e-7, B 5e-8 (the costopt budgets).
             Also reported: max_v ||(Gamma_h - Gamma_ref) v|| / ||Gamma_ref v|| over the probe vectors, the relative
             error of the 12 lowest eigenvalues on Range(Pi), and the Frobenius-norm error.
             The operator error is NOT a physical-observable error; physical effects of dt are judged separately.
  Lanczos    for one fixed Gamma_h, the production routine (direct-start Lanczos, full reorthogonalisation) at rank r
             against the dense matrix function f(Gamma_h) on Range(Pi) (pure Krylov error):
               overall  ||f_r - f|| / ||f||
               kmin     max over the three k = 2 pi/L axis vectors of |sum_i e^{ik q_i}(f_r - f)_i| / |sum_i e^{ik q_i} f_i|
               low3/12  ||U_low^T (f_r - f)|| / ||U_low^T f||, U_low the 3 / 12 lowest eigenvectors of Gamma_h
             functions: noise sqrt(kT m (1 - e^{-2 dt l/m})), damping e^{-dt l/m}, coupling maps a, b of
             lj075_coupled.py (step dt/2 -> dt), for dt in 0.00125 0.0025 0.005 0.01.
             probes: Gaussian (noise input), Maxwell momenta (damping input), the state's actual momenta, longitudinal
             and transverse plane waves at the smallest k (4 wave vectors), slow-mode-dominated vectors (random
             combination of the 12 lowest eigenvectors plus 10% random).
             Rank rule (as costopt_hiacc): smallest rank whose four measures are <= budget/10 on every probe and state.
             Damping and coupling maps are close to a constant times the identity (f(0) = 1 resp. 1/sqrt 2), which
             Lanczos reproduces exactly; their errors are therefore ALSO reported relative to the increment
             ||(f - f(0))(Gamma) v|| ("overall_inc"), with the corresponding required ranks.
  structure  symmetry, momentum null space Gamma 1 = 0, positive definiteness on Range(Pi) (Gamma_h and Gamma_ref),
             literal matvec vs dense assembly, k = 0 retained.
  FDT        friction-noise consistency at the chosen ranks: ||R_r(R_r v) + S_r(S_r v)/(kT m) - v|| / ||v|| (same op);
             dense: exact identity R^2 + S^2/(kT m) = I on Range(Pi).
  combined   O-step (R_r p + S_r xi) at the chosen ranks with Gamma_h vs the dense O-step with Gamma_ref.
  reference  LatticeFriction pair tensors at this L vs a brute-force image sum (image_sum_tensors, all images within
             r_tail + sqrt(3) L/2); real-space pair set of the KD-tree path vs the image enumeration.

    python3 lj075_verify.py --laws A --states 'raw/states/*.npz' [--tag representative] [--pppm-A xi s eta p]
Results: lj075_results/verify_<tag>_<law>.json (cached by a key of everything they depend on).
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "toy_models"))
import lj075_common as C  # noqa: E402
import prl_toy_models as v2  # noqa: E402
import radial_kernels as rk  # noqa: E402
import toy_configs as tc  # noqa: E402
from lj075_coupled import coupling_funcs  # noqa: E402
from test_lanczos_fdt import DenseRef, lanczos, proj, real_pairs, tridiag  # noqa: E402
from test_true_dynamics import lanczos_apply  # noqa: E402

CONFIG = "lj_rho0.75_kT1.0_costopt"
DTS = (0.00125, 0.0025, 0.005, 0.01)
RANKS = (2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56)
RMAX = max(RANKS)
PROTOCOL = dict(version=3, dts=DTS, ranks=RANKS, probes="4 gauss, 4 maxwell, actual p, 4 k x (L,T) plane waves, "
                "2 slow-mode", rule="max(overall, kmin, low3, low12) <= budget/10")


def lattice_matrix(lat, q, chunk=8192):
    i, j, dr, _ = v2.pair_geometry(q, lat.box)
    K = np.concatenate([lat.pair_tensors(dr[c:c + chunk]) for c in range(0, len(dr), chunk)])
    return v2.assemble_friction(i, j, K, len(q))


def reference_selfcheck(lat, L, law, mdl, rng, n=64):
    """LatticeFriction (Chebyshev far field) vs brute-force image sum at this L."""
    dr = rng.uniform(-L / 2, L / 2, (n, 3))
    reach = lat.r_tail + np.sqrt(3) * L / 2
    shifts = v2.lattice_shifts(lat.box, reach)
    brute = v2.image_sum_tensors(dr, shifts, kernel=law, gamma=mdl["gamma"], kappa=mdl["kappa"], r_ref=mdl["r_ref"])
    fast = lat.pair_tensors(dr)
    scale = np.abs(brute).max()
    return dict(n_displacements=n, n_shifts=len(shifts), max_abs_err_over_max=float(np.abs(fast - brute).max() / scale),
                max_rel_err_per_pair=float(np.max(np.linalg.norm(fast - brute, axis=(1, 2))
                                                  / np.linalg.norm(brute, axis=(1, 2)))))


def plane_waves(q, L):
    out = {}
    for n in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)):
        k = 2 * np.pi / L * np.array(n, float)
        kh = k / np.linalg.norm(k)
        e = np.cross(kh, [0, 0, 1.0] if abs(kh[2]) < 0.9 else [1.0, 0, 0])
        e /= np.linalg.norm(e)
        c = np.cos(q @ k)[:, None]
        out[f"pwL_{''.join(map(str, n))}"] = proj((c * kh).ravel())
        out[f"pwT_{''.join(map(str, n))}"] = proj((c * e).ravel())
    return out


def functions(kT, m):
    F = {}
    for dt in DTS:
        F[f"noise_{dt}"] = (lambda dt: (lambda l: np.sqrt(kT * m * -np.expm1(-2 * dt * l / m))))(dt)
        F[f"damp_{dt}"] = (lambda dt: (lambda l: np.exp(-dt * l / m)))(dt)
        if dt > DTS[0]:
            a, b = coupling_funcs(dt / 2, m)
            F[f"coupleA_{dt}"], F[f"coupleB_{dt}"] = a, b
    return F


def verify_state(r, fast, lat, q, p_actual, rng, xi_actual=None):
    N, L = r["N"], r["L"]
    kT, m = r["model"]["kT"], r["model"]["mass"]
    op = fast.gamma_h(q)
    t0 = time.perf_counter()
    Gh = op.dense()
    Gr = lattice_matrix(lat, q % L)
    Dh, Dr = DenseRef(Gh, N), DenseRef(Gr, N)
    T = np.kron(np.ones((N, 1)), np.eye(3))
    probes = {f"gauss{k}": proj(rng.standard_normal(3 * N)) for k in range(4)}
    probes.update({f"maxwell{k}": proj(rng.normal(0, np.sqrt(kT * m), 3 * N)) for k in range(4)})
    if p_actual is not None:
        probes["actual_p"] = proj(np.asarray(p_actual, float).ravel())
    if xi_actual is not None:
        probes["actual_xi"] = proj(np.asarray(xi_actual, float).ravel())
    probes.update(plane_waves(q % L, L))
    for k in range(2):
        c = rng.standard_normal(12)
        v = Dh.U[:, :12] @ c
        probes[f"slow{k}"] = proj(v / np.linalg.norm(v) + 0.1 * rng.standard_normal(3 * N) / np.sqrt(3 * N))
    diff = Gh - Gr
    # k != 0 part of the reference: remove the analytically exact k = 0 term c0 (N I - 1 1^T) (x) I_3, c0 = ghat(0)/(3V)
    c0 = lat.zero_mode
    Gk0 = c0 * np.kron(N * np.eye(N) - np.ones((N, N)), np.eye(3))
    gv = max(float(np.linalg.norm(diff @ v) / np.linalg.norm(Gr @ v)) for v in probes.values())
    rec = dict(op_err=float(np.linalg.norm(diff, 2) / np.linalg.norm(Gr, 2)),
               op_err_fro=float(np.linalg.norm(diff) / np.linalg.norm(Gr)),
               op_err_vs_k_nonzero_part=float(np.linalg.norm(diff, 2) / np.linalg.norm(Gr - Gk0, 2)),
               k0_shift=float(N * c0),
               op_err_gv_probes=gv,
               op_err_low12_eigs=float(np.max(np.abs(Dh.lam[:12] - Dr.lam[:12]) / Dr.lam[:12])),
               sym_resid=float(np.linalg.norm(Gh - Gh.T) / np.linalg.norm(Gh)),
               null_resid=float(np.abs(Gh @ T).max() / np.abs(Gh).max()),
               null_resid_ref=float(np.abs(Gr @ T).max() / np.abs(Gr).max()),
               literal_vs_dense=float(max(np.linalg.norm(op.matvec(v) - Gh @ v) / np.linalg.norm(Gh @ v)
                                          for v in list(probes.values())[:4])),
               lam_min=float(Dh.lam[0]), lam_max=float(Dh.lam[-1]), lam_min_ref=float(Dr.lam[0]),
               lam_max_ref=float(Dr.lam[-1]), n_real_pairs=int(len(op.w)), dense_s=time.perf_counter() - t0)
    ph = np.exp(1j * 2 * np.pi / L * (q % L))
    funcs = functions(kT, m)
    lows = {3: Dh.U[:, :3], 12: Dh.U[:, :12]}
    tab = {fn: {rk_: dict(overall=0.0, kmin=0.0, low3=0.0, low12=0.0) for rk_ in RANKS} for fn in funcs}
    tab_inc = {fn: {rk_: 0.0 for rk_ in RANKS} for fn in funcs}
    const = {fn: (1.0 if fn.startswith("damp") else (1 / np.sqrt(2) if fn.startswith("couple") else 0.0)) for fn in funcs}
    worst_probe = {fn: {} for fn in funcs}
    ritz_min = np.inf
    t1 = time.perf_counter()
    for pname, v in probes.items():
        Lz = lanczos(op, v, RMAX)
        eig = {}
        for rk_ in RANKS:
            ss = min(rk_, Lz["s"])
            th, U = np.linalg.eigh(tridiag(Lz["al"], Lz["be"], ss))
            eig[rk_] = (ss, th, U)
            ritz_min = min(ritz_min, th[0])
        for fn, f in funcs.items():
            ref = Dh.apply(f, v)
            inc = np.linalg.norm(ref - const[fn] * proj(v))
            Jr = ph.T @ ref.reshape(N, 3)
            lowr = {k: np.linalg.norm(B.T @ ref) for k, B in lows.items()}
            for rk_ in RANKS:
                ss, th, U = eig[rk_]
                if th[0] <= 0:
                    raise RuntimeError("negative Ritz value")
                d = proj(Lz["nz"] * (Lz["Q"][:ss].T @ (U @ (f(th) * U[0])))) - ref
                e = dict(overall=float(np.linalg.norm(d) / np.linalg.norm(ref)),
                         kmin=float(np.max(np.linalg.norm(ph.T @ d.reshape(N, 3), axis=1) / np.linalg.norm(Jr, axis=1))),
                         low3=float(np.linalg.norm(lows[3].T @ d) / lowr[3]),
                         low12=float(np.linalg.norm(lows[12].T @ d) / lowr[12]))
                tab_inc[fn][rk_] = max(tab_inc[fn][rk_], float(np.linalg.norm(d) / inc))
                t = tab[fn][rk_]
                for kk, vv in e.items():
                    if vv > t[kk]:
                        t[kk] = vv
                        worst_probe[fn].setdefault(str(rk_), {})[kk] = pname
    rec.update(krylov={fn: {str(k): v for k, v in t.items()} for fn, t in tab.items()}, worst_probe=worst_probe,
               krylov_increment={fn: {str(k): v for k, v in t.items()} for fn, t in tab_inc.items()},
               ritz_min=float(ritz_min), lanczos_s=time.perf_counter() - t1, probes=list(probes))
    return rec, op, Dh, Dr, probes


def required(krylov_by_state, budget, factor=0.1):
    out = {}
    for fn in next(iter(krylov_by_state.values())):
        out[fn] = next((k for k in RANKS if all(_mx(st[fn][str(k)]) <= factor * budget
                                                 for st in krylov_by_state.values())), None)
    return out


def _mx(x):
    return max(x.values()) if isinstance(x, dict) else x


def chosen_checks(r, op, Dh, Dr, probes, rn, rd, rc):
    """O-step, FDT consistency and coupling at the chosen ranks, every dt."""
    kT, m, N = r["model"]["kT"], r["model"]["mass"], r["N"]
    out = {}
    pv = ([probes["actual_p"]] if "actual_p" in probes else []) + [v for k, v in probes.items() if k.startswith("maxwell")]
    xv = ([probes["actual_xi"]] if "actual_xi" in probes else []) + [v for k, v in probes.items() if k.startswith("gauss")]
    pv, xv = pv[:3], xv[:3]                       # actual O-step inputs first where available
    for dt in DTS:
        fn = lambda l, dt=dt: np.sqrt(kT * m * -np.expm1(-2 * dt * l / m))   # noqa: E731
        fd = lambda l, dt=dt: np.exp(-dt * l / m)                            # noqa: E731
        comb, fdt, kry, comb_i, kry_i, fdt_i, op_only_i = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for p, x in zip(pv, xv):
            o_l = lanczos_apply(op, p, rd, fd)[0] + lanczos_apply(op, x, rn, fn)[0]
            o_h = Dh.apply(fd, p) + Dh.apply(fn, x)
            o_r = Dr.apply(fd, p) + Dr.apply(fn, x)
            comb = max(comb, float(np.linalg.norm(o_l - o_r) / np.linalg.norm(o_r)))
            kry = max(kry, float(np.linalg.norm(o_l - o_h) / np.linalg.norm(o_h)))
            # relative to the O-step increment p_new - p (the part the friction actually changes)
            inc_r = np.linalg.norm(o_r - p)
            comb_i = max(comb_i, float(np.linalg.norm(o_l - o_r) / inc_r))
            kry_i = max(kry_i, float(np.linalg.norm(o_l - o_h) / np.linalg.norm(o_h - p)))
            op_only_i = max(op_only_i, float(np.linalg.norm(o_h - o_r) / inc_r))
            Rv = lanczos_apply(op, p, rd, fd)[0]
            Sv = lanczos_apply(op, p, rn, fn)[0]
            resid = lanczos_apply(op, Rv, rd, fd)[0] + lanczos_apply(op, Sv, rn, fn)[0] / (kT * m) - p
            fdt = max(fdt, float(np.linalg.norm(resid) / np.linalg.norm(p)))
            fdt_i = max(fdt_i, float(np.linalg.norm(resid) / np.linalg.norm(Dh.apply(lambda l: fn(l) ** 2, p) / (kT * m))))
        dense_fdt = float(np.max(np.abs(fd(Dh.lam) ** 2 + fn(Dh.lam) ** 2 / (kT * m) - 1)))
        out[str(dt)] = dict(ostep_krylov=kry, ostep_vs_reference=comb, fdt_lanczos=fdt, fdt_dense=dense_fdt,
                            ostep_krylov_increment=kry_i, ostep_vs_reference_increment=comb_i,
                            ostep_pppm_only_increment=op_only_i, fdt_lanczos_increment=fdt_i)
        if dt > DTS[0]:
            a, b = coupling_funcs(dt / 2, m)
            c_err, c_inc = 0.0, 0.0
            for x1, x2 in zip(xv, xv[1:] + xv[:1]):
                c_l = lanczos_apply(op, x1, rc, a)[0] + lanczos_apply(op, x2, rc, b)[0]
                c_d = Dh.apply(a, x1) + Dh.apply(b, x2)
                c_err = max(c_err, float(np.linalg.norm(c_l - c_d) / np.linalg.norm(c_d)))
                c_inc = max(c_inc, float(np.linalg.norm(c_l - c_d) / np.linalg.norm(c_d - (x1 + x2) / np.sqrt(2))))
            out[str(dt)]["coupling_input_krylov"] = c_err
            out[str(dt)]["coupling_input_krylov_increment"] = c_inc
    return out


def load_states(patterns, L):
    out = []
    for pat in patterns:
        for f in sorted(glob.glob(str(C.OUT / pat)) + glob.glob(pat)):
            with np.load(f) as d:
                if abs(float(d["L"]) - L) > 1e-9 * L or abs(float(d["kT"]) - 1.0) > 1e-12:
                    raise ValueError(f"{f}: state at L {float(d['L'])}, kT {float(d['kT'])}, not the lj075 state point")
                out.append((Path(f).stem, d["q"].copy(), d["p"].copy() if "p" in d.files else None, str(f),
                            d["xi"].copy() if "xi" in d.files else None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--laws", nargs="+", default=["A", "B"])
    ap.add_argument("--states", nargs="+", default=["raw/states/*.npz"])
    ap.add_argument("--lattice-state", action="store_true", help="also a jittered fcc lattice (not canonical)")
    ap.add_argument("--tag", default="representative")
    ap.add_argument("--pppm-A", type=float, nargs=4, metavar=("XI", "S", "ETA", "P"))
    ap.add_argument("--pppm-B", type=float, nargs=4, metavar=("XI", "S", "ETA", "P"))
    ap.add_argument("--ranks-A", type=int, nargs=3, metavar=("NOISE", "DAMP", "COUPLE"), default=None)
    ap.add_argument("--ranks-B", type=int, nargs=3, metavar=("NOISE", "DAMP", "COUPLE"), default=None)
    args = ap.parse_args()
    for law in args.laws:
        cpu0, wall0 = C.cpu_seconds(), time.perf_counter()
        r = tc.resolve(CONFIG, law, 256, "lj", dt=0.005, allow_unverified=True)
        mdl = r["model"]
        pp = dict(r["pppm"])
        ov = getattr(args, f"pppm_{law}")
        if ov:
            pp = dict(xi=ov[0], s=ov[1], eta=ov[2], p=int(ov[3]))
        L = r["L"]
        states = load_states(args.states, L)
        if args.lattice_state:
            states.append(("fcc_jitter0.05", C.fcc_positions(256, L, 0.05, np.random.default_rng(5)), None, "constructed",
                           None))
        dig = lambda a: None if a is None else tc._digest(np.asarray(a).round(12).tolist())   # noqa: E731
        key = tc._digest(dict(protocol=PROTOCOL, pppm=pp, model=mdl, L=L, law=law, budget=r["operator_budget"],
                              states=[(s[0], dig(s[1]), dig(s[2]), dig(s[4])) for s in states],
                              ranks=getattr(args, f"ranks_{law}")))
        outf = C.OUT / f"verify_{args.tag}_{law}{'_' + '_'.join(map(str, ov)) if ov else ''}.json"
        if outf.exists() and json.loads(outf.read_text()).get("cache_key") == key:
            print(f"[{law}] cached {outf.name}")
            continue
        kern = rk.toy_kernel(law, mdl["gamma"], mdl["kappa"], mdl["r_ref"], pp["xi"])
        fast = rk.FastFriction(L, kern, pp["s"], pp["eta"], pp["p"], pair_search="tree")
        lat = v2.LatticeFriction(L, kernel=law, gamma=mdl["gamma"], kappa=mdl["kappa"], r_ref=mdl["r_ref"])
        rng = np.random.default_rng([2026, 10, 9, ord(law)])
        res = dict(config=CONFIG, law=law, pppm=pp, mesh=fast.record(), lattice=lat.record(), L=L, model=mdl,
                   budget=r["operator_budget"], protocol=PROTOCOL, cache_key=key, provenance=C.provenance(),
                   reference_check=reference_selfcheck(lat, L, law, mdl, rng))
        x0 = states[0][1] % L
        i1, j1, d1 = real_pairs(x0, L, fast.rc)
        i2, j2, d2 = rk.real_pairs_tree(x0, L, fast.rc)
        k1 = np.round(np.c_[np.minimum(i1, j1), np.maximum(i1, j1), d1 * np.sign(j1 - i1)[:, None]], 9)
        k2 = np.round(np.c_[np.minimum(i2, j2), np.maximum(i2, j2), d2 * np.sign(j2 - i2)[:, None]], 9)
        res["pair_set_tree_equals_images"] = bool(len(k1) == len(k2) and np.array_equal(
            k1[np.lexsort(k1.T[::-1])], k2[np.lexsort(k2.T[::-1])]))
        per, kept = {}, {}
        for name, q, p_act, src, xi_act in states:
            rec, op, Dh, Dr, probes = verify_state(r, fast, lat, q, p_act, rng, xi_act)
            rec["source"] = src
            per[name] = rec
            kept[name] = (op, Dh, Dr, probes)
            print(f"[{law} {name}] op {rec['op_err']:.2e} (gv {rec['op_err_gv_probes']:.1e}, low eigs "
                  f"{rec['op_err_low12_eigs']:.1e}) lam [{rec['lam_min']:.3f}, {rec['lam_max']:.2f}] sym "
                  f"{rec['sym_resid']:.0e} null {rec['null_resid']:.0e} [{rec['dense_s'] + rec['lanczos_s']:.0f} s]",
                  flush=True)
        res["per_state"] = per
        kry = {n: v["krylov"] for n, v in per.items()}
        res["required_rank_strict"] = required(kry, r["operator_budget"], 0.1)
        res["required_rank_budget"] = required(kry, r["operator_budget"], 1.0)
        req = res["required_rank_strict"]
        if any(v is None for v in req.values()):
            res["passed"] = False
            res["failure"] = f"no rank <= {RANKS[-1]} meets the strict rule for {[k for k, v in req.items() if v is None]}"
            C.write_json(outf, res)
            print(f"[{law}] FAIL: {res['failure']}", flush=True)
            continue
        auto = (max(v for k, v in req.items() if k.startswith("noise")),
                max(v for k, v in req.items() if k.startswith("damp")),
                max(v for k, v in req.items() if k.startswith("couple")))
        chosen = getattr(args, f"ranks_{law}") or auto
        if any(c not in RANKS for c in chosen):
            raise SystemExit(f"explicit ranks must be in the scanned set {RANKS}")
        res["required_rank_strict_increment"] = required({n: v["krylov_increment"] for n, v in per.items()},
                                                         r["operator_budget"], 0.1)
        res["chosen_ranks"] = dict(noise=chosen[0], damp=chosen[1], couple=chosen[2],
                                   rule="explicit" if getattr(args, f"ranks_{law}") else "max strict requirement over dt")
        res["chosen_checks"] = {n: chosen_checks(r, *kept[n], *chosen) for n in per}
        b = r["operator_budget"]
        cc = res["chosen_checks"]
        res["checks"] = dict(
            operator=max(v["op_err"] for v in per.values()) <= b,
            reference_chebyshev=res["reference_check"]["max_abs_err_over_max"] <= 1e-9,
            pair_set=res["pair_set_tree_equals_images"],
            symmetric=max(v["sym_resid"] for v in per.values()) <= 1e-13,
            null_space=max(max(v["null_resid"], v["null_resid_ref"]) for v in per.values()) <= 1e-12,
            psd_on_range=min(min(v["lam_min"], v["lam_min_ref"]) for v in per.values()) > 0,
            literal_matvec=max(v["literal_vs_dense"] for v in per.values()) <= 1e-12,
            k0_retained=bool(res["mesh"]["k0_retained"]),
            ritz_positive=min(v["ritz_min"] for v in per.values()) > 0,
            ostep_krylov=max(c[d]["ostep_krylov"] for c in cc.values() for d in c) <= b / 10,
            ostep_vs_reference=max(c[d]["ostep_vs_reference"] for c in cc.values() for d in c) <= b,
            fdt_lanczos=max(c[d]["fdt_lanczos"] for c in cc.values() for d in c) <= b / 10,
            coupling=max(c[d].get("coupling_input_krylov", 0) for c in cc.values() for d in c) <= b / 10,
            chosen_ranks_meet_strict_rule=all(
                _mx(per[n]["krylov"][fn][str(chosen[0] if fn.startswith("noise") else chosen[1] if fn.startswith("damp")
                                              else chosen[2])]) <= b / 10 for n in per for fn in per[n]["krylov"]))
        # reported, not part of 'passed' (the budgets are defined relative to the full norm, as in the project):
        res["operator_vs_k_nonzero_part_max"] = max(v["op_err_vs_k_nonzero_part"] for v in per.values())
        res["operator_vs_k_nonzero_within_budget"] = res["operator_vs_k_nonzero_part_max"] <= b
        res["increment_normalised"] = dict(
            ostep_pppm_only_increment_max=max(c[d]["ostep_pppm_only_increment"] for c in cc.values() for d in c),
            ostep_krylov_increment_max=max(c[d]["ostep_krylov_increment"] for c in cc.values() for d in c),
            ostep_vs_reference_increment_max=max(c[d]["ostep_vs_reference_increment"] for c in cc.values() for d in c),
            fdt_lanczos_increment_max=max(c[d]["fdt_lanczos_increment"] for c in cc.values() for d in c),
            coupling_increment_max=max(c[d].get("coupling_input_krylov_increment", 0) for c in cc.values() for d in c),
            chosen_ranks_meet_strict_rule_increment=all(
                per[n]["krylov_increment"][fn][str(chosen[0] if fn.startswith("noise") else chosen[1]
                                                  if fn.startswith("damp") else chosen[2])] <= b / 10
                for n in per for fn in per[n]["krylov_increment"]))
        res["passed"] = all(res["checks"].values())
        res["cpu_s"], res["wall_s"] = C.cpu_seconds() - cpu0, time.perf_counter() - wall0
        C.write_json(outf, res)
        print(f"[{law}] strict required ranks {req}; chosen {chosen}; checks "
              f"{ {k: v for k, v in res['checks'].items() if not v} or 'all pass'}; cpu {res['cpu_s']:.0f} s", flush=True)



def runner_entry(law, dt, tag="representative"):
    """Write the toy_run.verification_status entry for the configuration at (law, dt) from a verify result whose
    PPPM set and ranks equal the configuration's (refuses otherwise)."""
    r = tc.resolve(CONFIG, law, 256, "lj", dt=dt, allow_unverified=True)
    res = json.loads((C.OUT / f"verify_{tag}_{law}.json").read_text())
    if res["pppm"] != r["pppm"]:
        raise ValueError(f"verify result PPPM {res['pppm']} != configuration {r['pppm']}")
    if (abs(res["L"] - r["L"]) > 1e-12 * r["L"] or res["model"] != r["model"] or res["budget"] != r["operator_budget"]
            or res["protocol"]["version"] != PROTOCOL["version"] or res["mesh"]["pair_search"] != r["pair_search"]
            or not res["passed"]):
        raise ValueError("verify result does not bind to the current configuration (L, model, budget, protocol, "
                         "pair search) or did not pass")
    ch = res["chosen_ranks"]
    if (r["rank_noise"], r["rank_damp"]) != (ch["noise"], ch["damp"]):
        raise ValueError(f"configuration ranks {r['rank_noise']}/{r['rank_damp']} != verified {ch}")
    if not any(abs(dt - d) < 1e-15 for d in DTS):
        raise ValueError(f"dt {dt} not covered by the verification")
    per = res["per_state"]
    entry = dict(passed=res["passed"], operator_hash=tc.operator_hash(r), budget=res["budget"],
                 op_err=max(v["op_err"] for v in per.values()), checks=res["checks"],
                 lam_min=min(v["lam_min"] for v in per.values()), ritz_min=min(v["ritz_min"] for v in per.values()),
                 per_state={k: dict(op_err=v["op_err"], lam_min=v["lam_min"]) for k, v in per.items()},
                 source=f"lj075_results/verify_{tag}_{law}.json", dt=dt, ranks=ch)
    f = HERE / tc.CONFIGS[CONFIG]["verification_file"]
    db = json.loads(f.read_text()) if f.exists() else {}
    db[f"{CONFIG}_{law}_N256_dt{dt:g}"] = entry          # one entry per dt (operator_hash includes dt)
    C.write_json(f, db)
    return entry


if __name__ == "__main__":
    main()
