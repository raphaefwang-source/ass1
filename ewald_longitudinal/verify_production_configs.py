#!/usr/bin/env python3
"""
Static confirmation of the named production configurations (toy_configs.py) at exactly the parameters the runner
uses: PPPM set, pair search and the N-dependent Lanczos ranks.

Test states (max over states is reported):
- N = 64: the v2 double-well and LJ states (seed 101);
- N > 64: the state of toy_cost_opt.config (replicated v2 state, or a jittered lattice at N = 256) and a liquid-like
  double-well state whose positions come from scalar Langevin dynamics with the conservative forces only (t = 50 from
  fcc / sc; positions are canonical whatever the friction). Not a claim of full equilibration; a lattice alone
  under-represents the soft long-wavelength modes (lambda_min 1.78 at N = 256 vs 1.73 at N = 64).

Per config x law x N x state:
- operator:
  - N <= 512: spectral error ||Gamma_h - Gamma_ref||_2 / ||Gamma_ref||_2 against the dense v2 full-periodic
    reference (N = 64: double-well and LJ states);
  - N > 512: Gamma v error on 48 sampled particles against the direct full-periodic sum;
- structure (N <= 512): symmetry, momentum null space, lambda_min on Range(Pi), k = 0 retained, literal matvec vs
  dense assembly;
- Lanczos at the configured ranks, through the production routine test_true_dynamics.lanczos_apply:
  - noise sqrt-action on z ~ N(0, I) and damping action on Maxwell momenta;
  - errors overall, in the k = 2 pi / L plane-wave components and (N <= 512) in the lowest 3 / 12 eigenmodes;
  - reference: dense f(Gamma_h) for N <= 512, a converged rank-96 / rank-32 Lanczos result otherwise;
  - the combined O-step error.

Cache: each entry stores the operator hash (toy_configs.operator_hash: config, law, N, dt, model, PPPM parameters,
pair search, ranks) together with this protocol's version and vector counts. An entry is reused only when that
whole key matches; any parameter change recomputes it.

    python3 verify_production_configs.py [--configs costopt_hiacc baseline_hiacc] [--N 64 256 512 1728 4096]
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import radial_kernels as rk  # noqa: E402
import toy_configs as tc  # noqa: E402
import toy_cost_opt as tco  # noqa: E402
from test_lanczos_fdt import DenseRef, lanczos, tridiag, proj  # noqa: E402
from test_true_dynamics import lanczos_apply  # noqa: E402

OUT = HERE / "cost_optimization_results" / "production_config_verification.json"
STATES = HERE / "cost_optimization_results" / "verification_states"
LIQ = dict(t=50.0, dt=0.005, gamma=1.0, seed=7, start="fcc if N = 4 n^3 else sc, jitter 0.05")
PROTOCOL = dict(version=2, vectors_dense=4, vectors_large=3, rows_large=48, ref_ranks=(96, 32), conv_ranks=(80, 24),
                states=["toy_cost_opt.config(N, law, pot, seed=101, rng_seed=0)",
                        f"N > 64: scalar-Langevin double-well liquid {LIQ}"])


def liquid_state(N):
    """Positions from scalar Langevin dynamics (BAOAB, exact OU step, friction LIQ['gamma'], kT 0.7) with the
    neighbour-list double-well forces only; cached by its generation parameters."""
    STATES.mkdir(parents=True, exist_ok=True)
    f = STATES / f"liquid_double_well_N{N}_{tc._digest(dict(LIQ, N=N, model=tc.MODEL))}.npz"
    if f.exists():
        with np.load(f) as d:
            return d["q"].copy()
    import toy_dynamics as td
    L, kT, m = tc.box_length(N), tc.MODEL["kT"], tc.MODEL["mass"]
    rng = np.random.default_rng([LIQ["seed"], N])
    n = round((N / 4) ** (1 / 3))
    if 4 * n ** 3 == N:
        g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
        basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
        q = ((g[:, None, :] + basis[None]).reshape(-1, 3) + 0.25) * (L / n)
    else:
        n = round(N ** (1 / 3))
        g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
        q = (g + 0.5) * (L / n)
    q = q + rng.uniform(-0.05, 0.05, q.shape)
    p = rng.normal(0, np.sqrt(m * kT), q.shape)
    dt, c1 = LIQ["dt"], np.exp(-LIQ["gamma"] * LIQ["dt"] / m)
    c2 = np.sqrt(m * kT * (1 - c1 ** 2))
    F, _, rmin = td.conservative_force_neighbor(q, L, "double_well")
    for _ in range(int(round(LIQ["t"] / dt))):
        p = p + 0.5 * dt * F
        q = q + 0.5 * dt / m * p
        p = c1 * p + c2 * rng.standard_normal(q.shape)
        q = q + 0.5 * dt / m * p
        F, _, rmin = td.conservative_force_neighbor(q, L, "double_well")
        p = p + 0.5 * dt * F
        if rmin < 0.45:
            raise RuntimeError("scalar Langevin preparation: close encounter")
    q = q % L
    np.savez(f, q=q, generation=np.array(json.dumps(dict(LIQ, N=N))))
    return q


def f_noise(r):
    kT, m, dt = r["model"]["kT"], r["model"]["mass"], r["dt"]
    return lambda l: np.sqrt(kT * m * -np.expm1(-2 * dt * l / m))


def f_damp(r):
    m, dt = r["model"]["mass"], r["dt"]
    return lambda l: np.exp(-dt * l / m)


def lanczos_ref(op, v, smax, sconv, fun):
    Lz = lanczos(op, v, smax)
    out = {}
    for s in (sconv, smax):
        ss = min(s, Lz["s"])
        th, U = np.linalg.eigh(tridiag(Lz["al"], Lz["be"], ss))
        out[s] = proj(Lz["nz"] * (Lz["Q"][:ss].T @ (U @ (fun(th) * U[0]))))
    return out[smax], float(np.linalg.norm(out[sconv] - out[smax]) / np.linalg.norm(out[smax]))


def verify(r):
    """Max over the test states of verify_state(); per-state records kept."""
    N, law = r["N"], r["law"]
    states = {"v2_dw_lj" if N == 64 else ("replicated_v2" if round((N / 64) ** (1 / 3)) ** 3 * 64 == N
                                         else "jittered_lattice"): tco.config(N, law)[0]}
    if N > 64:
        states["liquid_scalar_langevin"] = liquid_state(N)
    per = {name: verify_state(r, q, with_lj=(N == 64)) for name, q in states.items()}
    first = next(iter(per.values()))
    out = dict(per_state=per, mesh=first["mesh"], budget=first["budget"])
    for k in ("op_err", "gv_rows_err", "sym_resid", "null_resid", "literal_vs_dense"):
        if k in first:
            out[k] = max(v[k] for v in per.values())
    for k in ("lam_min", "ritz_min"):
        if k in first:
            out[k] = min(v[k] for v in per.values())
    out["ritz_max"] = max(v["ritz_max"] for v in per.values())
    out["krylov"] = {kind: {c: (None if first["krylov"][kind][c] is None else max(v["krylov"][kind][c] for v in per.values()))
                            for c in first["krylov"][kind]} for kind in first["krylov"]}
    out["ostep"] = dict(krylov=max(v["ostep"]["krylov"] for v in per.values()))
    out["checks"] = {c: all(v["checks"][c] for v in per.values()) for c in first["checks"]}
    out["passed"] = all(v["passed"] for v in per.values())
    return out


def verify_state(r, q, with_lj):
    N, law, L = r["N"], r["law"], r["L"]
    pp = r["pppm"]
    kern = rk.toy_kernel(law, r["model"]["gamma"], r["model"]["kappa"], r["model"]["r_ref"], pp["xi"])
    fast = rk.FastFriction(L, kern, pp["s"], pp["eta"], pp["p"], pair_search=r["pair_search"])
    op = fast.gamma_h(q)
    rec = dict(mesh=fast.record(), n_real_pairs=len(op.w))
    x = q % L
    ph = np.exp(1j * 2 * np.pi / L * x)
    rng = np.random.default_rng([13, N])
    dense = N <= 512
    if dense:
        Gh = op.dense()
        Dh = DenseRef(Gh, N)
        errs = {}
        for pot in (("double_well", "lj") if with_lj else ("double_well",)):
            qq = q if pot == "double_well" else tco.config(N, law, pot)[0]
            Gr = tco.lattice_matrix(tco.lattice(L, law), qq)
            Gq = Gh if pot == "double_well" else fast.gamma_h(qq).dense()
            errs[pot] = float(np.linalg.norm(Gq - Gr, 2) / np.linalg.norm(Gr, 2))
            if pot == "double_well":
                Dr = DenseRef(Gr, N)
        T = np.kron(np.ones((N, 1)), np.eye(3))
        X = proj(rng.standard_normal(3 * N))
        rec.update(op_err=max(errs.values()), op_err_by_state=errs,
                   sym_resid=float(np.linalg.norm(Gh - Gh.T) / np.linalg.norm(Gh)),
                   null_resid=float(np.abs(Gh @ T).max() / np.abs(Gh).max()),
                   literal_vs_dense=float(np.linalg.norm(op.matvec(X) - Gh @ X) / np.linalg.norm(Gh @ X)),
                   lam_min=float(Dh.lam[0]), lam_max=float(Dh.lam[-1]), lam_min_ref=float(Dr.lam[0]))
    else:
        lat = tco.lattice(L, law)
        rows = np.sort(rng.choice(N, PROTOCOL["rows_large"], replace=False))
        rec.update(gv_rows_err=max(tco.gv_rows_error(op, lat, q, rng.standard_normal((N, 3)), rows) for _ in range(2)))
    nvec = PROTOCOL["vectors_dense"] if dense else PROTOCOL["vectors_large"]
    kr = {}
    ostep = dict(krylov=0.0)
    ritz = [np.inf, 0.0]
    vecs = {}
    for kind, fun, rank, (smax, sconv) in (("noise", f_noise(r), r["rank_noise"], (96, 80)),
                                            ("damp", f_damp(r), r["rank_damp"], (32, 24))):
        e = dict(rank=rank, overall=0.0, kmin=0.0, low3=None, low12=None, conv=None)
        if dense:
            e.update(low3=0.0, low12=0.0, total_vs_ref=0.0)
        vecs[kind] = []
        for _ in range(nvec):
            v = rng.standard_normal(3 * N)
            if kind == "damp":
                v *= np.sqrt(r["model"]["kT"] * r["model"]["mass"])
            v = proj(v)
            approx, (lo, hi) = lanczos_apply(op, v, rank, fun)          # production routine, production rank
            ritz = [min(ritz[0], lo), max(ritz[1], hi)]
            if dense:
                ref = Dh.apply(fun, v)
                e["total_vs_ref"] = max(e["total_vs_ref"], float(np.linalg.norm(approx - Dr.apply(fun, v))
                                                                 / np.linalg.norm(Dr.apply(fun, v))))
            else:
                ref, conv = lanczos_ref(op, v, smax, sconv, fun)
                e["conv"] = max(e["conv"] or 0.0, conv)
            d = approx - ref
            e["overall"] = max(e["overall"], float(np.linalg.norm(d) / np.linalg.norm(ref)))
            Jd, Jr = ph.T @ d.reshape(N, 3), ph.T @ ref.reshape(N, 3)
            e["kmin"] = max(e["kmin"], float(np.max(np.linalg.norm(Jd, axis=1) / np.linalg.norm(Jr, axis=1))))
            if dense:
                for n_low, name in ((3, "low3"), (12, "low12")):
                    B = Dh.U[:, :n_low]
                    e[name] = max(e[name], float(np.linalg.norm(B.T @ d) / np.linalg.norm(B.T @ ref)))
            vecs[kind].append((approx, ref))
        kr[kind] = e
    for (an, rn), (ad, rd) in zip(vecs["noise"], vecs["damp"]):
        ostep["krylov"] = max(ostep["krylov"], float(np.linalg.norm(an + ad - rn - rd) / np.linalg.norm(rn + rd)))
    rec.update(krylov=kr, ostep=ostep, ritz_min=float(ritz[0]), ritz_max=float(ritz[1]))
    tol = r["operator_budget"]
    op_val = rec["op_err"] if dense else rec["gv_rows_err"]
    kry = [kr[k][c] for k in kr for c in ("overall", "kmin", "low3", "low12") if kr[k][c] is not None]
    checks = dict(operator=op_val <= tol, krylov=max(kry) <= tol, ostep=ostep["krylov"] <= tol,
                  ritz_positive=ritz[0] > 0, k0_retained=bool(rec["mesh"]["k0_retained"]))
    if dense:
        checks.update(symmetric=rec["sym_resid"] <= 1e-13, null_space=rec["null_resid"] <= 1e-12,
                      psd_on_range=rec["lam_min"] > 0, literal_matvec=rec["literal_vs_dense"] <= 1e-12)
    else:
        checks.update(reference_converged=max(kr[k]["conv"] for k in kr) <= 1e-12)
    rec.update(budget=tol, checks=checks, passed=all(checks.values()))
    return rec


RANKS_N = (8, 10, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80)
RANKS_D = (3, 4, 5, 6, 8, 10, 12, 16)
SCAN = HERE / "cost_optimization_results" / "production_rank_scan.json"


def rank_scan(name, law, N):
    """Errors per Lanczos rank (overall, k = 2 pi/L, lowest 3 / 12 eigenmodes; dense f(Gamma_h) reference) for the
    operator of `name` on every test state at N <= 512. Returns {state: {noise: {rank: errs}, damp: {...}}}."""
    r = tc.resolve(name, law, N, "double_well", rank_override=(1, 1), allow_unverified=True)
    pp = r["pppm"]
    kern = rk.toy_kernel(law, r["model"]["gamma"], r["model"]["kappa"], r["model"]["r_ref"], pp["xi"])
    fast = rk.FastFriction(r["L"], kern, pp["s"], pp["eta"], pp["p"], pair_search=r["pair_search"])
    states = {"v2_or_replicated_or_lattice": tco.config(N, law)[0]}
    if N > 64:
        states["liquid_scalar_langevin"] = liquid_state(N)
    out = {}
    for sname, q in states.items():
        op = fast.gamma_h(q)
        Dh = DenseRef(op.dense(), N)
        ph = np.exp(1j * 2 * np.pi / r["L"] * (q % r["L"]))
        rng = np.random.default_rng([17, N])
        res = dict(lam_min=float(Dh.lam[0]), lam_max=float(Dh.lam[-1]))
        for kind, fun, ranks in (("noise", f_noise(r), RANKS_N), ("damp", f_damp(r), RANKS_D)):
            tab = {k: dict(overall=0.0, kmin=0.0, low3=0.0, low12=0.0) for k in ranks}
            for _ in range(PROTOCOL["vectors_dense"]):
                v = rng.standard_normal(3 * N)
                if kind == "damp":
                    v *= np.sqrt(r["model"]["kT"] * r["model"]["mass"])
                v = proj(v)
                ref = Dh.apply(fun, v)
                Lz = lanczos(op, v, max(ranks))
                Jr = ph.T @ ref.reshape(N, 3)
                for k in ranks:
                    ss = min(k, Lz["s"])
                    th, U = np.linalg.eigh(tridiag(Lz["al"], Lz["be"], ss))
                    d = proj(Lz["nz"] * (Lz["Q"][:ss].T @ (U @ (fun(th) * U[0])))) - ref
                    e = tab[k]
                    e["overall"] = max(e["overall"], float(np.linalg.norm(d) / np.linalg.norm(ref)))
                    e["kmin"] = max(e["kmin"], float(np.max(np.linalg.norm(ph.T @ d.reshape(N, 3), axis=1)
                                                            / np.linalg.norm(Jr, axis=1))))
                    for n_low, nm in ((3, "low3"), (12, "low12")):
                        B = Dh.U[:, :n_low]
                        e[nm] = max(e[nm], float(np.linalg.norm(B.T @ d) / np.linalg.norm(B.T @ ref)))
            res[kind] = {str(k): v for k, v in tab.items()}
        out[sname] = res
    return out


def required_ranks(scan, budget):
    """Smallest noise / damping ranks whose four error measures are <= budget on every state."""
    req = {}
    for kind in ("noise", "damp"):
        ranks = sorted(int(k) for k in next(iter(scan.values()))[kind])
        req[kind] = next((k for k in ranks if all(max(st[kind][str(k)].values()) <= budget for st in scan.values())),
                         None)
    return req


def main_scan(args):
    db = json.loads(SCAN.read_text()) if SCAN.exists() else {}
    for name in args.configs:
        for law in args.laws:
            for N in (args.N or [64, 256, 512]):
                r = tc.resolve(name, law, N, "double_well", rank_override=(1, 1), allow_unverified=True)
                key = f"{name}_{law}_N{N}"
                ck = tc._digest(dict(op={k: r[k] for k in ("config", "law", "N", "dt", "model", "pppm", "pair_search")},
                                     protocol=PROTOCOL, ranks=(RANKS_N, RANKS_D)))
                if key not in db or db[key]["cache_key"] != ck:
                    t0 = time.process_time()
                    db[key] = dict(cache_key=ck, scan=rank_scan(name, law, N), cpu_s=None)
                    db[key]["cpu_s"] = time.process_time() - t0
                    SCAN.write_text(json.dumps(db, indent=1, default=float) + "\n")
                budget = tc.CONFIGS[name]["operator_budget"][law]
                req = required_ranks(db[key]["scan"], budget)
                db[key]["required_at_budget"] = dict(budget=budget, **req)
                SCAN.write_text(json.dumps(db, indent=1, default=float) + "\n")
                lam = {s: (round(v["lam_min"], 3), round(v["lam_max"], 1)) for s, v in db[key]["scan"].items()}
                print(f"[scan {key}] budget {budget:.0e}: required noise {req['noise']} damping {req['damp']}; "
                      f"spectra {lam}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["costopt_hiacc", "baseline_hiacc"])
    ap.add_argument("--laws", nargs="+", default=list(tc.KERNELS))
    ap.add_argument("--N", type=int, nargs="+", default=None,
                    help="default: 64 256 512 (the sizes the configurations allow; costopt_hiacc refuses N > 512)")
    ap.add_argument("--scan", action="store_true", help="rank scan on all test states instead of verification")
    args = ap.parse_args()
    if args.scan:
        return main_scan(args)
    db = json.loads(OUT.read_text()) if OUT.exists() else {}
    t_all = time.process_time()
    for name in args.configs:
        for law in args.laws:
            Ns = args.N or [64, 256, 512]
            for N in Ns:
                r = tc.resolve(name, law, N, "double_well")
                key = f"{name}_{law}_N{N}"
                cache_key = tc._digest(dict(op=tc.operator_hash(r), protocol=PROTOCOL, budget=r["operator_budget"]))
                if key in db and db[key].get("cache_key") == cache_key:
                    print(f"[{key}] cached ({cache_key}) passed={db[key]['passed']}")
                    continue
                if key in db:
                    print(f"[{key}] parameters or protocol changed: recomputing (old {db[key].get('cache_key')})")
                t0 = time.process_time()
                rec = verify(r)
                rec.update(config=name, law=law, N=N, operator_hash=tc.operator_hash(r), cache_key=cache_key,
                           protocol=PROTOCOL, resolved=r, cpu_s=time.process_time() - t0)
                db[key] = rec
                OUT.write_text(json.dumps(db, indent=1, default=float) + "\n")
                kr = rec["krylov"]
                opv = rec.get("op_err", rec.get("gv_rows_err"))
                print(f"[{key}] ranks {r['rank_noise']}/{r['rank_damp']} M {rec['mesh']['M']} op {opv:.2e} "
                      f"(budget {rec['budget']:.0e}) noise {kr['noise']['overall']:.1e}/{kr['noise']['kmin']:.1e}"
                      f"/{kr['noise']['low3'] if kr['noise']['low3'] is None else format(kr['noise']['low3'], '.1e')} "
                      f"damp {kr['damp']['overall']:.1e}/{kr['damp']['kmin']:.1e} ostep {rec['ostep']['krylov']:.1e} "
                      f"Ritz [{rec['ritz_min']:.3f}, {rec['ritz_max']:.1f}] states {list(rec['per_state'])} -> "
                f"{'PASS' if rec['passed'] else 'FAIL'} "
                      f"{[k for k, v in rec['checks'].items() if not v]} [{rec['cpu_s']:.0f} s]", flush=True)
    print(f"total cpu {time.process_time() - t_all:.0f} s")


if __name__ == "__main__":
    main()
