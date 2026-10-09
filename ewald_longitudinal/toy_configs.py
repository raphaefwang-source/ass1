#!/usr/bin/env python3
"""
Named, explicit configurations for the toy friction dynamics (production runner `toy_run.py`).

The model is fixed:
- pure longitudinal K = g(r) r̂r̂ᵀ;
- full periodic images, k = 0 retained;
- g(r_ref) = gamma = 0.5, kappa = 0.7, r_ref = 1.3;
- kT = 0.7, m = 1;
- density 64/5.5^3;
- dt = 0.005 (validated value).

State points (STATE_POINTS; each configuration names one, the default being the original toy state):
  toy_v2              kT 0.7, density 64/5.5^3 (MODEL above); all configurations before 2026-10-09.
  lj_rho0.75_kT1.0    LJ liquid, kT 1.0, density 0.75 (same gamma, kappa, r_ref, potential parameters).
Nothing validated at one state point is assumed at another: each configuration carries its own validated dt
(dt_validated, None = not validated), allowed potentials, minimum N and verified ranks.

Configurations:
  costopt_hiacc   cost-optimised high-accuracy sets of cost_optimization_results/REPORT.md
                  (law A: operator budget 2e-7; law B: 5e-8). KD-tree real-space pairs, neighbour-list forces.
                  Lanczos ranks: smallest ranks of the scan (production_rank_scan.json) whose overall,
                  k = 2 pi/L and lowest-eigenmode Krylov errors are <= budget/10 on every test state at that N
                  (v2 / replicated / lattice state and a liquid-like state, which has a much wider spectrum); the
                  factor 10 is a margin for states more clustered than the test state. Verified at the budget with
                  independent vectors at N = 64, 256, 512 (production_config_verification.json).
                  These ranks are higher than the cost-optimisation report's (A 32/4, B 12/5), which were chosen on
                  replicated states only; B 12/5 fails on the liquid-like state at N = 256 and 512.
                  N > 512 is refused: A 32/4 fails at N = 1728 on the liquid-like state and no larger-N rows have
                  been verified on such states.
  lj_rho0.75_kT1.0_costopt
                  LJ at rho 0.75, kT 1.0, N = 256 only. PPPM sets and ranks re-verified at this state point
                  (lj075_verify.py, lj075_results/); dt validated by lj075_dt_study.py. Until those results are in,
                  dt_validated is None and every run needs an explicit --dt with --allow-unverified.
  baseline_hiacc  the original high-accuracy version (xi 0.7, s 4.10, eta 0.7, p 7, ranks 40/16, image-enumerated
                  pairs, all-pair forces): reproduces runs made before 2026-10-08.

Nothing falls back to a default silently. resolve() returns every parameter that the runner uses, with its
provenance, and the runner prints and stores it.
"""
import hashlib
import json

MODEL = dict(gamma=0.5, kappa=0.7, r_ref=1.3, kT=0.7, mass=1.0, density=64 / 5.5 ** 3,
             epsilon=1.0, sigma=1.0, r_on=2.0, r_cut=2.5, stop_rmin=0.45)
DT_VALIDATED = 0.005                    # validated at the toy_v2 state point only
STATE_POINTS = {
    "toy_v2": dict(MODEL),
    "lj_rho0.75_kT1.0": dict(MODEL, kT=1.0, density=0.75),
}
POTENTIALS = ("double_well", "lj")
KERNELS = ("A", "B")

CONFIGS = {
    "costopt_hiacc": dict(
        description="cost-optimised high-accuracy PPPM + Lanczos (REPORT.md budgets: A 2e-7, B 5e-8)",
        pair_search="tree",
        force_method="neighbor",
        operator_budget={"A": 2e-7, "B": 5e-8},
        # eta is the realised mesh parameter eta* = h kc / pi at N = 512 (transferable: any box gets a mesh at least
        # as fine). Values from cost_optimization_results/candidates.json, tags A_2e-7 and B_5e-8.
        pppm={"A": dict(xi=0.85, s=4.1, eta=0.6779116381586452, p=7),
              "B": dict(xi=1.0, s=3.9, eta=0.6827747058642308, p=7)},
        # (N_max, rank_noise, rank_damp): the first row with N <= N_max is used. Requirements grow with N and with
        # clustering (lambda_min falls); see production_config_verification.json. No row beyond N = 512.
        ranks={"A": ((256, 32, 5), (512, 40, 5)), "B": ((256, 16, 6), (512, 24, 8))},
        verified_N=(64, 256, 512),
    ),
    "lj_rho0.75_kT1.0_costopt": dict(
        description="LJ rho 0.75, kT 1.0: cost-optimised PPPM sets of costopt_hiacc (budgets A 2e-7, B 5e-8), "
                    "operator and ranks re-verified at this state point (lj075_results/)",
        state_point="lj_rho0.75_kT1.0",
        verification_file="lj075_results/verification_runner.json",
        potentials=("lj",),
        N_allowed=(256,),
        dt_validated=None,                    # set only after lj075_dt_study.py
        pair_search="tree",
        force_method="neighbor",
        operator_budget={"A": 2e-7, "B": 5e-8},
        pppm={"A": dict(xi=0.85, s=4.1, eta=0.6779116381586452, p=7),
              "B": dict(xi=1.0, s=3.9, eta=0.6827747058642308, p=7)},
        # provisional: the costopt_hiacc N <= 256 row, NOT verified at this state point (verified_N empty)
        ranks={"A": ((256, 32, 5),), "B": ((256, 16, 6),)},
        verified_N=(),
    ),
    "baseline_hiacc": dict(
        description="original high-accuracy PPPM + Lanczos 40/16, image-enumerated pairs, all-pair forces "
                    "(reproduces runs before 2026-10-08)",
        pair_search="images",
        force_method="allpairs",
        operator_budget={"A": 2e-7, "B": 5e-8},     # checked against the same budgets as costopt_hiacc (informational)
        pppm={"A": dict(xi=0.7, s=4.10, eta=0.7, p=7), "B": dict(xi=0.7, s=4.10, eta=0.7, p=7)},
        ranks={"A": ((None, 40, 16),), "B": ((None, 40, 16),)},
        verified_N=None,                                       # the fixed ranks 40/16 are not N-dependent
    ),
}


class ConfigError(ValueError):
    pass


def box_length(N, density=None):
    """Cubic box edge at the given density (default: the toy_v2 density, for the configurations before 2026-10-09)."""
    return (N / (MODEL["density"] if density is None else density)) ** (1 / 3)


def model_of(name):
    """The state point (model dict) of a configuration."""
    return dict(STATE_POINTS[CONFIGS[name].get("state_point", "toy_v2")])


def validated_dt(name):
    """The dt validated for this configuration's state point, or None if none has been validated yet."""
    return CONFIGS[name].get("dt_validated", DT_VALIDATED)


def select_ranks(name, law, N, override=None):
    """(rank_noise, rank_damp, provenance). Refuses N beyond the verified table unless ranks are overridden."""
    cfg = CONFIGS[name]
    if override is not None:
        return int(override[0]), int(override[1]), dict(rule="user override", verified_at_this_N=False)
    for n_max, rn, rd in cfg["ranks"][law]:
        if n_max is None or N <= n_max:
            ver = cfg["verified_N"]
            if ver is not None and len(ver) == 0:
                return rn, rd, dict(rule="provisional ranks, NOT verified at this state point", table_row_N_max=n_max,
                                    verified_at_this_N=False, verified_N=[],
                                    note="provisional ranks: verify at this state point before production")
            prov = dict(rule=("fixed ranks of the original version" if n_max is None else
                              f"strict Krylov rule, table row N <= {n_max}"),
                        table_row_N_max=n_max,
                        verified_at_this_N=(ver is None or N in ver),
                        verified_N=list(ver) if ver else None)
            if ver is not None and N not in ver:
                prov["note"] = ("ranks taken from the next verified size up; assumes the rank requirement grows "
                                "monotonically with N (observed for 64 -> 4096)")
            return rn, rd, prov
    raise ConfigError(f"{name}: no verified Lanczos ranks for N = {N} (law {law}); largest verified N is "
                      f"{cfg['ranks'][law][-1][0]}. Verify first or pass explicit ranks (unverified).")


def resolve(name, law, N, potential, dt=None, rank_override=None, allow_unverified=False):
    """Every parameter the runner uses, with provenance. Raises ConfigError instead of guessing.
    dt=None means the configuration's validated dt (refused if it has none)."""
    if name not in CONFIGS:
        raise ConfigError(f"unknown config {name!r}; choose from {sorted(CONFIGS)}")
    cfg = CONFIGS[name]
    pots = cfg.get("potentials", POTENTIALS)
    if law not in KERNELS or potential not in pots:
        raise ConfigError(f"kernel must be in {KERNELS}, potential in {pots} for {name}")
    model = model_of(name)
    L = box_length(N, model["density"])
    if N < 64:
        raise ConfigError("N >= 64 required (neighbour list needs r_cut < L/2; toy density)")
    if not model["r_cut"] * model["sigma"] < L / 2:
        raise ConfigError(f"N = {N} gives L = {L:.4f}: the neighbour list needs r_cut < L/2")
    if "N_allowed" in cfg and N not in cfg["N_allowed"]:
        raise ConfigError(f"{name} is defined for N in {cfg['N_allowed']} only (got {N})")
    dt_ok = validated_dt(name)
    if dt is None:
        if dt_ok is None:
            raise ConfigError(f"{name} has no validated dt yet: pass dt explicitly with allow_unverified")
        dt = dt_ok
    warnings = []
    rn, rd, prov = select_ranks(name, law, N, rank_override)
    if rank_override is not None:
        if not allow_unverified:
            raise ConfigError("explicit ranks are unverified; add --allow-unverified to use them")
        warnings.append("Lanczos ranks overridden by the user (unverified)")
    if dt_ok is None or abs(dt - dt_ok) > 1e-15:
        if not allow_unverified:
            raise ConfigError(f"ranks and operator were validated at dt = {dt_ok}; dt = {dt} needs "
                              "--allow-unverified")
        warnings.append(f"dt = {dt} differs from the validated dt = {dt_ok}" if dt_ok is not None else
                        f"dt = {dt}: no dt has been validated for {name} yet")
    if not prov["verified_at_this_N"] and rank_override is None:
        warnings.append(prov.get("note", "ranks not verified at this N"))
    return dict(config=name, description=cfg["description"], law=law, potential=potential, N=int(N),
                L=L, dt=float(dt), model=model, pppm=dict(cfg["pppm"][law]),
                pair_search=cfg["pair_search"], force_method=cfg["force_method"],
                rank_noise=int(rn), rank_damp=int(rd), rank_provenance=prov,
                operator_budget=cfg["operator_budget"][law], warnings=warnings)


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()[:16]


def operator_hash(r):
    """Key of everything a static operator / Lanczos verification depends on (not the potential or run length)."""
    return _digest({k: r[k] for k in ("config", "law", "N", "dt", "model", "pppm", "pair_search",
                                      "rank_noise", "rank_damp")})


def physics_hash(r):
    """Key of everything a trajectory depends on, except run length and output settings."""
    return _digest({k: r[k] for k in ("config", "law", "potential", "N", "dt", "model", "pppm", "pair_search",
                                      "force_method", "rank_noise", "rank_damp")})


if __name__ == "__main__":
    for name in CONFIGS:
        for law in KERNELS:
            for N in CONFIGS[name].get("N_allowed", (64, 256, 512)):
                pot = CONFIGS[name].get("potentials", POTENTIALS)[0]
                r = resolve(name, law, N, pot, dt=validated_dt(name) or 0.005, allow_unverified=True)
                print(f"{name:15s} {law} N={N:5d} {r['model']['kT']} {r['model']['density']:.4f} L={r['L']:.5f}: pppm {r['pppm']} ranks {r['rank_noise']}/{r['rank_damp']} "
                      f"pairs {r['pair_search']} forces {r['force_method']} op-hash {operator_hash(r)}")
