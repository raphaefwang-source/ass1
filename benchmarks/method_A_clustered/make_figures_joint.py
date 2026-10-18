#!/usr/bin/env python3
"""
Figures and tables of the joint error-budget optimisation (results_joint/ -> figures_joint/, results_joint/*.csv).

    python3 make_figures_joint.py
Colours: approaches in fixed categorical order (A blue, A_hz orange, B aqua, C yellow); grey = existing baseline.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common as C  # noqa: E402
from joint_optimizer import RESULTS_JOINT, FIGURES_JOINT, sigma_hz, APPROACHES, LABEL, TAU  # noqa: E402
from parameter_optimizer import C_p  # noqa: E402
from make_figures import style, INK2, GREY, MUTED  # noqa: E402
from scipy import special  # noqa: E402

AP_COL = {"A": "#2a78d6", "A_hz": "#eb6834", "B": "#1baf7a", "C": "#eda100"}
AP_MK = {"A": "o", "A_hz": "s", "B": "D", "C": "^"}
FAMS = ("uniform", "moderate", "strong", "lj_equilibrated")
FLAB = {"uniform": "uniform", "moderate": "moderate", "strong": "strong", "lj_equilibrated": "LJ eq."}


def save(fig, name):
    FIGURES_JOINT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_JOINT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def load():
    return dict(sel=C.read_json(RESULTS_JOINT / "joint_selection.json")["groups"],
                bench=C.read_json(RESULTS_JOINT / "bench.json")["rows"],
                err=C.read_json(RESULTS_JOINT / "errors.json"),
                alias=C.read_json(RESULTS_JOINT / "alias_bound_validation.json"))


def err_of(D, config, ap, role, xi):
    for r in D["err"]["rows"]:
        if r["config"] == config and r["approach"] == ap and r["role"] == role and abs(r["xi"] - xi) < 1e-9:
            return r
    return {}


def bench_of(D, config, ap, role=None):
    return [r for r in D["bench"] if r["config"] == config and r["approach"] == ap and (role is None or r["role"] == role)]


# ----------------------------------------------------------------------------- figure J1: alias bound
def fig_alias(D):
    p = 7
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    x = np.linspace(1e-3, 0.5 - 1e-6, 600)
    ax.plot(x, sigma_hz(x, p), color=AP_COL["C"], label="Hurwitz-zeta  |x|^p [zeta(p,1+|x|) + zeta(p,1-|x|)]")
    ax.plot(x, 2 * special.zeta(p, 1) * (x / (1 - x)) ** p, color=AP_COL["A"], ls="--", label="original  2 zeta(p) (x/(1-x))^p")
    u = np.linspace(0, 1, 1001)
    xs = np.arange(1, 24) / 48
    mp = np.arange(-300, 301)
    mp = mp[mp != 0]
    sup = []
    for xx in xs:
        r = (np.sinc(xx + mp) / np.sinc(xx)) ** p
        sup.append(np.abs((np.exp(-2j * np.pi * np.outer(u, mp)) * r).sum(1)).max())
    ax.plot(xs, sup, ls="none", marker="o", mfc="white", color=INK2, ms=5,
            label="sup over particle offset of |beta| (existing B-spline, M = 48)")
    ax.set_yscale("log")
    ax.set_xlabel("x = |k_a| h / (2 pi) = |m_a| / M")
    ax.set_ylabel("per-axis alias sum, p = 7")
    ax.set_title("(a) per-axis alias bound", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    rows = D["alias"]["checks"]["8_operator_level"]
    for cfg, mk in (("strong_N256_s1", "D"), ("uniform_N256_s1", "o"), ("strong_N64_s1", "v")):
        rr = sorted([r for r in rows if r["config"] == cfg], key=lambda r: r["M"])
        Ms = [r["M"] for r in rr]
        ax.plot(Ms, [r["bound_orig"] for r in rr], color=AP_COL["A"], ls="--", marker=mk, ms=4, lw=1)
        ax.plot(Ms, [r["bound_hz"] for r in rr], color=AP_COL["C"], marker=mk, ms=4, lw=1.2)
        ax.plot(Ms, [r["measured"] for r in rr], color=INK2, marker=mk, ms=4, lw=1, mfc="white", label=f"{cfg}")
    ax.plot([], [], color=AP_COL["A"], ls="--", label="original C(7) bound")
    ax.plot([], [], color=AP_COL["C"], label="Hurwitz-zeta bound")
    ax.plot([], [], color=INK2, mfc="white", marker="o", label="measured ||Gamma_L^mesh - Gamma_L^kc||_2 (markers: config)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("FFT mesh size M (fixed kc)")
    ax.set_ylabel("absolute spectral-norm mesh error")
    ax.set_title("(b) operator level: bounds vs measured", fontsize=9)
    ax.legend(fontsize=6.5)
    fig.suptitle("Figure J1. Mode-dependent Hurwitz-zeta alias bound, validated on the existing B-spline PPPM",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "figJ1_alias_bound")


# ----------------------------------------------------------------------------- figure J2: budget allocation
def fig_budget(D):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    ax = axes[0]
    w = 0.19
    comp_col = {"S": "#2a78d6", "F": "#eb6834", "mesh": "#1baf7a", "quad": "#eda100"}
    for k, fam in enumerate(FAMS):
        g = D["sel"].get(f"{fam}_N256")
        if not g:
            continue
        for a_, ap in enumerate(APPROACHES):
            o = g["approaches"][ap]["opt"]
            x0 = k + (a_ - 1.5) * (w + 0.02)
            bottom = 0.0
            tl = TAU * o["lam_lb_per_seed"][0]              # the constraint holds per configuration: seed 1's own budget
            for c, v in (("S", o["eps_S_abs_per_seed"][0] / tl), ("F", o["eps_F_abs"] / tl), ("mesh", o["eps_mesh_abs"] / tl)):
                ax.bar(x0, v, w, bottom=bottom, color=comp_col[c], edgecolor="#fcfcfb", linewidth=1,
                       label=(f"eps_{c}" if (k == 0 and a_ == 0) else None))
                bottom += v
            ax.text(x0, bottom + 0.02, ap.replace("_hz", "'"), ha="center", fontsize=7, color=INK2)
    ax.axhline(1.0, color=INK2, ls="--", lw=1)
    ax.set_xticks(range(len(FAMS)))
    ax.set_xticklabels([FLAB[f] for f in FAMS])
    ax.set_ylabel("certified bound / (tau Lambda_lb), seed 1")
    ax.set_ylim(0, 1.15)
    ax.set_title("(a) budget used by each certified term at the optimum (N = 256, seed 1); A' = A_hz", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax = axes[1]
    for ap in ("A", "B", "C"):
        for k, fam in enumerate(FAMS):
            cfg = f"{fam}_N256_s1" if fam != "lj_equilibrated" else "lj_equilibrated_N256_s305"
            r = next((e for e in D["err"]["rows"] if e["config"] == cfg and e["approach"] == ap and e["role"] == "opt"
                      and "meas_real_rel" in e), None)
            if r is None:
                continue
            off = {"A": -0.2, "B": 0.0, "C": 0.2}[ap]
            for c, key, mk in (("S", "meas_real_rel", "o"), ("F", "meas_fourier_rel", "s"), ("mesh", "meas_mesh_rel", "^")):
                ax.plot(k + off, r[key], marker=mk, color=AP_COL[ap], ls="none", ms=6,
                        label=(f"{ap}: measured {c}" if k == 0 else None))
            ax.plot(k + off, r["E_op"], marker="_", color=INK2, ms=14, mew=2, ls="none",
                    label=("measured E_op" if (k == 0 and ap == "A") else None))
    ax.axhline(C.EPS_TARGET, color=INK2, ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(FAMS)))
    ax.set_xticklabels([FLAB[f] for f in FAMS])
    ax.set_ylabel("measured error / ||Gamma_ref||_2")
    ax.set_title("(b) measured components at the optimum (seed 1, N = 256)", fontsize=9)
    ax.set_ylim(1e-14, 1e-6)
    ax.legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Figure J2. Where the error budget goes: fixed split vs joint allocation", fontsize=10.5, x=0.01,
                 ha="left")
    fig.tight_layout()
    save(fig, "figJ2_budget_allocation")


# ----------------------------------------------------------------------------- figure J3: time vs xi
def fig_time(D):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    for ax, cfg in zip(axes, ("strong_N256_s1", "uniform_N256_s1")):
        fam = cfg.split("_N")[0]
        g = D["sel"][f"{fam}_N256"]
        for ap in APPROACHES:
            per = g["approaches"][ap]["per_xi"]
            xs = sorted(float(k) for k in per)
            xs = [x_ for x_ in xs if abs(round(x_ / 0.025) * 0.025 - x_) < 1e-9]
            ax.plot(xs, [1e3 * per[f"{x_:.4f}"]["T_pred"] for x_ in xs], color=AP_COL[ap], lw=1.4,
                    label=f"{ap}: predicted")
            meas = sorted([r for r in bench_of(D, cfg, ap) if r["role"] in ("scan", "opt") or r["role"].startswith("xi_")],
                          key=lambda r: r["xi"])
            if meas:
                ax.plot([r["xi"] for r in meas], [1e3 * r["t_action_s"] for r in meas], ls="none", marker=AP_MK[ap],
                        color=AP_COL[ap], mfc="white", ms=5, label=f"{ap}: measured")
        bl = bench_of(D, cfg, "baseline")
        if bl:
            ax.axhline(1e3 * bl[0]["t_action_s"], color=GREY, ls=":", lw=1.3, label="existing baseline (p 8), measured")
        ax.set_yscale("log")
        ax.set_ylim(top=300)
        ax.set_xlabel("Ewald parameter xi")
        ax.set_ylabel("cached action time [ms]")
        ax.set_title(f"({'ab'[list(axes).index(ax)]}) {cfg}: best (kc, M, rc) per xi", fontsize=9)
        ax.legend(fontsize=6.5, ncol=2)
    fig.suptitle("Figure J3. Predicted and measured action time vs xi at certified tau = 2e-7", fontsize=10.5, x=0.01,
                 ha="left")
    fig.tight_layout()
    save(fig, "figJ3_time_vs_xi")


# ----------------------------------------------------------------------------- figure J4: speedup and error
def fig_speedup(D, summary):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    cfgs = [s["config"] for s in summary]
    xs = np.arange(len(cfgs))
    for ap, off in zip(APPROACHES, (-0.27, -0.09, 0.09, 0.27)):
        axes[0].plot(xs + off, [s[f"{ap}_speedup"] for s in summary], marker=AP_MK[ap], color=AP_COL[ap], ls="none",
                     ms=6, label=LABEL[ap])
        axes[1].plot(xs + off, [s[f"{ap}_E_op"] for s in summary], marker=AP_MK[ap], color=AP_COL[ap], ls="none", ms=6)
    axes[1].plot(xs, [s["baseline_E_op"] for s in summary], marker="*", color=GREY, ls="none", ms=8,
                 label="existing baseline (p = 8)")
    axes[0].axhline(1.0, color=GREY, ls=":", lw=1.2)
    axes[1].axhline(C.EPS_TARGET, color=INK2, ls="--", lw=1)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([c.replace("lj_equilibrated", "lj_eq").replace("_s", " s") for c in cfgs], rotation=60,
                           ha="right", fontsize=7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("measured speedup over baseline (T_base / T)")
    axes[0].set_title("(a) speedup at each approach's predicted optimum", fontsize=9)
    axes[0].legend(fontsize=6.5)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("measured E_op")
    axes[1].set_title("(b) measured operator error (dense full-periodic reference)", fontsize=9)
    axes[1].legend(fontsize=6.5, loc="center right")
    fig.suptitle("Figure J4. Measured speedup and accuracy, all configurations", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "figJ4_speedup_error")


# ----------------------------------------------------------------------------- tables
def tables(D):
    configs = []
    for r in D["bench"]:
        if r["config"] not in configs:
            configs.append(r["config"])
    configs.sort(key=lambda n: (int(n.split("_N")[1].split("_")[0]), n))
    summary, params = [], []
    for cfg in configs:
        N = int(cfg.split("_N")[1].split("_")[0])
        fam = cfg.split("_N")[0]
        g = D["sel"][f"{fam}_N{N}"]
        bl = bench_of(D, cfg, "baseline")[0]
        eb = err_of(D, cfg, "baseline", "baseline", bl["xi"])
        s = dict(config=cfg, N=N, family=fam, baseline_T_ms=1e3 * bl["t_action_s"], baseline_E_op=eb.get("E_op"),
                 baseline_M=bl["M"])
        for ap in APPROACHES:
            o = g["approaches"][ap]["opt"]
            b = [r for r in bench_of(D, cfg, ap, "opt")][0]
            e = err_of(D, cfg, ap, "opt", b["xi"])
            cand = bench_of(D, cfg, ap)
            fast = min(cand, key=lambda r: r["t_action_s"])
            s.update({f"{ap}_xi": o["xi"], f"{ap}_T_pred_ms": 1e3 * b["T_pred_s"], f"{ap}_T_ms": 1e3 * b["t_action_s"],
                      f"{ap}_speedup": bl["t_action_s"] / b["t_action_s"], f"{ap}_E_op": e.get("E_op"),
                      f"{ap}_measured_fastest_xi": fast["xi"], f"{ap}_measured_fastest_T_ms": 1e3 * fast["t_action_s"],
                      f"{ap}_measured_fastest_role": fast["role"], f"{ap}_n_measured": len(cand)})
            params.append(dict(config=cfg, N=N, approach=ap, label=LABEL[ap], xi=o["xi"], a=o["a"], b=o["b"],
                               rc=b["rc"], kc=b["kc"], M=b["M"], p=b["p"], eta=b["eta"], n_unique=b["n_unique"],
                               n_img=b["n_img"], share_S=o["share_S"], share_F=o["share_F"], share_mesh=o["share_mesh"],
                               share_quad=o["share_quad"], cert_rel_group=o["cert_rel_max"],
                               E_op=e.get("E_op"), E_op_k0=e.get("E_op_k0"), lam_min_h=e.get("lam_min_h"),
                               T_pred_ms=1e3 * b["T_pred_s"], T_meas_ms=1e3 * b["t_action_s"],
                               T_IQR_rel=b["t_action_iqr_rel"], T_setup_ms=1e3 * b["t_setup_s"],
                               speedup_vs_baseline=bl["t_action_s"] / b["t_action_s"],
                               speedup_vs_A=bench_of(D, cfg, "A", "opt")[0]["t_action_s"] / b["t_action_s"],
                               measured_fastest_xi=fast["xi"], measured_fastest_kc=fast["kc"], measured_fastest_M=fast["M"],
                               measured_fastest_T_ms=1e3 * fast["t_action_s"], n_measured=len(cand)))
        summary.append(s)
    C.write_csv(RESULTS_JOINT / "joint_summary.csv", summary)
    C.write_csv(RESULTS_JOINT / "joint_parameter_results.csv", params)
    tb = [{k: r.get(k) for k in ("config", "N", "approach", "role", "xi", "rc", "kc", "M", "p", "eta", "n_unique", "n_img",
                                 "t_action_s", "t_action_q1_s", "t_action_q3_s", "t_action_min_s", "t_action_max_s",
                                 "t_action_n", "t_real_s", "t_spread_s", "t_fft_influence_s", "t_gather_s",
                                 "t_existing_pairdiff_matvec_s", "T_pred_s", "T_pred_real_s", "T_pred_spread_gather_s",
                                 "T_pred_fft_s", "T_pred_overhead_s", "t_setup_s", "cert_rel_group")} for r in D["bench"]]
    C.write_csv(RESULTS_JOINT / "joint_timing.csv", tb)
    er = [{k: r.get(k) for k in ("config", "N", "approach", "role", "xi", "rc", "kc", "M", "E_op", "E_op_k0", "E_fro",
                                 "sym_resid", "null_resid", "lam_min_h", "lam_min_ref", "cached_vs_dense_max",
                                 "basis_vs_independent", "E_op_basis", "meas_real_rel", "meas_fourier_rel", "meas_mesh_rel",
                                 "cert_S_abs", "cert_F_abs", "cert_mesh_orig_abs", "cert_mesh_hz_abs", "norm2_ref",
                                 "lam_lb", "passes")} for r in D["err"]["rows"]]
    C.write_csv(RESULTS_JOINT / "joint_errors.csv", er)
    return summary


def reproducibility():
    import hashlib
    from joint_optimizer import NEGL, M_CAP, A_GRID, G_RC, VARIANT
    files = sorted([f for f in list(RESULTS_JOINT.glob("*")) + list(FIGURES_JOINT.glob("*.png")) if f.is_file()
                    and f.name != "reproducibility_joint.json"])
    cmds = (RESULTS_JOINT / "commands.log").read_text().splitlines() if (RESULTS_JOINT / "commands.log").exists() else []
    C.write_json(RESULTS_JOINT / "reproducibility_joint.json", dict(
        environment=C.provenance(), commands_executed=cmds,
        problem=dict(objective="min T_pred over (xi, kc, M, rc), p = 7", tau=TAU, constraint="eps_S + eps_F + eps_mesh + "
                     "eps_quad <= tau * Lambda_lb for every configuration of a level", approaches=LABEL, mesh_bound=VARIANT,
                     negligible_cut=NEGL, M_cap=M_CAP, a_grid=A_GRID, rc_grid_points=G_RC,
                     xi_grid="0.40..1.60 step 0.025, refined 0.005 within +-0.025 of each approach's best"),
        inputs=dict(configs="configs/*.npz (unchanged)", references="raw/<config>_Gref.npy (unchanged, git-ignored)",
                    lambda_lb_and_norms="results/references.json (unchanged)"),
        files_sha256={str(f.relative_to(C.HERE)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}))


def main():
    C.RESULTS = RESULTS_JOINT
    C.log_command()
    style()
    D = load()
    summary = tables(D)
    fig_alias(D); fig_budget(D); fig_time(D); fig_speedup(D, summary)
    reproducibility()
    C.log(f"joint figures -> {FIGURES_JOINT}; {len(summary)} configurations")


if __name__ == "__main__":
    main()
