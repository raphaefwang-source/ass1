#!/usr/bin/env python3
"""
Figures and summary tables of the p search (Stage 1) and the direct vs PPPM vs dense comparison (Stage 2) at N = 512.
Reads results_psearch/ and results_direct512/, writes figures_psearch/ and the summary CSVs.

    python3 make_figures_psearch.py
"""
import common as C  # noqa: F401  (thread settings before numpy)
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common as C  # noqa: E402
from run_psearch_dense512 import RESULTS_PSEARCH, CONFIGS, IMPLS, path  # noqa: E402
from run_direct512 import RESULTS_DIRECT, out_path, QS  # noqa: E402

FIG = C.HERE / "figures_psearch"
TAU = C.EPS_TARGET
COL = {4: "#8c564b", 5: "#1f77b4", 6: "#ff7f0e", 7: "#2ca02c", 8: "#d62728", 9: "#9467bd", 10: "#17becf", 11: "#7f7f7f"}
SHORT = {"strong_N512_s1": "strong", "uniform_N512_s1": "uniform", "moderate_N512_s1": "moderate"}


def load_stage1(name, tag="main"):
    sel = C.read_json(path("selection", name))
    bench = C.read_json(path("bench", name, tag))
    err = C.read_json(path("errors", name))["rows"]
    opt = C.read_json(path("optimum", name, tag))
    for c in bench["candidates"]:
        c["E_op"] = err[c["key"]]["E_op"]
    return sel, bench, opt


def per_p_rows(name, sel, bench, opt):
    """One row per p: model optimum (certified errors, prediction) and the measured best candidate of that p."""
    rows = []
    rk = sel["rank_impl"]
    conf = {(r["key"], r["impl"]): r for r in opt["confirm_rows"]}
    for p in sorted(int(k) for k in sel["per_p"]):
        o = sel["per_p"][str(p)]["opt"]
        cs = [c for c in bench["candidates"] if c["p"] == p and c["E_op"] <= TAU]
        if not cs:
            continue
        cp = [r for r in opt["confirm_rows"] if r["p"] == p]
        if cp:                                   # confirmed: the fastest confirmed (candidate, implementation) of p
            r0 = min(cp, key=lambda r: r["T_action_median_ms"])
            mb = next(c for c in cs if c["key"] == r0["key"])
            ci = r0["impl"]
        else:                                    # not confirmed: the fastest by screening
            mb = min(cs, key=lambda c: c["T_action_screen_best_ms"])
            ci = mb["T_action_screen_best_impl"]
        oc = next((c for c in bench["candidates"] if c["key"] == f"p{p}_xi{o['xi']:.4f}_kc{o['kc']:.6f}_M{o['M']}_rc{o['rc']:.6f}"), None)
        cr = conf.get((mb["key"], ci))
        rows.append(dict(config=name, p=p,
                         model_xi=o["xi"], model_rc=o["rc"], model_kc=o["kc"], model_M=o["M"], model_eta=o["eta"],
                         model_n_img=o["n_img_mean"], model_eps_S=o["eps_S_abs"], model_eps_F=o["eps_F_abs"],
                         model_eps_mesh=o["eps_mesh_abs"], model_eps_quad_est=o["eps_quad_abs_est"],
                         model_cert_rel=o["cert_rel_max"], model_cert_rel_no_quad=o["eps_cert_rel_no_quad"],
                         model_T_pred_ms=o[f"T_pred_{rk}_ms"],
                         model_E_op=oc["E_op"] if oc else None,
                         model_T_action_cached_ms=oc["T_action_cached_screen_median_ms"] if oc else None,
                         model_T_action_pairdiff_ms=oc["T_action_pairdiff_screen_median_ms"] if oc else None,
                         model_T_setup_cached_ms=oc["T_setup_cached_ms"] if oc else None,
                         model_T_setup_pairdiff_ms=oc["T_setup_pairdiff_ms"] if oc else None,
                         best_key=mb["key"], best_roles="+".join(mb["roles"]), best_xi=mb["xi"], best_rc=mb["rc"],
                         best_kc=mb["kc_shell"], best_M=mb["M"], best_eta=mb["eta_actual"], best_n_img=mb["n_img"],
                         best_eps_S=mb["eps_S_abs"], best_eps_F=mb["eps_F_abs"], best_eps_mesh=mb["eps_mesh_abs"],
                         best_eps_quad_est=mb["eps_quad_abs_est"], best_cert_rel=mb["cert_rel_max"],
                         best_E_op=mb["E_op"], best_impl=ci, best_T_pred_ms=mb[f"T_pred_{ci}_ms"],
                         best_T_action_screen_ms=mb["T_action_screen_best_ms"],
                         best_T_action_confirm_ms=cr["T_action_median_ms"] if cr else None,
                         best_T_action_confirm_q1_ms=cr["T_action_q1_ms"] if cr else None,
                         best_T_action_confirm_q3_ms=cr["T_action_q3_ms"] if cr else None,
                         best_T_setup_ms=mb[f"T_setup_{ci}_ms"]))
    return rows


def fig_per_p(data):
    fig, axs = plt.subplots(1, len(data), figsize=(4.6 * len(data), 3.9), sharey=True)
    for ax, (name, (sel, bench, opt, rows)) in zip(np.atleast_1d(axs), data.items()):
        ps = [r["p"] for r in rows]
        ax.plot(ps, [r["model_T_pred_ms"] for r in rows], "k--o", mfc="none", label="model optimum: predicted")
        for impl, mk in (("cached", "s"), ("pairdiff", "^")):
            best = [min([c[f"T_action_{impl}_screen_median_ms"] for c in bench["candidates"]
                         if c["p"] == p and c["E_op"] <= TAU]) for p in ps]
            ax.plot(ps, best, mk + "-", ms=5, alpha=0.8, label=f"measured best of p ({impl}, screening)")
        cf = [(r["p"], r["best_T_action_confirm_ms"], r["best_T_action_confirm_q1_ms"], r["best_T_action_confirm_q3_ms"])
              for r in rows if r["best_T_action_confirm_ms"] is not None]
        if cf:
            p_, m_, a_, b_ = map(np.array, zip(*cf))
            ax.errorbar(p_, m_, yerr=[m_ - a_, b_ - m_], fmt="o", color="C3", ms=6, capsize=3,
                        label="confirmed (150 rounds, IQR)")
        b = opt["best"]
        ax.plot([b["p"]], [b["T_action_median_ms"]], "*", color="gold", mec="k", ms=16, label="measured optimum")
        ax.set_title(SHORT[name] + "_N512_s1")
        ax.set_xlabel("B-spline order p")
        ax.grid(alpha=0.3)
    np.atleast_1d(axs)[0].set_ylabel("one action $\\Gamma_h X$ [ms]")
    np.atleast_1d(axs)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "figP1_time_vs_p.png", dpi=150)
    plt.close(fig)


def fig_pred_vs_meas(data):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, impl in zip(axs, IMPLS):
        allv = []
        for name, (sel, bench, opt, rows) in data.items():
            mk = {"strong_N512_s1": "o", "uniform_N512_s1": "s", "moderate_N512_s1": "^"}[name]
            for c in bench["candidates"]:
                pr, me = c[f"T_pred_{impl}_ms"], c[f"T_action_{impl}_screen_median_ms"]
                ax.plot(pr, me, mk, color=COL[c["p"]], ms=4, alpha=0.8)
                allv += [pr, me]
        lo, hi = min(allv) * 0.9, max(allv) * 1.1
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
        ax.plot([lo, hi], [1.1 * lo, 1.1 * hi], "k:", lw=0.6)
        ax.plot([lo, hi], [0.9 * lo, 0.9 * hi], "k:", lw=0.6)
        ax.set_xlabel("predicted (per-p cost model) [ms]")
        ax.set_ylabel("measured median, screening [ms]")
        ax.set_title(f"{impl} action, all candidates (dotted: $\\pm$10 %)")
        ax.grid(alpha=0.3)
    for p, c in COL.items():
        axs[1].plot([], [], "o", color=c, label=f"p = {p}")
    axs[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "figP2_pred_vs_meas.png", dpi=150)
    plt.close(fig)


def fig_budget(data):
    fig, axs = plt.subplots(1, len(data), figsize=(4.6 * len(data), 3.9), sharey=True)
    for ax, (name, (sel, bench, opt, rows)) in zip(np.atleast_1d(axs), data.items()):
        lam = sel["lam_lb"]
        ps = np.array([r["p"] for r in rows])
        parts = [("model_eps_S", "real space $\\varepsilon_S$"), ("model_eps_F", "Fourier $\\varepsilon_F$"),
                 ("model_eps_mesh", "mesh $\\varepsilon_{mesh}$ (HZ)"), ("model_eps_quad_est", "quadrature (estimate)")]
        bottom = np.zeros(len(rows))
        for k, lab in parts:
            v = np.array([r[k] for r in rows]) / (TAU * lam)
            ax.bar(ps, v, bottom=bottom, label=lab, width=0.7)
            bottom += v
        ax.plot(ps, [(r["model_E_op"] or np.nan) / TAU for r in rows], "kD", ms=6, label="measured $E_{op}$/$\\tau$")
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("p (model optimum of each order)")
        ax.set_title(SHORT[name] + "_N512_s1")
    np.atleast_1d(axs)[0].set_ylabel("certified share of $\\tau\\Lambda_{lb}$;  measured $E_{op}/\\tau$")
    np.atleast_1d(axs)[0].legend(fontsize=7, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG / "figP3_error_budget_vs_p.png", dpi=150)
    plt.close(fig)


def fig_xi(data, name="strong_N512_s1"):
    sel, bench, opt, rows = data[name]
    rk = sel["rank_impl"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for p in sorted(int(k) for k in sel["per_p"]):
        per = sel["per_p"][str(p)]["per_xi"]
        xs = np.array(sorted(float(k) for k in per))
        ax.plot(xs, [per[f"{x:.4f}"][f"T_pred_{rk}_ms"] for x in xs], "-", color=COL[p], lw=1, label=f"p = {p}")
        cs = [c for c in bench["candidates"] if c["p"] == p]
        ax.plot([c["xi"] for c in cs], [c[f"T_action_{rk}_screen_median_ms"] for c in cs], "o", color=COL[p], ms=4)
    b = opt["best"]
    ax.plot([b["xi"]], [b["T_action_median_ms"]], "*", color="gold", mec="k", ms=15)
    ax.set_xlabel("$\\xi$")
    ax.set_ylabel(f"one action [ms] ({rk})")
    ax.set_ylim(0, 3 * b["T_action_median_ms"])
    ax.set_title("strong_N512_s1: predicted per-$\\xi$ optimum (lines),\nscreened candidates (dots), confirmed optimum (star)", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "figP4_xi_scan_strong.png", dpi=150)
    plt.close(fig)


def fig_convergence(conv):
    fig, axs = plt.subplots(1, len(conv), figsize=(4.6 * len(conv), 3.9), sharey=True)
    for ax, (name, d) in zip(np.atleast_1d(axs), conv.items()):
        R = np.array([r["R"] for r in d["rows"]])
        ax.semilogy(R, [r["E_ref"] for r in d["rows"]], "-", label="$\\|\\Gamma_D(R)-\\Gamma_{ref}\\|/\\|\\Gamma_{ref}\\|$")
        ax.semilogy(R, [max(r["E_conv"], 1e-17) for r in d["rows"]], "--",
                    label="$\\|\\Gamma_D(R)-\\Gamma_D(R_{max})\\|/\\|\\Gamma_D(R_{max})\\|$")
        ax.semilogy(R, [r["bound_rel_lam"] for r in d["rows"]], ":", label="certified tail bound / $\\Lambda_{lb}$")
        ax.axhline(TAU, color="k", lw=0.8)
        ax.axvline(d["R_meas"], color="C3", lw=0.8, label=f"$R_{{meas}}$ = {d['R_meas']}")
        ax.axvline(d["R_cert"], color="C4", lw=0.8, ls="--", label=f"$R_{{cert}}$ = {d['R_cert']}")
        ax.set_xlabel("direct truncation radius R")
        ax.set_title(SHORT[name] + "_N512_s1")
        ax.set_ylim(1e-13, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5)
    np.atleast_1d(axs)[0].set_ylabel("relative spectral-norm error")
    fig.tight_layout()
    fig.savefig(FIG / "figD1_direct_convergence.png", dpi=150)
    plt.close(fig)


METHODS = (("direct_images_Rmeas", "direct, image-resolved"), ("direct_pairs_Rmeas", "direct, pair-summed"),
           ("dense_phys_Rmeas", "dense physical $\\Gamma$ (GEMV)"), ("pppm", "PPPM (Stage 1 optimum)"))


def fig_stage2(s2):
    fig, axs = plt.subplots(1, len(s2), figsize=(4.6 * len(s2), 4.0), sharey=True)
    for ax, (name, d) in zip(np.atleast_1d(axs), s2.items()):
        sm = d["summary"]
        xs = np.arange(len(METHODS))
        ax.bar(xs - 0.2, [sm[k]["T_setup_ms"] for k, _ in METHODS], 0.4, label="setup")
        ax.bar(xs + 0.2, [sm[k]["T_action_ms"] for k, _ in METHODS], 0.4, label="one action")
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for _, lab in METHODS], rotation=25, ha="right", fontsize=7)
        ax.set_yscale("log")
        ax.set_title(SHORT[name] + "_N512_s1")
        ax.grid(alpha=0.3, axis="y")
    np.atleast_1d(axs)[0].set_ylabel("median time [ms]")
    np.atleast_1d(axs)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "figD2_setup_action_times.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, len(s2), figsize=(4.6 * len(s2), 3.9), sharey=True)
    q = np.logspace(0, 3, 200)
    for ax, (name, d) in zip(np.atleast_1d(axs), s2.items()):
        sm = d["summary"]
        for k, lab in METHODS:
            ax.loglog(q, sm[k]["T_setup_ms"] + q * sm[k]["T_action_ms"], label=lab)
        for qq in QS:
            ax.axvline(qq, color="k", lw=0.4, alpha=0.4)
        ax.set_xlabel("actions per configuration q")
        ax.set_title(SHORT[name] + "_N512_s1")
        ax.grid(alpha=0.3)
    np.atleast_1d(axs)[0].set_ylabel("$T_{setup} + q\\,T_{action}$ [ms] (measured medians)")
    np.atleast_1d(axs)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "figD3_end_to_end.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data, allrows, cand = {}, [], []
    for name in CONFIGS:
        if not path("optimum", name, "main").exists():
            continue
        sel, bench, opt = load_stage1(name)
        rows = per_p_rows(name, sel, bench, opt)
        data[name] = (sel, bench, opt, rows)
        allrows += rows
        for c in bench["candidates"]:
            cand.append({k: (v if not isinstance(v, list) else "+".join(map(str, v))) for k, v in c.items()
                         if not k.startswith("T_pred_cached_") and not k.startswith("T_pred_pairdiff_")
                         and k not in ("B", "g", "eps_S_abs_per_seed", "lam_lb_per_seed", "cert_rel_per_seed",
                                       "n_img_per_seed")} | dict(config=name))
    if data:
        C.write_csv(RESULTS_PSEARCH / "psearch_per_p.csv", allrows)
        C.write_csv(RESULTS_PSEARCH / "psearch_candidates.csv", cand)
        fig_per_p(data)
        fig_pred_vs_meas(data)
        fig_budget(data)
        if "strong_N512_s1" in data:
            fig_xi(data)
    conv = {n: C.read_json(out_path("convergence", n)) for n in CONFIGS if out_path("convergence", n).exists()}
    if conv:
        fig_convergence(conv)
        C.write_csv(RESULTS_DIRECT / "direct_convergence.csv",
                    [dict(config=n, **r) for n, d in conv.items() for r in d["rows"]])
    s2 = {n: C.read_json(out_path("stage2", n)) for n in CONFIGS if out_path("stage2", n).exists()}
    if s2:
        fig_stage2(s2)
        rows = []
        for n, d in s2.items():
            for k, v in d["summary"].items():
                rows.append(dict(config=n, method=k, T_setup_ms=v["T_setup_ms"], T_action_ms=v["T_action_ms"],
                                 action_ratio_vs_pppm=v["action_speedup_vs_pppm"],
                                 setup_ratio_vs_pppm=v["setup_ratio_vs_pppm"],
                                 **{f"total_q{q}_ms": v["total_ms"][str(q)] for q in QS},
                                 break_even_q_vs_pppm=v["break_even_q_vs_pppm"]))
        C.write_csv(RESULTS_DIRECT / "stage2_summary.csv", rows)
    C.log(f"figures written to {FIG}")


if __name__ == "__main__":
    main()
