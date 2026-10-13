"""
Tables and figures of the Lanczos rank study (sec6_lanczos.py report). Reads sec6_results/lanczos/raw/*.json and
rank_selection.json; writes sec6_results/lanczos/{required_ranks.csv, error_at_ranks.csv, reference_checks.csv,
summary.json, fig_lanczos_*.png}.
"""
import csv
import json

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sec6_config  # noqa: E402
from sec6_lanczos import OUT, RAW, required_rank  # noqa: E402

FAM = {"caseA": ("#0b0b0b", "D", "case A"), "caseB": ("#52514e", "v", "case B"),
       "uniform": ("#2a78d6", "o", "uniform"), "lattice": ("#eb6834", "s", "perturbed lattice"),
       "cluster": ("#1baf7a", "^", "clustered")}
INK2, GRID = "#52514e", "#e4e3df"
CAND = (8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 200)


def style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                         "savefig.dpi": 200, "lines.linewidth": 1.3, "mathtext.fontset": "cm"})


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def main():
    style()
    B = sec6_config.budgets()
    margin = B["reference_margin"]
    small = json.loads((RAW / "small_cases.json").read_text())["cases"]
    large = json.loads((RAW / "large_cases.json").read_text())["cases"]
    budgets = [("noise|prod", B["noise_rel_action"]), ("damp|prod", B["damping_rel_action"])] + \
              [("noise|prod", t) for t in B["noise_high_accuracy"]] + \
              [(f"noise|tau{t:g}", B["noise_rel_action"]) for t in (0.2, 0.5, 2, 5)] + \
              [(f"damp|tau{t:g}", B["damping_rel_action"]) for t in (0.2, 0.5, 2, 5)]
    # required ranks per case and budget
    req_rows = []
    for c in small:
        for key, tol in budgets:
            r = required_rank(c["err"][key], tol, c["S"])
            req_rows.append(dict(scale="small", config=c["name"], family=c["family"], N=c["N"], pos=c["pos_seed"],
                                 vec=c["vec_index"], function=key, tol=tol, required_rank=r if r else "not reached",
                                 reference="dense", ref_ok=True, tau_prod=c["tau_prod"], lam_max=c["lam_max"],
                                 lam_source=c["lam_source"], S=c["S"]))
    ref_rows = []
    for c in large:
        for key, tol in budgets:
            chk = c["ref_check"][key]
            ok = max(chk["S-20"], chk["S-40"]) <= margin * tol
            r = required_rank(c["err"][key], tol, c["S"] - 40) if ok else None
            req_rows.append(dict(scale="large", config=f"uniform_N{c['N']}_pos{c['pos_index']}", family="uniform",
                                 N=c["N"], pos=c["pos_index"], vec=c["vec_index"], function=key, tol=tol,
                                 required_rank=(r if r else f"not reached up to {c['S'] - 40}") if ok else "UNRESOLVED",
                                 reference=f"Lanczos S={c['S']}", ref_ok=ok, ref_S_minus_20=chk["S-20"],
                                 ref_S_minus_40=chk["S-40"], tau_prod=c["tau_prod_est"], lam_max=c["lam_max_est"],
                                 lam_source=c["lam_source"], S=c["S"]))
        for key in ("noise|prod", "damp|prod"):
            ref_rows.append(dict(N=c["N"], pos=c["pos_index"], vec=c["vec_index"], function=key, S=c["S"],
                                 **{f"diff_{k}": v for k, v in c["ref_check"][key].items()},
                                 orth=c["orth"], coeff_vs_vector=c["coeff_vs_vector_rel_dev"],
                                 negative_ritz=sum(c["ritz"]["n_negative"])))
    write_csv(OUT / "required_ranks.csv", req_rows)
    write_csv(OUT / "reference_checks.csv", ref_rows)
    # sample maxima at candidate ranks, production dt
    err_rows = []
    groups = {}
    for c in small:
        groups.setdefault((c["N"], c["family"], "small"), []).append((c, c["S"]))
    for c in large:
        groups.setdefault((c["N"], "uniform", "large"), []).append((c, c["S"] - 40))
    for (N, fam, scale), cs in sorted(groups.items()):
        for key in ("noise|prod", "damp|prod"):
            row = dict(scale=scale, N=N, family=fam, function=key, n_cases=len(cs),
                       n_positions=len({(c.get("name"), c.get("pos_index")) for c, _ in cs}))
            for r in CAND:
                v = [c["err"][key][r - 1] for c, upto in cs if r <= upto]
                row[f"r{r}"] = max(v) if v and len(v) == len(cs) else "not computed"
            err_rows.append(row)
    write_csv(OUT / "error_at_ranks.csv", err_rows)
    neg_small = sum(sum(c["ritz"]["n_negative"]) for c in small)
    neg_large = sum(sum(c["ritz"]["n_negative"]) for c in large)
    summ = dict(provenance=sec6_config.provenance("sec6_lanczos.py", stage="report"), n_small=len(small),
                n_large=len(large), negative_ritz_small=neg_small, negative_ritz_large=neg_large,
                literal_vs_dense_max=max(c["literal_vs_dense"] for c in small),
                orth_max=max(c["orth"] for c in small + large),
                coeff_vs_vector_max=max(c["coeff_vs_vector_rel_dev"] for c in large),
                tau_prod_small=[min(c["tau_prod"] for c in small), max(c["tau_prod"] for c in small)],
                tau_prod_large_est={str(N): [min(c["tau_prod_est"] for c in large if c["N"] == N),
                                             max(c["tau_prod_est"] for c in large if c["N"] == N)]
                                    for N in sorted({c["N"] for c in large})},
                lam_max_large_est={str(N): [min(c["lam_max_est"] for c in large if c["N"] == N),
                                            max(c["lam_max_est"] for c in large if c["N"] == N)]
                                   for N in sorted({c["N"] for c in large})},
                ref_diff_max={key: dict(S20=max(c["ref_check"][key]["S-20"] for c in large),
                                        S40=max(c["ref_check"][key]["S-40"] for c in large))
                              for key in large[0]["ref_check"]})
    sel_f = OUT / "rank_selection.json"
    if sel_f.exists():
        summ["selection"] = json.loads(sel_f.read_text())["production"]
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")

    # figure 1: error vs rank at the production dt
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    for ax, key, tol, title in ((axs[0], "noise|prod", B["noise_rel_action"], "noise $f_{dt}(\\Gamma_h)\\Pi z$"),
                                (axs[1], "damp|prod", B["damping_rel_action"], "damping $e^{-dt\\Gamma_h/m}\\Pi v$")):
        for c in small:
            col, mk, lab = FAM[c["family"]]
            e = np.array(c["err"][key])
            ax.plot(np.arange(1, len(e) + 1), np.maximum(e, 1e-17), "-", color=col, lw=0.5, alpha=0.5)
        cmap = plt.get_cmap("Greys")
        Ns = sorted({c["N"] for c in large})
        for i, N in enumerate(Ns):
            for c in [c for c in large if c["N"] == N]:
                e = np.array(c["err"][key][:c["S"] - 40])
                ax.plot(np.arange(1, len(e) + 1), np.maximum(e, 1e-17), "-", color=cmap(0.35 + 0.6 * i / max(1, len(Ns) - 1)),
                        lw=0.6)
        ax.axhline(tol, color="#c0392b", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 2)
        ax.set_xlim(0, 200 if key.startswith("noise") else 40)
        ax.set_xlabel("Lanczos rank $s$")
        ax.set_ylabel("relative action error")
        ax.set_title(title, loc="left")
    handles = [plt.Line2D([], [], color=FAM[f][0], lw=1.2, label=FAM[f][2]) for f in FAM]
    handles.append(plt.Line2D([], [], color="0.4", lw=1.2, label="uniform N=512..16000 (grey: N)"))
    axs[0].legend(handles=handles, loc="upper right", fontsize=5.6)
    # required rank vs N: uniform fixed-density series as lines, other families as markers
    ax = axs[2]
    for (key, tol), col in ((("noise|prod", 1e-3), "#2a78d6"), (("noise|prod", 1e-6), "#eb6834"),
                            (("noise|prod", 1e-8), "#1baf7a"), (("damp|prod", 1e-6), "#0b0b0b")):
        uni, oth = {}, {}
        for r in req_rows:
            if r["function"] == key and r["tol"] == tol and isinstance(r["required_rank"], int):
                tgt = uni if r["family"] == "uniform" and (r["N"] >= 128) else oth
                tgt.setdefault((r["N"], r["family"]), []).append(r["required_rank"])
        Ns = sorted({n for n, f in uni})
        ax.plot(Ns, [max(uni[(n, "uniform")]) for n in Ns], "o-", color=col, ms=3,
                label=f"{'noise' if key.startswith('noise') else 'damping'} {tol:g} (uniform, rho_A)")
        for (n, fam), v in oth.items():
            ax.plot([n], [max(v)], FAM[fam][1], color=col, ms=4, mfc="white", mew=0.9)
    if sel_f.exists():
        prodr = json.loads(sel_f.read_text())["production"]
        for v, lab in ((prodr["noise_rank"], "chosen noise rank"), (prodr["damping_rank"], "chosen damping rank")):
            ax.axhline(v, color=INK2, lw=0.7, ls=":")
            ax.text(35, v + 3, f"{lab} {v}", fontsize=6, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("$N$")
    ax.set_ylabel("largest required rank (sample max)")
    ax.set_title("required rank, production $dt$", loc="left")
    ax.legend(loc="upper left", fontsize=5.6)
    fig.savefig(OUT / "fig_lanczos_rank.png")
    print(json.dumps({k: v for k, v in summ.items() if k != "provenance"}, indent=1)[:3000])
