#!/usr/bin/env python3
"""
Independent check of Fig. A1 / Fig. A2 (../paper_figures.py six_sweeps) against results/sweeps.csv.

For every panel, setting and family the curves drawn in the matplotlib figure (read back from the Line2D objects,
not from the plotting code's own bookkeeping) are compared with medians / minima / maxima recomputed here from the
CSV (median lines and min-max bands): Fig. A1 must equal E_G and Fig. A2 must equal E_GX (and differ from the other
metric); k_c/xi must equal
2 sF_eff of the recorded rows; every point must aggregate exactly 3 seeds; the floors must come from ref_floor_G /
ref_floor_GX. Nothing is written to results/.

    python3 validate_figA.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paper_figures as PF          # noqa: E402

FAMS = ("uniform", "moderate", "strong")
XDEF = {"sR": lambda r: float(r["value"]), "sF": lambda r: 2.0 * float(r["sF_eff"]), "M": lambda r: float(r["value"]),
        "p": lambda r: float(r["value"]), "rho": lambda r: float(r["value"]), "kappa": lambda r: float(r["value"])}


def load():
    with open(HERE / "results" / "sweeps.csv") as fh:
        return [r for r in csv.DictReader(fh) if r["mesh_valid"] == "True"]


def expected(rows, sw, st, fam, metric):
    sel = [r for r in rows if r["sweep"] == sw and r["setting"] == st and r["family"] == fam]
    xs = sorted({XDEF[sw](r) for r in sel})
    out = []
    for x in xs:
        v = np.array([float(r[metric]) for r in sel if XDEF[sw](r) == x])
        out.append((x, np.median(v), v.min(), v.max(), len(v)))
    return out


def main():
    rows = load()
    figs = {}
    PF.save = lambda fig, path: figs.__setitem__(Path(path).name, fig)        # keep the figure, do not write files
    PF.style()
    import ewald_exp_lib as E                                              # the reader used by the plots stage
    typed = [r for r in E.read_csv(HERE / "results" / "sweeps.csv") if r.get("mesh_valid")]
    for metric in ("E_G", "E_GX"):
        PF.six_sweeps(typed, metric, Path("."), [])
    n_checked, problems = 0, []
    for metric, other, name in (("E_G", "E_GX", "figA1_gamma_operator_6sweeps"),
                                ("E_GX", "E_G", "figA2_gamma_action_6sweeps")):
        fig = figs[name]
        for ax, (sw, title, *_rest) in zip(fig.axes, PF.A_PANELS):
            lines = [ln for ln in ax.lines if ln.get_marker() in ("o", "s", "^")]
            settings = [st for st in ("baseline", "moderate") if any(r["sweep"] == sw and r["setting"] == st for r in rows)]
            k = 0
            for st in settings:
                for fam in FAMS:
                    exp = expected(rows, sw, st, fam, metric)
                    oth = expected(rows, sw, st, fam, other)
                    ln = lines[k]
                    k += 1
                    xd, yd = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
                    ex = np.array([e[0] for e in exp])
                    em = np.array([e[1] for e in exp])
                    if not (np.array_equal(xd, ex) and np.array_equal(yd, em)):
                        problems.append(f"{name} {title} {st} {fam}: plotted data differ from CSV {metric}")
                    if np.any(yd == np.array([o[1] for o in oth])):
                        problems.append(f"{name} {title} {st} {fam}: a plotted value equals {other}")
                    if any(e[4] != 3 for e in exp):
                        problems.append(f"{name} {title} {st} {fam}: not 3 seeds per point")
                    expected_style = "--" if st == "moderate" else "-"
                    if ln.get_linestyle() != expected_style:
                        problems.append(f"{name} {title} {st} {fam}: line style {ln.get_linestyle()}")
                    n_checked += len(exp)
            if k != len(lines):
                problems.append(f"{name} {title}: {len(lines)} data lines drawn, {k} expected")
            # shaded min-max bands: one per curve, in drawing order, spanning exactly the seed minima and maxima
            bands = [c for c in ax.collections if hasattr(c, "get_paths") and len(c.get_paths())]
            if len(bands) != k:
                problems.append(f"{name} {title}: {len(bands)} bands for {k} curves")
            k = 0
            for st in settings:
                for fam in FAMS:
                    exp = expected(rows, sw, st, fam, metric)
                    yv = np.concatenate([pth.vertices[:, 1] for pth in bands[k].get_paths()])
                    k += 1
                    if not (np.all(np.isin([e[2] for e in exp], yv)) and np.all(np.isin([e[3] for e in exp], yv))
                            and yv.min() == min(e[2] for e in exp) and yv.max() == max(e[3] for e in exp)):
                        problems.append(f"{name} {title} {st} {fam}: min-max band differs from CSV {metric}")
        # floors
        fkey = PF.A_METRICS[metric]["floor"]
        for ax, (sw, title, *_r) in zip(fig.axes, PF.A_PANELS):
            sel = [r for r in rows if r["sweep"] == sw]
            xs = sorted({XDEF[sw](r) for r in sel})
            fm = np.array([np.median([float(r[fkey]) for r in sel if XDEF[sw](r) == x]) for x in xs])
            fl = [ln for ln in ax.lines if ln.get_marker() == "x"]
            if sw in ("rho", "kappa"):
                if not (fl and np.array_equal(np.asarray(fl[0].get_ydata(), float), fm)):
                    problems.append(f"{name} {title}: per-x floor differs from {fkey}")
            else:
                hl = [ln for ln in ax.lines if ln.get_marker() in ("None", "", None) and len(ln.get_ydata()) == 2
                      and ln.get_ydata()[0] == ln.get_ydata()[1]]
                if not (hl and hl[0].get_ydata()[0] == float(np.median(fm)) and np.ptp(fm) == 0):
                    problems.append(f"{name} {title}: constant floor differs from {fkey}")
    # Fourier axis
    sf = [r for r in rows if r["sweep"] == "sF"]
    uF = sorted({2.0 * float(r["sF_eff"]) for r in sf})
    plotted_uF = set()
    for name in ("figA1_gamma_operator_6sweeps", "figA2_gamma_action_6sweeps"):
        for ln in figs[name].axes[1].lines:
            if ln.get_marker() in ("o", "s", "^"):
                plotted_uF |= set(map(float, ln.get_xdata()))
    if sorted(plotted_uF) != uF:
        problems.append("k_c/xi axis values are not exactly 2 * sF_eff")
    print(f"checked {n_checked} plotted (median, x) points in 2 figures x 6 panels")
    print(f"k_c/xi plotted: {len(uF)} values = 2 sF_eff, range {uF[0]:.4f} .. {uF[-1]:.4f}; "
          f"requested s_F {min(float(r['sF_req']) for r in sf):g} .. {max(float(r['sF_req']) for r in sf):g}")
    for sw in XDEF:
        for st in ("baseline", "moderate"):
            n = len({XDEF[sw](r) for r in rows if r["sweep"] == sw and r["setting"] == st})
            print(f"  {sw:6s} {st:9s}: {n:2d} x values" + ("" if n else "  (not part of the sweep design: no data)"))
    if problems:
        print("PROBLEMS:")
        for p_ in problems:
            print("  " + p_)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
