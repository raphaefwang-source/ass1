"""Weighted nonlinear least-squares fits of the two candidate radial laws, a smoothing-spline baseline,
goodness-of-fit metrics and a group-level (configuration / time-block) bootstrap.

Model predictions are BIN AVERAGES over the actually sampled pair distances in each bin, not values at the bin
centre: <gamma(r)>_bin = mean_{pairs in bin} gamma(r_p). This removes the Jensen bias of the 1/r law at small r."""
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit


def gamma_A(r, A, kappa):
    return A * np.exp(-kappa * r) / r


def gamma_B(r, B, kappa):
    return B * r * np.exp(-kappa * r)


MODELS = {"A": gamma_A, "B": gamma_B}
LABELS = {"A": "Model A: A e^{-κr}/r", "B": "Model B: B r e^{-κr}"}


class BinAverager:
    """Precomputes per-bin pair-distance samples so that model bin averages are cheap during fitting."""

    def __init__(self, r_samples, b_samples, bins, max_per_bin=4000, rng=None):
        rng = np.random.default_rng(1) if rng is None else rng
        self.bins = np.asarray(bins)
        self.r, self.w = [], []
        for b in self.bins:
            rb = r_samples[b_samples == b]
            if len(rb) > max_per_bin:
                rb = rng.choice(rb, max_per_bin, replace=False)
            self.r.append(rb)
        self.flat = np.concatenate(self.r)
        self.seg = np.repeat(np.arange(len(self.bins)), [len(x) for x in self.r])
        self.cnt = np.bincount(self.seg, minlength=len(self.bins))

    def __call__(self, f, *theta):
        return np.bincount(self.seg, f(self.flat, *theta), len(self.bins)) / self.cnt


def _init_guess(model, r, y):
    """Log-linear initial guess on positive bins: ln(y r) = ln A - k r (A), ln(y/r) = ln B - k r (B)."""
    k = y > 0
    if k.sum() < 2:
        return (max(np.nanmax(np.abs(y)), 1e-12), 1.0)
    z = np.log(y[k] * r[k]) if model == "A" else np.log(y[k] / r[k])
    s, c = np.polyfit(r[k], z, 1)
    return (np.exp(c), max(-s, 1e-3))


def fit_one(model, avg, rbar, y, sig):
    f = MODELS[model]
    p0 = _init_guess(model, rbar, y)
    popt, pcov = curve_fit(lambda _x, a, k: avg(f, a, k), rbar, y, p0=p0, sigma=sig, absolute_sigma=True,
                           bounds=([-np.inf, 1e-6], [np.inf, 50.0]), maxfev=20000)
    pred = avg(f, *popt)
    return popt, np.sqrt(np.diag(pcov)), pred


def metrics(y, pred, sig, k):
    res = y - pred
    n = len(y)
    chi2 = float(np.sum((res / sig) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return dict(n_bins=n, n_params=k, chi2=chi2, chi2_per_dof=chi2 / max(n - k, 1),
                weighted_rmse=float(np.sqrt(np.mean((res / sig) ** 2))), rmse=float(np.sqrt(np.mean(res ** 2))),
                mae=float(np.mean(np.abs(res))), r2=1 - float(np.sum(res ** 2)) / ss_tot if ss_tot > 0 else np.nan,
                aic=chi2 + 2 * k, bic=chi2 + k * np.log(n))


class _RSpline:
    """gamma(r) = u(r)/r with u a weighted cubic smoothing spline of u = r*gamma. The variable change is regular for
    both candidate laws (A: u = A e^{-kr}, B: u = B r^2 e^{-kr}) and for generic finite-range friction, so the
    baseline does not favour either model; it removes the 1/r stiffness that defeats a spline in gamma itself."""

    def __init__(self, r, y, sig):
        self.s = UnivariateSpline(r, y * r, w=1 / (sig * r), k=3, s=len(y))

    def __call__(self, r):
        return self.s(r) / r

    def n_coeffs(self):
        return len(self.s.get_coeffs())


def spline_fit(rbar, y, sig):
    """Nonparametric baseline evaluated at the mean pair distance of each bin (no within-bin averaging, so a small
    curvature bias remains where gamma varies strongly inside a bin). Smoothing s = n (chi^2 ~ n); the effective
    parameter count (number of spline coefficients) makes its AIC/BIC indicative only."""
    s = _RSpline(rbar, y, sig)
    return s, s.n_coeffs()


def fit_all(rbar, y, sig, avg):
    out = {}
    for m in MODELS:
        popt, perr, pred = fit_one(m, avg, rbar, y, sig)
        out[m] = dict(params=popt.tolist(), errors=perr.tolist(), pred=pred.tolist()) | metrics(y, pred, sig, 2)
    spl, k = spline_fit(rbar, y, sig)
    pred = spl(rbar)
    out["spline"] = dict(pred=pred.tolist(), n_coeffs=k) | metrics(y, pred, sig, k)
    out["delta_aic_B_minus_A"] = out["B"]["aic"] - out["A"]["aic"]
    return out


def group_replicates(stats, use, n_boot=300, rng=None):
    """Replicate bin means by resampling independent groups (configurations / time blocks) with replacement."""
    rng = np.random.default_rng(7) if rng is None else rng
    gs, gc = stats["group_sum"][:, use], stats["group_cnt"][:, use]
    G = len(gs)
    Y = []
    for _ in range(n_boot):
        idx = rng.integers(0, G, G)
        c = gc[idx].sum(0)
        if np.all(c > 0):
            Y.append(gs[idx].sum(0) / c)
    return np.array(Y)


def bootstrap(Y, rbar, sig, avg):
    """Refit both models on every replicate Y (weights fixed at the full-sample SE). Returns percentile CIs and
    the fraction of replicates in which Model A has the lower chi^2."""
    pars = {m: [] for m in MODELS}
    wins = []
    for y in Y:
        chi = {}
        for m in MODELS:
            try:
                p, _, pred = fit_one(m, avg, rbar, y, sig)
            except (RuntimeError, ValueError):
                continue
            pars[m].append(p)
            chi[m] = np.sum(((y - pred) / sig) ** 2)
        if len(chi) == 2:
            wins.append(chi["A"] < chi["B"])
    ci = {m: dict(p2_5=np.percentile(np.array(v), 2.5, axis=0).tolist(),
                  p97_5=np.percentile(np.array(v), 97.5, axis=0).tolist(), n=len(v)) for m, v in pars.items() if v}
    return dict(ci=ci, frac_A_better=float(np.mean(wins)) if wins else float("nan"), n_replicates=len(Y))
