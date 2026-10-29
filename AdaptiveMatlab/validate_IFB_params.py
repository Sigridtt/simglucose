"""Choose and validate IFBTauMin (the two IFB plasma-insulin time constants)
for the simglucose virtual population.

    python AdaptiveMatlab/validate_IFB_params.py
    python AdaptiveMatlab/validate_IFB_params.py --bootstrap 500 --show
    python AdaptiveMatlab/validate_IFB_params.py --out-dir results/tau_fit

QUESTION
  adaptiveController780G uses ONE pair (tau_1, tau_2) for every patient
  (InsulinFeedbackV2.m). Which pair best describes the insulin kinetics of the
  simglucose population, can tau_1 and tau_2 be determined separately, and how
  well does the chosen pair generalise to patients it was not fitted to? The
  reference is Loutseiko et al. 2011 (appendix, p. 1349): [50 70] min, from a
  study in dogs.

DATA
  simglucose's insulin model is linear (simglucose/patient/t1dpatient.py):
      x10' = u - (ka1 + kd) x10                         subcutaneous depot 1
      x11' = kd x10 - ka2 x11                           subcutaneous depot 2
      x5'  = -(m2 + m4) x5 + m1 x9 + ka1 x10 + ka2 x11  plasma insulin
      x9'  = -(m1 + m30) x9 + m2 x5                     liver insulin
  For each of the 30 virtual patients the exact plasma-insulin (x5) response to
  a step in insulin delivery is computed on the 5-min controller grid and
  scaled so its final value is 1. These kinetics are simglucose's UVA/Padova
  parameters: the result describes the simulated population, not measured
  data for a specific insulin.

MODEL
  The IFB estimate of InsulinFeedbackV2.m (Loutseiko 2011, Eq. 3 / A13-A14):
      Ip(n) = K0 u(n-1) + K1 Ip(n-1) - K2 Ip(n-2),
      K1 = p1 + p2, K2 = p1 p2, K0 = 1 - K1 + K2, p_i = exp(-dt / tau_i),
  i.e. two first-order lags in series with unit gain. The model is symmetric
  in tau_1 and tau_2: (a, b) and (b, a) give exactly the same curve. No order is
  assumed. Pairs are only sorted when reported ("shorter" / "longer" tau) as a
  naming convention.

FIT CRITERION (one criterion everywhere)
  J(tau) = mean over the patients used of the mean squared difference between
  the patient's step response and the IFB step response over 8 h (least
  squares). Reported errors are root mean squared errors (RMSE), as a fraction
  of the final value (0.05 = 5 %). Every fit is a global grid search (both
  orders, 1-min grid from --grid-min to --grid-max) refined by Nelder-Mead, so
  it cannot stop in a local minimum.

1. IDENTIFIABILITY (ifb_tau_identifiability.png)
   a. Structural: the permutation symmetry is shown on the full error surface
      (mirror image about tau_1 = tau_2). Apart from the order, two distinct
      time constants are structurally identifiable from a noise-free curve.
   b. Practical, profile likelihood (Raue et al., Bioinformatics 2009): for a
      fixed value of one time constant, J is minimised over the other. A flat
      profile means that time constant is not identifiable. The same is done
      for the sum S = tau_1 + tau_2 (the mean absorption time of the two lags)
      and the split fraction f = shorter / S. The near-optimal set uses a
      practical tolerance (--tol, relative to the best J); it is not a
      statistical confidence interval.
   c. Statistical, patient bootstrap (Efron and Tibshirani 1993): patients are
      resampled with replacement and the pair refitted (--bootstrap times),
      giving 95 % percentile intervals for the shorter tau, the longer tau, S
      and t50 (time to half of the final plasma insulin).
   d. Local, Fisher information: correlation of log tau_1 and log tau_2 from
      J'J of the step response at the optimum (|correlation| near 1 = the
      two cannot be separated) and its condition number.
   A quantity is called practically identifiable when BOTH hold: its bootstrap
   95 % interval is narrower than +-25 % of its estimate (the population-best
   value is reproducible) AND its near-optimal profile range is narrower than
   that too (no very different value fits almost as well). The two answer
   different questions and are reported separately: a value can be precisely
   reproducible across patient resamples and still lie on a flat ridge.

2. LEAVE-ONE-OUT VALIDATION (ifb_tau_validation.png)
   For each patient, the pair is fitted on the other 29 and tested on the
   left-out patient. The chosen pair is the fit on all 30 patients; its
   expected error on a new patient is the leave-one-out error. Compared with:
       - Loutseiko [50 70] (no fitting, literature reference);
       - one pair per age group, also leave-one-out (does a group-specific
         value generalise better than one value for all?);
       - each patient's own best pair (oracle: what no single value can beat).
   Plots: learning curve (training and held-out error against the number of
   training patients, repeated random subsets), held-out error per candidate,
   training error against leave-one-out error per patient, and the residuals
   of the chosen pair over time.

Outputs below --out-dir: ifb_tau_identifiability.png, ifb_tau_validation.png,
ifb_tau_validation.csv (per patient) and ifb_tau_validation.json (summary).

This validates only the insulin kinetics (tau). The feedback gain IFBGamma is
a controller design choice and has to be tested in closed loop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.signal import lfilter

SIMGLUCOSE_ROOT = Path(__file__).resolve().parents[1]
PATIENT_FILE = SIMGLUCOSE_ROOT / "simglucose" / "params" / "vpatient_params.csv"
DEFAULT_OUT_DIR = SIMGLUCOSE_ROOT / "simglucose_results" / "ifb_tau_validationV2"
LITERATURE_TAU = np.array([50.0, 70.0])   # Loutseiko et al. 2011, appendix p. 1349 (dogs)
DT = 5.0            # controller sample time [min] (GuardianRT)
HORIZON = 480.0     # [min]
N = int(HORIZON / DT)
TIME = np.arange(N + 1) * DT
IDENT_REL_WIDTH = 0.25   # practically identifiable: 95 % interval within +-25 % of the estimate

# Group colours (validated categorical palette), plus line style and marker so
# the groups are not told apart by colour alone.
GROUPS = {
    "adult": dict(color="#2a78d6", ls="-", marker="o"),
    "adolescent": dict(color="#eb6834", ls="--", marker="s"),
    "child": dict(color="#1baf7a", ls=":", marker="^"),
}
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
PLURAL = {"adult": "adults", "adolescent": "adolescents", "child": "children"}


# ------------------------------------------------------------------ model
def plasma_step(p) -> np.ndarray:
    """Normalised plasma-insulin step response of one simglucose patient."""
    A = np.array([[-(p.ka1 + p.kd), 0.0, 0.0, 0.0],
                  [p.kd, -p.ka2, 0.0, 0.0],
                  [p.ka1, p.ka2, -(p.m2 + p.m4), p.m1],
                  [0.0, 0.0, p.m2, -(p.m1 + p.m30)]])
    B = np.array([1.0, 0.0, 0.0, 0.0])
    M = expm(np.block([[A, B[:, None]], [np.zeros((1, 5))]]) * DT)   # zero-order hold
    Ad, Bd = M[:4, :4], M[:4, 4]
    x, y = np.zeros(4), np.zeros(N + 1)
    for k in range(N):
        x = Ad @ x + Bd
        y[k + 1] = x[2]
    return y / -np.linalg.solve(A, B)[2]          # steady state = 1


def ifb_step(tau) -> np.ndarray:
    """Step response of the IFB estimate of InsulinFeedbackV2.m (symmetric in tau)."""
    p1, p2 = np.exp(-DT / tau[0]), np.exp(-DT / tau[1])
    k1, k2 = p1 + p2, p1 * p2
    return lfilter([0.0, 1 - k1 + k2], [1.0, -k1, k2], np.ones(N + 1))


def time_to(y, level) -> float:
    reached = np.flatnonzero(y >= level)
    return float(TIME[reached[0]]) if len(reached) else float("nan")


def ordered(tau) -> np.ndarray:
    """Naming convention only: (shorter, longer)."""
    return np.sort(np.asarray(tau, dtype=float))


# ------------------------------------------------------------------ fitting
class Fitter:
    """Least-squares fits of one shared tau pair; grid search over all pairs
    (computed once) followed by an unconstrained Nelder-Mead refinement."""

    def __init__(self, Y, grid):
        self.Y = Y                                       # patients x time
        self.grid = grid
        ii, jj = np.triu_indices(len(grid))              # unordered pairs (symmetry)
        self.pairs = np.column_stack([grid[ii], grid[jj]])
        G = np.array([ifb_step(t) for t in self.pairs])  # pairs x time
        # per pair and patient mean squared error, via expansion (memory-light)
        self.mse = (np.sum(G ** 2, axis=1)[:, None] - 2 * G @ Y.T
                    + np.sum(Y ** 2, axis=1)[None, :]) / Y.shape[1]

    def J(self, tau, idx):
        return float(np.mean(np.mean((self.Y[idx] - ifb_step(tau)) ** 2, axis=1)))

    def fit(self, idx):
        idx = np.asarray(idx)
        start = self.pairs[np.argmin(self.mse[:, idx].mean(axis=1))]
        res = minimize(lambda lt: self.J(np.exp(lt), idx), np.log(start), method="Nelder-Mead",
                       options=dict(xatol=1e-4, fatol=1e-12))
        return ordered(np.exp(res.x))

    def rmse(self, tau, idx):
        """Per-patient RMSE of one pair."""
        return np.sqrt(np.mean((self.Y[np.asarray(idx)] - ifb_step(tau)) ** 2, axis=1))


# ------------------------------------------------------------------ identifiability
def identifiability(fit, all_idx, best, n_boot, tol, seed):
    grid, pairs = fit.grid, fit.pairs
    J = fit.mse[:, all_idx].mean(axis=1)
    j_min = fit.J(best, all_idx)
    near = J <= (1 + tol) * j_min
    S = pairs.sum(axis=1)
    f = pairs[:, 0] / S                                  # shorter / sum

    # profiles (min of J over the other parameter)
    prof_tau = np.full(len(grid), np.inf)                # one tau fixed (either position)
    for k, g in enumerate(grid):
        prof_tau[k] = J[(pairs[:, 0] == g) | (pairs[:, 1] == g)].min()
    s_vals = np.unique(S)
    prof_S = np.array([J[S == s].min() for s in s_vals])
    f_bins = np.linspace(0.02, 0.5, 49)
    prof_f = np.array([J[np.abs(f - b) <= 0.005].min() if np.any(np.abs(f - b) <= 0.005) else np.nan
                       for b in f_bins])

    # patient bootstrap
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        t = fit.fit(rng.choice(all_idx, size=len(all_idx), replace=True))
        boot.append([t[0], t[1], t.sum(), time_to(ifb_step(t), 0.5)])
    boot = np.array(boot)
    est = np.array([best[0], best[1], best.sum(), time_to(ifb_step(best), 0.5)])
    lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)

    # local Fisher information in log tau
    h = 1e-3
    Jac = np.column_stack([(ifb_step(best * np.exp(h * e)) - ifb_step(best * np.exp(-h * e))) / (2 * h)
                           for e in np.eye(2)])
    F = Jac.T @ Jac
    C = np.linalg.inv(F)
    corr = float(C[0, 1] / np.sqrt(C[0, 0] * C[1, 1]))

    # verdict per quantity
    names = ["shorter tau", "longer tau", "sum S = tau_1 + tau_2", "t50"]
    thr = (1 + tol) * j_min
    profile_closed = {
        # profile of one tau: rises above the tolerance on both sides inside the grid
        "shorter tau": prof_tau[0] > thr and prof_tau[-1] > thr and np.ptp(grid[prof_tau <= thr]) < IDENT_REL_WIDTH * 2 * best[0],
        "longer tau": prof_tau[0] > thr and prof_tau[-1] > thr and np.ptp(grid[prof_tau <= thr]) < IDENT_REL_WIDTH * 2 * best[1],
        "sum S = tau_1 + tau_2": prof_S[0] > thr and prof_S[-1] > thr,
        "t50": True,
    }
    quantities = {}
    for k, name in enumerate(names):
        narrow = (hi[k] - lo[k]) <= 2 * IDENT_REL_WIDTH * est[k]
        quantities[name] = dict(estimate=float(est[k]), ci95=[float(lo[k]), float(hi[k])],
                                bootstrap_precise=bool(narrow), profile_closed=bool(profile_closed[name]),
                                identifiable=bool(narrow and profile_closed[name]))
    near_pairs = pairs[near]
    result = dict(
        best_J=j_min, tol=tol, grid=[float(grid[0]), float(grid[-1])],
        near_optimal_pairs=int(near.sum()),
        near_optimal_shorter_range=[float(near_pairs[:, 0].min()), float(near_pairs[:, 0].max())],
        near_optimal_longer_range=[float(near_pairs[:, 1].min()), float(near_pairs[:, 1].max())],
        near_optimal_sum_range=[float(S[near].min()), float(S[near].max())],
        local_correlation_log_tau=corr, fisher_condition_number=float(np.linalg.cond(F)),
        bootstrap_samples=n_boot, quantities=quantities)
    plot = dict(J=J, near=near, prof_tau=prof_tau, s_vals=s_vals, prof_S=prof_S,
                f_bins=f_bins, prof_f=prof_f, boot=boot, thr=thr)
    return result, plot


def plot_identifiability(fit, best, ident, P, out_dir):
    grid = fit.grid
    n = len(grid)
    Jfull = np.full((n, n), np.nan)
    ii, jj = np.triu_indices(n)
    Jfull[ii, jj] = P["J"]
    Jfull[jj, ii] = P["J"]                                # mirror: the model is symmetric
    view = min(grid[-1], 150.0)
    fig, ax = plt.subplots(2, 3, figsize=(17, 10), facecolor=SURFACE)
    for x in ax.flat:
        x.set_facecolor(SURFACE)
        x.spines[["top", "right"]].set_visible(False)

    # A: full error surface (both orders)
    a = ax[0, 0]
    j_min = ident["best_J"]
    lev = np.linspace(100 * np.sqrt(j_min), 100 * np.sqrt(min(np.nanmax(Jfull), 16 * j_min)), 15)
    T1, T2 = np.meshgrid(grid, grid, indexing="ij")
    cf = a.contourf(T1, T2, np.clip(100 * np.sqrt(Jfull), None, lev[-1]), levels=lev, cmap="Blues_r")
    fig.colorbar(cf, ax=a, label="RMSE of one pair for all patients (%)")
    a.contour(T1, T2, (Jfull <= P["thr"]).astype(float), levels=[0.5], colors=INK, linewidths=1.5)
    a.plot([grid[0], view], [grid[0], view], color=INK_2, lw=0.8, ls=":")
    b = P["boot"]
    a.scatter(b[:, 0], b[:, 1], s=6, color=INK_2, alpha=0.35, label="bootstrap estimates")
    a.scatter(b[:, 1], b[:, 0], s=6, color=INK_2, alpha=0.35)
    for t in (best, best[::-1]):
        a.scatter(*t, s=170, marker="*", color=INK, edgecolor=SURFACE, zorder=4)
    a.scatter([], [], s=120, marker="*", color=INK, label=f"chosen {best[0]:.1f}/{best[1]:.1f} (and mirror)")
    a.scatter(*LITERATURE_TAU, s=100, marker="X", color="#e34948", edgecolor=SURFACE, zorder=4,
              label=f"Loutseiko {LITERATURE_TAU[0]:g}/{LITERATURE_TAU[1]:g}")
    a.plot([], [], color=INK, lw=1.5, label=f"within {100 * ident['tol']:g} % of the best J")
    a.set(xlabel="tau_1 (min)", ylabel="tau_2 (min)", xlim=(grid[0], view), ylim=(grid[0], view),
          title="A. Error surface: symmetric, with a ridge")
    a.legend(loc="upper right", fontsize=8, frameon=False)

    # B: profile of one tau
    a = ax[0, 1]
    a.plot(grid, 100 * np.sqrt(P["prof_tau"]), color=INK, lw=2, label="profile of one time constant")
    a.axhline(100 * np.sqrt(P["thr"]), color=INK_2, lw=1, ls="--", label=f"tolerance ({100 * ident['tol']:g} % on J)")
    a.set(xlabel="fixed time constant (min)", ylabel="best RMSE over the other (%)", xlim=(grid[0], view),
          ylim=(0, 300 * np.sqrt(j_min)), title="B. Profile of one time constant (either): flat = not identifiable")
    a.grid(True, color=GRID, lw=0.8)
    a.legend(loc="upper right", fontsize=8.5, frameon=False)

    # C: profile of the sum
    a = ax[0, 2]
    a.plot(P["s_vals"], 100 * np.sqrt(P["prof_S"]), color=INK, lw=2, label="profile of S = tau_1 + tau_2")
    a.axhline(100 * np.sqrt(P["thr"]), color=INK_2, lw=1, ls="--", label="tolerance")
    a.axvline(best.sum(), color=INK_2, lw=1, ls=":")
    a.set(xlabel="S = tau_1 + tau_2 (min)", ylabel="best RMSE over the split (%)", xlim=(20, 300),
          ylim=(0, 300 * np.sqrt(j_min)), title="C. Profile of the sum: sharp = identifiable")
    a.grid(True, color=GRID, lw=0.8)
    a.legend(loc="upper right", fontsize=8.5, frameon=False)

    # D: profile of the split fraction
    a = ax[1, 0]
    a.plot(P["f_bins"], 100 * np.sqrt(P["prof_f"]), color=INK, lw=2, label="profile of f = shorter / S")
    a.axhline(100 * np.sqrt(P["thr"]), color=INK_2, lw=1, ls="--", label="tolerance")
    a.set(xlabel="split f = shorter / (tau_1 + tau_2)", ylabel="best RMSE over S (%)",
          ylim=(0, 300 * np.sqrt(j_min)), title="D. Profile of the split: wide = not identifiable")
    a.grid(True, color=GRID, lw=0.8)
    a.legend(loc="upper right", fontsize=8.5, frameon=False)

    # E: bootstrap distributions (relative to the estimate)
    a = ax[1, 1]
    q = ident["quantities"]
    labels = list(q)
    a.axvspan(1 - IDENT_REL_WIDTH, 1 + IDENT_REL_WIDTH, color="#cde2fb", zorder=0,
              label=f"+-{100 * IDENT_REL_WIDTH:g} % of the estimate")
    for k, lab in enumerate(labels, start=1):
        est, (l, h) = q[lab]["estimate"], q[lab]["ci95"]
        a.plot([l / est, h / est], [k, k], color=INK, lw=2.5, solid_capstyle="round")
        a.scatter(np.median(P["boot"][:, k - 1]) / est, k, s=40, color=INK, zorder=3)
    a.plot([], [], color=INK, lw=2.5, label="bootstrap 95 % interval (dot: median)")

    def verdict(v):
        if v["identifiable"]:
            return "identifiable"
        return ("precise, but profile flat: NOT identifiable" if v["bootstrap_precise"]
                else "NOT identifiable")
    a.set_yticks(range(1, len(labels) + 1),
                 [f"{lab}: {q[lab]['estimate']:.0f} [{q[lab]['ci95'][0]:.0f}-{q[lab]['ci95'][1]:.0f}]\n{verdict(q[lab])}"
                  for lab in labels], fontsize=8)
    a.set(xlabel="value / chosen estimate", ylim=(0.4, len(labels) + 0.6),
          title=f"E. Patient bootstrap ({len(P['boot'])} refits) and verdict")
    a.grid(True, axis="x", color=GRID, lw=0.8)
    a.legend(loc="lower right", fontsize=8, frameon=False)

    # F: near-optimal pairs give the same curve
    a = ax[1, 2]
    a.plot(TIME, fit.Y.mean(axis=0), color="#e34948", lw=2, label="simglucose mean (30 patients)")
    npairs = fit.pairs[P["near"]]
    order = np.argsort(npairs[:, 0])
    for k, ls in zip(np.linspace(0, len(order) - 1, 4).round().astype(int), ["-", "--", "-.", ":"]):
        t = npairs[order[k]]
        a.plot(TIME, ifb_step(t), color=INK, lw=1.2, ls=ls, label=f"tau = {t[0]:g}/{t[1]:g} (S = {t.sum():g})")
    a.set(xlabel="Time after insulin step (min)", ylabel="Plasma insulin (fraction of final value)",
          xlim=(0, HORIZON), ylim=(0, 1.05), title="F. Near-optimal pairs: same curve, same S")
    a.grid(True, color=GRID, lw=0.8)
    a.legend(loc="lower right", fontsize=8, frameon=False)

    sep = q["shorter tau"]["identifiable"] and q["longer tau"]["identifiable"]
    fig.suptitle("Identifiability of IFBTauMin: tau_1 and tau_2 are "
                 + ("separately identifiable" if sep else "NOT separately identifiable")
                 + f"; the sum S is {'identifiable' if q['sum S = tau_1 + tau_2']['identifiable'] else 'not identifiable'}"
                 f" (local correlation {ident['local_correlation_log_tau']:+.2f})", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "ifb_tau_identifiability.png", dpi=150, facecolor=SURFACE)


# ------------------------------------------------------------------ validation
def hbox(ax, data, **kw):
    """Horizontal box plot for old and new Matplotlib versions."""
    try:
        return ax.boxplot(data, orientation="horizontal", **kw)
    except TypeError:
        return ax.boxplot(data, vert=False, **kw)


def learning_curve(fit, all_idx, sizes, repeats, seed):
    rng = np.random.default_rng(seed)
    out = []
    for n in sizes:
        tr_e, va_e = [], []
        exact = n == len(all_idx) - 1          # leave-one-out: every patient held out once
        subsets = [np.delete(all_idx, i) for i in all_idx] if exact else \
            [rng.choice(all_idx, size=n, replace=False) for _ in range(repeats)]
        for tr in subsets:
            va = np.setdiff1d(all_idx, tr)
            t = fit.fit(tr)
            tr_e.append(fit.rmse(t, tr).mean())
            va_e.append(fit.rmse(t, va).mean())
        out.append(dict(n=n, train=np.array(tr_e), val=np.array(va_e)))
    return out


def plot_validation(fit, res, best, lc, cand, out_dir):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10), facecolor=SURFACE)
    for x in ax.flat:
        x.set_facecolor(SURFACE)
        x.spines[["top", "right"]].set_visible(False)
        x.grid(True, color=GRID, lw=0.8)

    # A: learning curve
    a = ax[0, 0]
    n = [d["n"] for d in lc]
    for key, color, ls, lab in (("train", "#2a78d6", "-", "training patients"),
                                ("val", "#eb6834", "--", "held-out patients")):
        m = [100 * d[key].mean() for d in lc]
        q1 = [100 * np.percentile(d[key], 25) for d in lc]
        q3 = [100 * np.percentile(d[key], 75) for d in lc]
        a.plot(n, m, color=color, ls=ls, lw=2, marker="o", ms=4, label=f"{lab} (mean, IQR band)")
        a.fill_between(n, q1, q3, color=color, alpha=0.15, lw=0)
    a.axhline(100 * res.error_literature.mean(), color="#e34948", lw=1, ls=":",
              label=f"Loutseiko {LITERATURE_TAU[0]:g}/{LITERATURE_TAU[1]:g}, all patients")
    a.set(xlabel="number of training patients", ylabel="mean RMSE (%)",
          title=f"A. Learning curve (random subsets; {n[-1]} = exact leave-one-out)")
    a.legend(loc="upper right", fontsize=8.5, frameon=False)

    # B: held-out error per candidate
    a = ax[0, 1]
    labels = list(cand)
    data = [100 * cand[k] for k in labels]
    hbox(a, data, widths=0.5, showfliers=False, medianprops=dict(color=INK),
              boxprops=dict(color=INK_2), whiskerprops=dict(color=INK_2), capprops=dict(color=INK_2))
    rng = np.random.default_rng(0)
    for k, d in enumerate(data, start=1):
        cols = [GROUPS[g]["color"] for g in res.group]
        a.scatter(d, k + rng.uniform(-0.15, 0.15, len(d)), s=14, c=cols, edgecolor=SURFACE, lw=0.5, zorder=3)
    a.set_yticks(range(1, len(labels) + 1), [f"{k}\nmean {np.mean(d):.2f} %" for k, d in zip(labels, data)],
                 fontsize=8.5)
    for grp, g in GROUPS.items():
        a.scatter([], [], s=20, color=g["color"], label=PLURAL[grp])
    a.set(xlabel="RMSE on the patient (%)", title="B. Error per patient for each candidate")
    a.legend(loc="lower right", fontsize=8, frameon=False)

    # C: training vs leave-one-out error per patient
    a = ax[1, 0]
    for grp, g in GROUPS.items():
        r = res[res.group == grp]
        a.scatter(100 * r.error_all_fit, 100 * r.error_loo, s=40, color=g["color"], marker=g["marker"],
                  edgecolor=SURFACE, lw=1, label=PLURAL[grp], zorder=3)
    lim = 100 * max(res.error_loo.max(), res.error_all_fit.max()) * 1.08
    a.plot([0, lim], [0, lim], color=INK_2, lw=1, ls="--", label="held-out = training")
    for r in res.nlargest(3, "error_loo").itertuples():
        a.annotate(r.patient, (100 * r.error_all_fit, 100 * r.error_loo), xytext=(5, -10),
                   textcoords="offset points", fontsize=8, color=INK_2)
    a.set(xlim=(0, lim), ylim=(0, lim), xlabel="training RMSE, pair fitted on all 30 (%)",
          ylabel="leave-one-out RMSE (%)", title="C. Training vs held-out error per patient")
    a.legend(loc="upper left", fontsize=8.5, frameon=False)

    # D: residuals over time of the chosen pair
    a = ax[1, 1]
    R = ifb_step(best)[None, :] - fit.Y
    a.axhline(0, color=INK_2, lw=1)
    a.fill_between(TIME, 100 * np.percentile(R, 10, axis=0), 100 * np.percentile(R, 90, axis=0),
                   color="#2a78d6", alpha=0.2, lw=0, label="10-90 % of patients")
    a.plot(TIME, 100 * R.mean(axis=0), color="#2a78d6", lw=2, label=f"mean residual, chosen {best[0]:.1f}/{best[1]:.1f}")
    a.plot(TIME, 100 * (ifb_step(LITERATURE_TAU)[None, :] - fit.Y).mean(axis=0), color="#e34948", lw=1.5,
           ls="--", label=f"mean residual, Loutseiko {LITERATURE_TAU[0]:g}/{LITERATURE_TAU[1]:g}")
    a.set(xlabel="Time after insulin step (min)", ylabel="IFB estimate - simglucose (% of final value)",
          xlim=(0, HORIZON), title="D. Residuals over time (positive = IFB too fast)")
    a.legend(loc="lower right", fontsize=8.5, frameon=False)

    fig.suptitle(f"Leave-one-out validation of IFBTauMin = {best[0]:.1f}/{best[1]:.1f} min "
                 f"(leave-one-out RMSE {100 * res.error_loo.mean():.2f} %, Loutseiko "
                 f"{100 * res.error_literature.mean():.2f} %, each patient's own {100 * res.error_oracle.mean():.2f} %)",
                 fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "ifb_tau_validation.png", dpi=150, facecolor=SURFACE)


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"directory for plots and tables (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--grid-min", type=float, default=5.0, help="smallest tau of the grid [min]")
    ap.add_argument("--grid-max", type=float, default=200.0, help="largest tau of the grid [min]")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="practical tolerance for near-optimal pairs and profiles, relative to the best J")
    ap.add_argument("--bootstrap", type=int, default=200, help="number of patient bootstrap refits")
    ap.add_argument("--repeats", type=int, default=30, help="random subsets per learning-curve point")
    ap.add_argument("--seed", type=int, default=0, help="seed for bootstrap and learning curve")
    ap.add_argument("--show", action="store_true", help="open the figure windows")
    a = ap.parse_args(argv)

    vp = pd.read_csv(PATIENT_FILE).set_index("Name")
    names = list(vp.index)
    groups = np.array([n.split("#")[0] for n in names])
    Y = np.array([plasma_step(p) for _, p in vp.iterrows()])
    fit = Fitter(Y, np.arange(a.grid_min, a.grid_max + 1.0, 1.0))
    all_idx = np.arange(len(names))
    print("Fitting on all patients, leave-one-out and per group ...", flush=True)

    best = fit.fit(all_idx)                                         # the chosen pair
    loo_tau = np.array([fit.fit(np.delete(all_idx, i)) for i in all_idx])
    loo_err = np.array([fit.rmse(loo_tau[i], [i])[0] for i in all_idx])
    oracle_tau = np.array([fit.fit([i]) for i in all_idx])
    oracle_err = np.array([fit.rmse(oracle_tau[i], [i])[0] for i in all_idx])
    group_loo_err = np.zeros(len(names))
    group_tau = {}
    for g in GROUPS:
        gi = all_idx[groups == g]
        group_tau[g] = fit.fit(gi)
        for i in gi:
            group_loo_err[i] = fit.rmse(fit.fit(gi[gi != i]), [i])[0]

    res = pd.DataFrame(dict(
        patient=names, group=groups,
        t50_min=[time_to(y, 0.5) for y in Y], t90_min=[time_to(y, 0.9) for y in Y],
        tau_short_own=oracle_tau[:, 0], tau_long_own=oracle_tau[:, 1], error_oracle=oracle_err,
        tau_short_loo=loo_tau[:, 0], tau_long_loo=loo_tau[:, 1], error_loo=loo_err,
        error_all_fit=fit.rmse(best, all_idx), error_group_loo=group_loo_err,
        error_literature=fit.rmse(LITERATURE_TAU, all_idx)))

    print(f"Identifiability (grid, profiles, {a.bootstrap} bootstrap refits) ...", flush=True)
    ident, P = identifiability(fit, all_idx, best, a.bootstrap, a.tol, a.seed)
    print("Learning curve ...", flush=True)
    sizes = [2, 3, 5, 8, 12, 16, 20, 24, 29]
    lc = learning_curve(fit, all_idx, sizes, a.repeats, a.seed)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out_dir / "ifb_tau_validation.csv", index=False)
    cand = {"chosen pair (leave-one-out)": res.error_loo.to_numpy(),
            "pair per age group (leave-one-out)": res.error_group_loo.to_numpy(),
            f"Loutseiko {LITERATURE_TAU[0]:g}/{LITERATURE_TAU[1]:g} (no fit)": res.error_literature.to_numpy(),
            "each patient's own pair (oracle)": res.error_oracle.to_numpy()}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    plot_identifiability(fit, best, ident, P, a.out_dir)
    plot_validation(fit, res, best, lc, cand, a.out_dir)

    summary = dict(
        chosen_tau_min=best.tolist(), chosen_sum_min=float(best.sum()),
        chosen_t50_min=time_to(ifb_step(best), 0.5), chosen_t90_min=time_to(ifb_step(best), 0.9),
        error_all_fit=float(res.error_all_fit.mean()), error_leave_one_out=float(res.error_loo.mean()),
        leave_one_out_tau_short_range=[float(res.tau_short_loo.min()), float(res.tau_short_loo.max())],
        leave_one_out_tau_long_range=[float(res.tau_long_loo.min()), float(res.tau_long_loo.max())],
        literature=dict(tau_min=LITERATURE_TAU.tolist(), source="Loutseiko et al. 2011, appendix p. 1349 (dogs)",
                        sum_min=float(LITERATURE_TAU.sum()), t50_min=time_to(ifb_step(LITERATURE_TAU), 0.5),
                        error=float(res.error_literature.mean())),
        error_group_leave_one_out=float(res.error_group_loo.mean()),
        group_tau_min={g: t.tolist() for g, t in group_tau.items()},
        error_oracle=float(res.error_oracle.mean()),
        simglucose_t50_min=dict(median=float(res.t50_min.median()), min=float(res.t50_min.min()),
                                max=float(res.t50_min.max())),
        learning_curve=[dict(n_train=d["n"], train=float(d["train"].mean()), held_out=float(d["val"].mean()))
                        for d in lc],
        identifiability=ident)
    (a.out_dir / "ifb_tau_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- text summary
    q = ident["quantities"]
    print(f"\nChosen IFBTauMin (fit on all 30): {best[0]:.1f}/{best[1]:.1f} min, S = {best.sum():.1f} min, "
          f"t50 {summary['chosen_t50_min']:.0f}, t90 {summary['chosen_t90_min']:.0f} min")
    print(f"  leave-one-out RMSE {100 * res.error_loo.mean():.2f} % | training {100 * res.error_all_fit.mean():.2f} % | "
          f"Loutseiko {100 * res.error_literature.mean():.2f} % | per age group (LOO) {100 * res.error_group_loo.mean():.2f} % | "
          f"oracle {100 * res.error_oracle.mean():.2f} %")
    print(f"  leave-one-out estimates: shorter {res.tau_short_loo.min():.1f}-{res.tau_short_loo.max():.1f}, "
          f"longer {res.tau_long_loo.min():.1f}-{res.tau_long_loo.max():.1f} min")
    print("Identifiability (bootstrap 95 % interval):")
    for k, v in q.items():
        print(f"  {k:24s} {v['estimate']:6.1f} [{v['ci95'][0]:6.1f}-{v['ci95'][1]:6.1f}]  "
              f"bootstrap {'precise' if v['bootstrap_precise'] else 'wide'}, "
              f"profile {'closed' if v['profile_closed'] else 'flat'} -> "
              f"{'identifiable' if v['identifiable'] else 'NOT identifiable'}")
    print(f"  near-optimal pairs (within {100 * a.tol:g} % of best J): shorter "
          f"{ident['near_optimal_shorter_range'][0]:g}-{ident['near_optimal_shorter_range'][1]:g}, longer "
          f"{ident['near_optimal_longer_range'][0]:g}-{ident['near_optimal_longer_range'][1]:g}, sum "
          f"{ident['near_optimal_sum_range'][0]:g}-{ident['near_optimal_sum_range'][1]:g} min")
    print(f"  local correlation (log tau) {ident['local_correlation_log_tau']:+.3f}, "
          f"Fisher condition number {ident['fisher_condition_number']:.0f}")
    print(f"Saved plots and tables to {a.out_dir}")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
