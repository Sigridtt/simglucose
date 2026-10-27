"""Fit and validate IFB plasma-insulin time constants against simglucose kinetics.

    python AdaptiveMatlab/validate_IFB_params.py
    python AdaptiveMatlab/validate_IFB_params.py --out-dir results/tau_fit
    python AdaptiveMatlab/validate_IFB_params.py --split-seed 7 --show

What it does
  1. simglucose's insulin model is linear (simglucose/patient/t1dpatient.py):
       x10' = u - (ka1 + kd) x10            subcutaneous depot 1
       x11' = kd x10 - ka2 x11              subcutaneous depot 2
       x5'  = -(m2 + m4) x5 + m1 x9 + ka1 x10 + ka2 x11    plasma insulin
       x9'  = -(m1 + m30) x9 + m2 x5         liver insulin
     For every virtual patient it computes the plasma-insulin (x5) response to a
     step in insulin delivery, scaled so the final value is 1 (exact, on the
     5-min controller grid).
  2. It computes the same step response for the IFB estimate of
     adaptiveController780G.m:
       Ip(n) = (1 - K1 + K2) u(n-1) + K1 Ip(n-1) - K2 Ip(n-2),
       K1 = a1 + a2, K2 = a1 a2, a_i = exp(-dt / tau_i)
  3. It fits one pooled tau_1, tau_2 pair using 8 patients from each cohort
     (adult, adolescent, child), then evaluates it on the remaining patients.

The split is random but reproducible. The error is the RMS difference between
the two step responses over 8 h, as a fraction of the final value (0.05 = 5 %).
All plots and tables are written below ``--out-dir``.

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

SIMGLUCOSE_ROOT = Path(__file__).resolve().parents[1]
PATIENT_FILE = SIMGLUCOSE_ROOT / "simglucose" / "params" / "vpatient_params.csv"
DEFAULT_OUT_DIR = SIMGLUCOSE_ROOT / "simglucose_results" / "ifb_tau_validation"
FIT_PER_GROUP = 8
DT = 5.0            # controller sample time [min] (GuardianRT)
HORIZON = 480.0     # [min]
N = int(HORIZON / DT)
TIME = np.arange(N + 1) * DT

# Group colours (validated categorical palette), plus line style and marker so
# the groups are not told apart by colour alone.
GROUPS = {
    "adult": dict(color="#2a78d6", ls="-", marker="o"),
    "adolescent": dict(color="#eb6834", ls="--", marker="s"),
    "child": dict(color="#1baf7a", ls=":", marker="^"),
}
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
PLURAL = {"adult": "adults", "adolescent": "adolescents", "child": "children"}


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
    """Step response of the IFB plasma-insulin estimate in adaptiveController780G."""
    a1, a2 = np.exp(-DT / tau[0]), np.exp(-DT / tau[1])
    k1, k2 = a1 + a2, a1 * a2
    ip = np.zeros(N + 1)
    for k in range(1, N + 1):
        ip[k] = (1 - k1 + k2) + k1 * ip[k - 1] - k2 * (ip[k - 2] if k >= 2 else 0.0)
    return ip


def rms(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def time_to(y, level) -> float:
    reached = np.flatnonzero(y >= level)
    return float(TIME[reached[0]]) if len(reached) else float("nan")


def fit_tau(curves, initial_tau) -> np.ndarray:
    """Fit one shared, ordered tau pair to a collection of step responses."""
    names = list(curves)

    def objective(log_tau):
        tau = np.sort(np.exp(log_tau))
        return float(np.mean([
            np.mean((ifb_step(tau) - curves[name]) ** 2)
            for name in names
        ]))

    fit = minimize(objective, np.log(initial_tau), method="Nelder-Mead")
    if not fit.success:
        raise RuntimeError(f"Tau fit failed: {fit.message}")
    return np.sort(np.exp(fit.x))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"directory for plots and tables (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="seed used to select the fit patients within each cohort")
    ap.add_argument("--show", action="store_true", help="open the figure window")
    a = ap.parse_args(argv)

    vp = pd.read_csv(PATIENT_FILE).set_index("Name")
    rng = np.random.default_rng(a.split_seed)
    curves = {}
    for name, p in vp.iterrows():
        curves[name] = plasma_step(p)
    split = {}
    for group in GROUPS:
        names = sorted(name for name in curves if name.startswith(f"{group}#"))
        if len(names) <= FIT_PER_GROUP:
            raise ValueError(f"{group} cohort has {len(names)} patients; need more than {FIT_PER_GROUP}")
        fit_names = set(rng.choice(names, size=FIT_PER_GROUP, replace=False))
        split.update({name: "fit" if name in fit_names else "validation" for name in names})

    fit_curves = {name: curves[name] for name, part in split.items() if part == "fit"}
    tau = fit_tau(fit_curves, np.array([50.0, 70.0]))
    patient_tau = {
        name: fit_tau({name: curves[name]}, tau)
        for name in curves
    }
    rows = []
    for name, y in curves.items():
        individual_tau = patient_tau[name]
        rows.append(dict(patient=name, group=name.split("#")[0], split=split[name],
                         t50_min=time_to(y, 0.5), t90_min=time_to(y, 0.9),
                         tau1_patient_fit=individual_tau[0],
                         tau2_patient_fit=individual_tau[1],
                         tau1_pooled_fit=tau[0], tau2_pooled_fit=tau[1],
                         error=rms(ifb_step(tau), y)))
    res = pd.DataFrame(rows)
    ifb = ifb_step(tau)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out_dir / "ifb_tau_validation.csv", index=False)
    summary = {
        "fit_per_group": FIT_PER_GROUP,
        "split_seed": a.split_seed,
        "tau_min": tau.tolist(),
        "fit_patients": sorted(name for name, part in split.items() if part == "fit"),
        "validation_patients": sorted(name for name, part in split.items() if part == "validation"),
        "fit_rms": float(res.loc[res.split == "fit", "error"].mean()),
        "validation_rms": float(res.loc[res.split == "validation", "error"].mean()),
    }
    (a.out_dir / "ifb_tau_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- figure
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), facecolor=SURFACE,
                           gridspec_kw=dict(width_ratios=[1.3, 1, 1.2]))
    for x in ax:
        x.set_facecolor(SURFACE)
        x.grid(True, color=GRID, linewidth=0.8)
        x.spines[["top", "right"]].set_visible(False)

    # A: step responses
    for name, y in curves.items():
        g = GROUPS[name.split("#")[0]]
        ax[0].plot(TIME, y, color=g["color"], ls=":", lw=0.8, alpha=0.7)
    fit_mean = np.mean([curves[name] for name, part in split.items() if part == "fit"], axis=0)
    validation_mean = np.mean(
        [curves[name] for name, part in split.items() if part == "validation"], axis=0
    )
    #ax[0].plot(TIME, ifb, color=INK, lw=2.5,
              # label=f"pooled fit, tau = {tau[0]:.1f}/{tau[1]:.1f} min")
    ax[0].plot(TIME, fit_mean, color="#8e44ad", lw=1.5, ls="-",
               label="mean of fit patients")
    ax[0].plot(TIME, validation_mean, color="#d62728", lw=1.5, ls="-",
               label="mean of validation patients")
    for grp, g in GROUPS.items():
        ax[0].plot([], [], color=g["color"], ls=":", lw=1.5,
                   label=f"simglucose {PLURAL[grp]} (individual)")
    ax[0].set(xlabel="Time after insulin step (min)", ylabel="Plasma insulin (fraction of final value)",
              title="A. Plasma-insulin step response", xlim=(0, HORIZON), ylim=(0, 1.05))
    ax[0].legend(loc="lower right", fontsize=8.5, frameon=False)

    # B: fitted time constants
    for grp, g in GROUPS.items():
        r = res[res.group == grp]
        for part, marker in (("fit", "o"), ("validation", "s")):
            q = r[r.split == part]
            ax[1].scatter(q.tau1_patient_fit, q.tau2_patient_fit, s=45,
                          color=g["color"], marker=marker,
                          edgecolor=SURFACE, linewidth=1,
                          label=f"{PLURAL[grp]} ({part}) individual fit", zorder=3)
    ax[1].scatter(*tau, s=180, marker="*", color=INK, zorder=4,
                  label=f"pooled fit {tau[0]:.1f}/{tau[1]:.1f}")
    ax[1].set(xlabel="Best-fit tau_1 (min)", ylabel="Best-fit tau_2 (min)",
              title="B. Pooled fit and patient groups")
    ax[1].legend(loc="upper left", fontsize=8.5, frameon=False)

    # C: error with the checked tau, per patient
    r = res.sort_values(["split", "error"])
    ypos = np.arange(len(r))
    for i, (_, row) in enumerate(r.iterrows()):
        g = GROUPS[row.group]
        ax[2].barh(i, 100 * row.error, height=0.75, color=g["color"],
                   edgecolor=SURFACE, linewidth=1,
                   alpha=0.7 if row.split == "fit" else 1)
    ax[2].set_yticks(ypos, r.patient, fontsize=7.5)
    validation_error = res.loc[res.split == "validation", "error"].mean()
    ax[2].axvline(100 * validation_error, color=INK, lw=1, ls="--",
                  label=f"validation mean ({100 * validation_error:.1f} %)")
    ax[2].legend(loc="lower right", fontsize=8.5, frameon=False)
    ax[2].set(xlabel=f"RMS error with pooled tau = {tau[0]:.1f}/{tau[1]:.1f} (% of final value)",
              title="C. Fit and validation error per patient")
    ax[2].grid(axis="y", visible=False)

    fig.suptitle(f"IFB plasma-insulin time constants vs simglucose insulin kinetics "
                 f"(fit {100 * summary['fit_rms']:.1f} %, "
                 f"validation {100 * summary['validation_rms']:.1f} %)", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(a.out_dir / "ifb_tau_validation.png", dpi=150, facecolor=SURFACE)

    # ------------------------------------------------------------- text summary
    pd.set_option("display.width", 160)
    print(res.round(3).to_string(index=False))
    print(f"\nPooled tau = {tau[0]:.1f}/{tau[1]:.1f} min: "
          f"t50 {time_to(ifb, 0.5):.0f} min, t90 {time_to(ifb, 0.9):.0f} min")
    print(f"simglucose: t50 {res.t50_min.median():.0f} min (range {res.t50_min.min():.0f}-{res.t50_min.max():.0f}), "
          f"t90 {res.t90_min.median():.0f} min (range {res.t90_min.min():.0f}-{res.t90_min.max():.0f})")
    print(f"Fit RMS = {100 * summary['fit_rms']:.2f} %, "
          f"validation RMS = {100 * summary['validation_rms']:.2f} %")
    print(f"Saved plots and tables to {a.out_dir}")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
