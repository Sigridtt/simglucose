"""
test_autosens_sensitivity_shift.py

Tests the two-layer adaptive controller (autotune + autosens) by injecting
a known mid-run sensitivity change via T1DPatient._params.p2u.

Structure:
    Day 1 (hours  0-24): Normal sensitivity — warmup + first autotune run
    Day 2 (hours 24-48): p2u * 1.5 — exercise / increased sensitivity
    Day 3 (hours 48-72): p2u restored — recovery, autosens should return to baseline

Expected autosens behaviour:
    Day 1: ratio settles at patient baseline (~0.88 for adult#001)
    Day 2: ratio rises toward autosens_max (1.2) — patient more sensitive
    Day 3: ratio returns toward Day 1 baseline — sensitivity restored

Expected autotune behaviour:
    Day 1→2: ISF corrected toward true patient ISF
    Day 2→3: ISF stable (autotune uses ISF-category points, less affected by p2u)

What this proves:
    - Autosens detects transient sensitivity changes independently of autotune
    - Autotune baseline is preserved across the sensitivity shift
    - The two layers operate independently and correctly on their own timescales
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt

from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.sim_engine import SimObj, sim
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario

from simglucose.controller.adaptive_loop_ctrller import AdaptiveLoopController
from simglucose.controller.loop_ctrller import LoopController
from loop_to_python_adaptive.autosens_isf import AutosensConfig

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)   # suppress simglucose noise

# ── Constants ───────────��─────────────────────────────────────────────────────
START_TIME   = datetime(2018, 1, 1, 8, 0, 0)
PATIENT_NAME = "adult#001"
DURATION     = timedelta(days=3)
SAVE_BASE    = os.path.join(
    os.path.dirname(__file__), "examples", "results", "test_autosens_shift"
)

# 3 days of meals (hour_offset, grams)
MEALS = []
for day in range(3):
    offset = day * 24
    MEALS.append((offset + 1,  45))   # breakfast
    MEALS.append((offset + 5,  70))   # lunch
    MEALS.append((offset + 11, 60))   # dinner


# ── Sensitivity-shift SimObj ──────────────────────────────────────────────────

class SensitivityShiftSimObj(SimObj):
    """
    SimObj subclass that modifies T1DPatient._params.p2u at specified hours,
    simulating exercise (multiplier > 1.0) or sickness (multiplier < 1.0).

    shifts: list of (hour_from_start, p2u_multiplier) tuples, e.g.
            [(24, 1.5), (48, 1.0)]
    """

    def __init__(self, *args, patient_ref, shifts, **kwargs):
        super().__init__(*args, **kwargs)
        self.patient_ref   = patient_ref
        self.shifts        = sorted(shifts, key=lambda x: x[0])
        self._original_p2u = float(patient_ref._params.p2u)
        self._shifts_done  = set()

    def simulate(self):
        # Do NOT call self.controller.reset() — preserve adaptive state
        obs, reward, done, info = self.env.reset()
        tic = time.time()

        while self.env.time < self.env.scenario.start_time + self.sim_time:
            if self.animate:
                self.env.render()

            # Elapsed hours from simulation start
            current_hour = (
                self.env.time - self.env.scenario.start_time
            ).total_seconds() / 3600

            # Fire any pending sensitivity shifts
            for (shift_hour, p2u_mult) in self.shifts:
                if shift_hour not in self._shifts_done and current_hour >= shift_hour:
                    new_p2u = self._original_p2u * p2u_mult
                    self.patient_ref._params.p2u = new_p2u
                    self._shifts_done.add(shift_hour)
                    label = (
                        'exercise (more sensitive)' if p2u_mult > 1.0
                        else 'restored' if p2u_mult == 1.0
                        else 'sickness (more resistant)'
                    )
                    print(
                        f"\n  [SensShift] t={current_hour:.1f}h — "
                        f"p2u {self._original_p2u:.4f} → {new_p2u:.4f}  ({label})"
                    )

            action = self.controller.policy(obs, reward, done, **info)
            obs, reward, done, info = self.env.step(action)

        toc = time.time()
        print(f"  Simulation took {toc - tic:.1f}s")


# ── Build controller ──────────────────────────────────────────────────────────

def make_controller(controller):
    if controller == 'adaptive':
        return AdaptiveLoopController(
            target=110,
            recommendation_type='automaticBolus',
            use_tdd_settings=True,
            insulin_type='novolog',
            warmup_days=1,
            adaptation_interval_hours=24,
            n_autotune_iterations=1,
            autosens_cfg=AutosensConfig(
                autosens_max=1.2,
                autosens_min=0.7,
                min_points=10,
                deviation_threshold=6.0,
            ),

        )
    else:
        return LoopController(
            target=110,
            recommendation_type='automaticBolus',
            use_tdd_settings=True,
            insulin_type='novolog',
        )

    



# ── Run one simulation condition ──────────────────────────────────────────────

def run_condition(label, shifts, save_sub, controller):
    save_path = os.path.join(SAVE_BASE, save_sub)
    os.makedirs(save_path, exist_ok=True)

    ctrl    = make_controller(controller)
    patient = T1DPatient.withName(PATIENT_NAME)
    sensor  = CGMSensor.withName("GuardianRT", seed=1)
    pump    = InsulinPump.withName("Insulet")
    env     = T1DSimEnv(patient, sensor, pump,
                        CustomScenario(start_time=START_TIME, scenario=MEALS))

    sim_obj = SensitivityShiftSimObj(
        env, ctrl, DURATION,
        animate=False,
        path=save_path,
        patient_ref=patient,
        shifts=shifts,
    )

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print("Simulation starts ...")
    sim_obj.simulate()
    sim_obj.save_results()
    print("Simulation Completed!")

    log = ctrl.get_autosens_log(PATIENT_NAME)
    isf = ctrl.get_isf_history(PATIENT_NAME)
    return pd.DataFrame(log), isf, ctrl


# ── Analysis helpers ──────────────────────────────────────────────────────────

def summarise_by_day(df, label):
    """Print per-day autosens ratio statistics."""
    print(f"\n  {label} — ratio by day:")
    print(f"  {'Day':<6} {'Hours':<12} {'Mean ratio':>11} {'Std':>7} {'Min':>7} {'Max':>7} {'n':>5}")
    print(f"  {'-'*54}")
    for day in range(3):
        h_start = day * 24
        h_end   = (day + 1) * 24
        mask    = (df['datetime'] >= START_TIME + timedelta(hours=h_start)) & \
                  (df['datetime'] <  START_TIME + timedelta(hours=h_end))
        sub = df.loc[mask, 'ratio'].dropna()
        if len(sub) == 0:
            print(f"  Day {day+1:<3} {f'{h_start}-{h_end}h':<12} {'no data':>11}")
            continue
        print(
            f"  Day {day+1:<3} {f'{h_start}-{h_end}h':<12} "
            f"{sub.mean():>11.4f} {sub.std():>7.4f} "
            f"{sub.min():>7.4f} {sub.max():>7.4f} {len(sub):>5}"
        )


def summarise_isf_history(isf_history, label):
    print(f"\n  {label} — autotune ISF history:")
    if not isf_history:
        print("    No autotune events recorded.")
        return
    for event in isf_history:
        print(
            f"    {event['datetime']}  "
            f"ISF {event['old_isf']:.3f} → {event['new_isf']:.3f} mg/dL/U  "
            f"(Δ {event['new_isf'] - event['old_isf']:+.3f})"
        )


def compute_bg_metrics(sim_obj_path, label):
    """Load saved BG trace and compute TIR/TBR/TAR."""
    csv_path = os.path.join(sim_obj_path, f"{PATIENT_NAME}.csv")
    if not os.path.exists(csv_path):
        print(f"  {label}: BG CSV not found at {csv_path}")
        return
    df  = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    bg  = df['BG'].dropna() if 'BG' in df.columns else df.iloc[:, 0].dropna()
    tir = ((bg >= 70) & (bg <= 180)).mean() * 100
    tbr = (bg < 70).mean() * 100
    tar = (bg > 180).mean() * 100
    print(f"\n  {label} — BG outcomes:")
    print(f"    Mean BG     : {bg.mean():.1f} mg/dL")
    print(f"    Std BG      : {bg.std():.1f} mg/dL")
    print(f"    TIR 70-180  : {tir:.1f}%")
    print(f"    TBR <70     : {tbr:.1f}%")
    print(f"    TAR >180    : {tar:.1f}%")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(df_baseline, df_exercise, df_sickness, save_path, controller_label):
    os.makedirs(save_path, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    datasets = [
        (df_baseline, 'Baseline (no shift)',   'steelblue'),
        (df_exercise, 'Exercise (p2u ×1.5)',   'darkorange'),
        (df_sickness, 'Sickness (p2u ×0.7)',   'crimson'),
    ]

    # ── Panel 1: autosens ratio over time ────────────────────────────────────
    ax = axes[0]
    for df, lbl, col in datasets:
        if not df.empty:
            ax.plot(df['datetime'], df['ratio'], label=lbl, color=col, linewidth=1.2)
    # Mark sensitivity shift boundaries
    for h, ls, lbl in [(24, '--', 'shift starts'), (48, ':', 'shift ends')]:
        t = START_TIME + timedelta(hours=h)
        ax.axvline(t, color='gray', linestyle=ls, linewidth=0.8, label=lbl)
    ax.axhline(1.0, color='black', linestyle='-', linewidth=0.5, alpha=0.4)
    ax.set_ylabel('Autosens ratio')
    ax.set_title(f'Autosens ratio over 3-day simulation — {controller_label}')
    ax.legend(fontsize=8)
    ax.set_ylim(0.65, 1.25)

    # ── Panel 2: effective ISF ────────────────────────────────────────────────
    ax = axes[1]
    for df, lbl, col in datasets:
        if not df.empty and 'effective_isf' in df.columns:
            ax.plot(df['datetime'], df['effective_isf'], label=lbl, color=col, linewidth=1.2)
    ax.set_ylabel('Effective ISF (mg/dL/U)')
    ax.set_title(f'Effective ISF = autotune_ISF / autosens_ratio — {controller_label}')
    ax.legend(fontsize=8)

    # ── Panel 3: deviation ───────────────────────────────────────────────────
    ax = axes[2]
    for df, lbl, col in datasets:
        if not df.empty and 'deviation' in df.columns:
            ax.plot(df['datetime'], df['deviation'], label=lbl, color=col,
                    linewidth=0.8, alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.4)
    ax.set_ylabel('Deviation (mg/dL/5min)')
    ax.set_title(f'Deviation = avgDelta − BGI  (positive = more sensitive than predicted) — {controller_label}')
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(save_path, 'autosens_shift_results.png')
    plt.savefig(out, dpi=150)
    print(f"\n  Plot saved to: {out}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────���──

if __name__ == '__main__':

    # Condition 1: no sensitivity shift — establishes patient baseline ratio
    df_baseline, isf_baseline, _ = run_condition(
        label    = "BASELINE — no sensitivity shift",
        shifts   = [],
        save_sub = "baseline",
        controller = 'adaptive',
    )

    # Condition 2: exercise at hour 24, restored at hour 48
    # p2u * 1.5 → patient absorbs glucose faster → BG falls more than BGI predicts
    # Expected: autosens ratio rises above baseline during day 2
    df_exercise, isf_exercise, _ = run_condition(
        label    = "EXERCISE — p2u ×1.5 at hour 24, restored at hour 48",
        shifts   = [(24, 1.5), (48, 1.0)],
        save_sub = "exercise",
        controller = 'adaptive',
    )

    # Condition 3: sickness at hour 24, restored at hour 48
    # p2u * 0.7 → patient absorbs glucose slower → BG falls less than BGI predicts
    # Expected: autosens ratio falls below baseline during day 2
    df_sickness, isf_sickness, _ = run_condition(
        label    = "SICKNESS — p2u ×0.7 at hour 24, restored at hour 48",
        shifts   = [(24, 0.7), (48, 1.0)],
        save_sub = "sickness",
        controller = 'adaptive',
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  AUTOSENS SENSITIVITY SHIFT TEST RESULTS")
    print("="*65)

    for df, isf_hist, lbl, save_sub in [
        (df_baseline, isf_baseline, "Baseline",     "baseline"),
        (df_exercise, isf_exercise, "Exercise",     "exercise"),
        (df_sickness, isf_sickness, "Sickness",     "sickness"),
    ]:
        summarise_by_day(df, lbl)
        summarise_isf_history(isf_hist, lbl)
        compute_bg_metrics(os.path.join(SAVE_BASE, save_sub), lbl)

    # ── Key assertions ────────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  PASS / FAIL CHECKS")
    print("="*65)

    def day_ratio(df, day):
        h_start = (day - 1) * 24
        h_end   = day * 24
        mask = (df['datetime'] >= START_TIME + timedelta(hours=h_start)) & \
               (df['datetime'] <  START_TIME + timedelta(hours=h_end))
        return df.loc[mask, 'ratio'].dropna()

    checks = []

    # 1. Ratio is always within safety bounds
    for df, lbl in [(df_baseline, 'baseline'), (df_exercise, 'exercise'), (df_sickness, 'sickness')]:
        r = df['ratio'].dropna()
        ok = (r >= 0.7).all() and (r <= 1.2).all()
        checks.append((f"Ratio always in [0.7, 1.2] — {lbl}", ok))

    # 2. Exercise raises ratio above baseline on day 2
    base_d2 = day_ratio(df_baseline, 2).mean()
    exer_d2 = day_ratio(df_exercise, 2).mean()
    checks.append((
        f"Exercise day 2 ratio ({exer_d2:.4f}) > baseline day 2 ({base_d2:.4f})",
        exer_d2 > base_d2
    ))

    # 3. Sickness lowers ratio below baseline on day 2
    sick_d2 = day_ratio(df_sickness, 2).mean()
    checks.append((
        f"Sickness day 2 ratio ({sick_d2:.4f}) < baseline day 2 ({base_d2:.4f})",
        sick_d2 < base_d2
    ))

    # 4. Exercise and sickness ratios diverge in opposite directions
    checks.append((
        f"Exercise ratio > sickness ratio on day 2 ({exer_d2:.4f} vs {sick_d2:.4f})",
        exer_d2 > sick_d2
    ))

    # 5. Ratio returns toward baseline on day 3 (within 10% of baseline)
    base_d3 = day_ratio(df_baseline, 3).mean()
    exer_d3 = day_ratio(df_exercise, 3).mean()
    sick_d3 = day_ratio(df_sickness, 3).mean()
    checks.append((
        f"Exercise ratio recovers on day 3 (within 0.05 of baseline: "
        f"{exer_d3:.4f} vs {base_d3:.4f})",
        abs(exer_d3 - base_d3) < 0.05
    ))
    checks.append((
        f"Sickness ratio recovers on day 3 (within 0.05 of baseline: "
        f"{sick_d3:.4f} vs {base_d3:.4f})",
        abs(sick_d3 - base_d3) < 0.05
    ))

    print()
    all_passed = True
    for desc, ok in checks:
        status = "  ✓ PASS" if ok else "  ✗ FAIL"
        print(f"{status}  {desc}")
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print("  All checks passed — two-layer autosens/autotune system validated.")
    else:
        print("  Some checks failed — review the per-day ratio tables above.")

    # ── Save logs and plot ────────────────────────────────────────────────────
    df_baseline.to_csv(os.path.join(SAVE_BASE, "log_baseline.csv"), index=False)
    df_exercise.to_csv(os.path.join(SAVE_BASE, "log_exercise.csv"), index=False)
    df_sickness.to_csv(os.path.join(SAVE_BASE, "log_sickness.csv"), index=False)

    plot_results(df_baseline, df_exercise, df_sickness,  SAVE_BASE, controller_label="Adaptive Loop Controller")

    print(f"\n  All logs saved to: {SAVE_BASE}")

    
        # Condition 1: no sensitivity shift — establishes patient baseline ratio
    df_baseline, isf_baseline, _ = run_condition(
        label    = "BASELINE — no sensitivity shift",
        shifts   = [],
        save_sub = "baseline",
        controller = 'loop',
    )

    # Condition 2: exercise at hour 24, restored at hour 48
    # p2u * 1.5 → patient absorbs glucose faster → BG falls more than BGI predicts
    # Expected: autosens ratio rises above baseline during day 2
    df_exercise, isf_exercise, _ = run_condition(
        label    = "EXERCISE — p2u ×1.5 at hour 24, restored at hour 48",
        shifts   = [(24, 1.5), (48, 1.0)],
        save_sub = "exercise",
        controller = 'loop',
    )

    # Condition 3: sickness at hour 24, restored at hour 48
    # p2u * 0.7 → patient absorbs glucose slower → BG falls less than BGI predicts
    # Expected: autosens ratio falls below baseline during day 2
    df_sickness, isf_sickness, _ = run_condition(
        label    = "SICKNESS — p2u ×0.7 at hour 24, restored at hour 48",
        shifts   = [(24, 0.7), (48, 1.0)],
        save_sub = "sickness",
        controller = 'loop',
    )
        # ── Print results ─────────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  AUTOSENS SENSITIVITY SHIFT TEST RESULTS")
    print("="*65)

    for df, isf_hist, lbl, save_sub in [
        (df_baseline, isf_baseline, "Baseline",     "baseline"),
        (df_exercise, isf_exercise, "Exercise",     "exercise"),
        (df_sickness, isf_sickness, "Sickness",     "sickness"),
    ]:
        summarise_by_day(df, lbl)
        summarise_isf_history(isf_hist, lbl)
        compute_bg_metrics(os.path.join(SAVE_BASE, save_sub), lbl)

    # ── Key assertions ────────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  PASS / FAIL CHECKS")
    print("="*65)

    def day_ratio(df, day):
        h_start = (day - 1) * 24
        h_end   = day * 24
        mask = (df['datetime'] >= START_TIME + timedelta(hours=h_start)) & \
               (df['datetime'] <  START_TIME + timedelta(hours=h_end))
        return df.loc[mask, 'ratio'].dropna()

    checks = []

    # 1. Ratio is always within safety bounds
    for df, lbl in [(df_baseline, 'baseline'), (df_exercise, 'exercise'), (df_sickness, 'sickness')]:
        r = df['ratio'].dropna()
        ok = (r >= 0.7).all() and (r <= 1.2).all()
        checks.append((f"Ratio always in [0.7, 1.2] — {lbl}", ok))

    # 2. Exercise raises ratio above baseline on day 2
    base_d2 = day_ratio(df_baseline, 2).mean()
    exer_d2 = day_ratio(df_exercise, 2).mean()
    checks.append((
        f"Exercise day 2 ratio ({exer_d2:.4f}) > baseline day 2 ({base_d2:.4f})",
        exer_d2 > base_d2
    ))

    # 3. Sickness lowers ratio below baseline on day 2
    sick_d2 = day_ratio(df_sickness, 2).mean()
    checks.append((
        f"Sickness day 2 ratio ({sick_d2:.4f}) < baseline day 2 ({base_d2:.4f})",
        sick_d2 < base_d2
    ))

    # 4. Exercise and sickness ratios diverge in opposite directions
    checks.append((
        f"Exercise ratio > sickness ratio on day 2 ({exer_d2:.4f} vs {sick_d2:.4f})",
        exer_d2 > sick_d2
    ))

    # 5. Ratio returns toward baseline on day 3 (within 10% of baseline)
    base_d3 = day_ratio(df_baseline, 3).mean()
    exer_d3 = day_ratio(df_exercise, 3).mean()
    sick_d3 = day_ratio(df_sickness, 3).mean()
    checks.append((
        f"Exercise ratio recovers on day 3 (within 0.05 of baseline: "
        f"{exer_d3:.4f} vs {base_d3:.4f})",
        abs(exer_d3 - base_d3) < 0.05
    ))
    checks.append((
        f"Sickness ratio recovers on day 3 (within 0.05 of baseline: "
        f"{sick_d3:.4f} vs {base_d3:.4f})",
        abs(sick_d3 - base_d3) < 0.05
    ))

    print()
    all_passed = True
    for desc, ok in checks:
        status = "  ✓ PASS" if ok else "  ✗ FAIL"
        print(f"{status}  {desc}")
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print("  All checks passed — two-layer autosens/autotune system validated.")
    else:
        print("  Some checks failed — review the per-day ratio tables above.")

    # ── Save logs and plot ────────────────────────────────────────────────────
    df_baseline.to_csv(os.path.join(SAVE_BASE, "log_baseline.csv"), index=False)
    df_exercise.to_csv(os.path.join(SAVE_BASE, "log_exercise.csv"), index=False)
    df_sickness.to_csv(os.path.join(SAVE_BASE, "log_sickness.csv"), index=False)

    plot_results(df_baseline, df_exercise, df_sickness,  SAVE_BASE, controller_label="Non-Adaptive Loop Controller")

    print(f"\n  All logs saved to: {SAVE_BASE}")
