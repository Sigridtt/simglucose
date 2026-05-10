import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
#metrics_df = metrics_all_contitions.csv which is in results/ablation_5conditions
import pkg_resources

from datetime import datetime, timedelta

#from debug_deviation_bias import SAVE_PATH


PATIENT_PARA_FILE = pkg_resources.resource_filename("simglucose", "params/vpatient_params.csv")

def get_all_patients():
    return list(pd.read_csv(PATIENT_PARA_FILE)["Name"].values)

CONDITIONS = {
    "A_Loop": {
        "label": "A: Loop only",
        "use_adaptive": False,
    },
    "B_Autotune": {
        "label": "B: Loop + Autotune",
        "use_adaptive": True,
        "enable_autotune": True,
        "enable_autosens": False,
        "enable_sick_detection": False,
    },
    "C_Autosens": {
        "label": "C: Loop + Autosens",
        "use_adaptive": True,
        "enable_autotune": False,
        "enable_autosens": True,
        "enable_sick_detection": False,
    },
    "D_Autotune_Autosens": {
        "label": "D: Loop +Autotune + Autosens",
        "use_adaptive": True,
        "enable_autotune": True,
        "enable_autosens": True,
        "enable_sick_detection": False,
    },
    "E_Full": {
        "label": "E: Full Adaptive",
        "use_adaptive": True,
        "enable_autotune": True,
        "enable_autosens": True,
        "enable_sick_detection": True,
    },
}
import os
import pandas as pd

def load_histories_from_root(root_path, conditions):
    """
    Builds:
      all_histories[condition][patient_name] -> BG dataframe
      all_illness_traces[condition][patient_name] -> illness trace dataframe
    """
    all_histories = {}
    all_illness_traces = {}

    for cond_key in conditions.keys():
        cond_path = os.path.join(root_path, cond_key)

        histories = {}
        traces = {}

        if not os.path.isdir(cond_path):
            print(f"Warning: missing folder {cond_path}")
            continue

        for fname in os.listdir(cond_path):
            fpath = os.path.join(cond_path, fname)

            if not fname.endswith(".csv"):
                continue

            # --- illness trace ---
            if "_illness_trace" in fname:
                patient_name = fname.replace("_illness_trace.csv", "")
                df = pd.read_csv(fpath)

                # ensure datetime index
                if "Time" in df.columns:
                    df["Time"] = pd.to_datetime(df["Time"])
                    df = df.set_index("Time")

                traces[patient_name] = df

            # --- history (BG) ---
            else:
                patient_name = fname.replace(".csv", "")
                df = pd.read_csv(fpath)

                if "Time" in df.columns:
                    df["Time"] = pd.to_datetime(df["Time"])
                    df = df.set_index("Time")

                histories[patient_name] = df

        all_histories[cond_key] = histories
        all_illness_traces[cond_key] = traces

    return all_histories, all_illness_traces

def metrics_for_window(df, t_start=None, t_end=None):
    """Compute TIR/TAR/TBR for a time window of a BG history dataframe."""
    if df is None or df.empty or "BG" not in df.columns:
        return {"TIR": np.nan, "TAR": np.nan, "TBR": np.nan}
    bg = df["BG"]
    if t_start is not None:
        bg = bg[bg.index >= t_start]
    if t_end is not None:
        bg = bg[bg.index < t_end]
    if bg.empty:
        return {"TIR": np.nan, "TAR": np.nan, "TBR": np.nan}
    return {
        "TIR": 100.0 * ((bg >= 70) & (bg <= 180)).mean(),
        "TAR": 100.0 * (bg > 180).mean(),
        "TBR": 100.0 * (bg < 70).mean(),
    }

def save_bg_4conditions_plot(save_path, patient_name, all_histories, all_illness_traces):
    """5-panel BG trace, one per condition, illness window shaded."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    START_TIME             = datetime(2018, 1, 1, 8, 0, 0)

    ILLNESS_START_DAY      = 4
    ILLNESS_END_DAY        = 10
    
    ILLNESS_START_TIME  = START_TIME + timedelta(days=ILLNESS_START_DAY)
    ILLNESS_END_TIME    = START_TIME + timedelta(days=ILLNESS_END_DAY)

    illness_start = pd.Timestamp(ILLNESS_START_TIME)
    illness_end   = pd.Timestamp(ILLNESS_END_TIME)

    cond_colors = {
        "A_Loop":     "tab:blue",
        "B_Autotune": "tab:red",
        "C_Autosens": "tab:orange",
        "D_Autotune_Autosens": "tab:green",
        "E_Full":     "tab:purple",
    }
    GROUP_ABCD = ["C_Autosens", "D_Autotune_Autosens", "B_Autotune","A_Loop"]
    GROUP_E    = ["E_Full"]

    for ax, group in zip(axes, [GROUP_ABCD, GROUP_E]):
        title_parts = []

        for cond_key in group:
            df     = all_histories.get(cond_key, {}).get(patient_name)
            color  = cond_colors[cond_key]
            label  = CONDITIONS[cond_key]["label"]

            if df is not None and "BG" in df.columns:
                ax.plot(df.index, df["BG"], color=color, linewidth=1.0,
                        alpha=0.9, label=label)

            # Sick-detection overlay per condition
            trace = all_illness_traces.get(cond_key, {}).get(patient_name)
            if trace is not None and "sick" in trace.columns:
                sick_series = trace["sick"].astype(float)
                if (sick_series > 0.5).any():
                    ax.fill_between(sick_series.index,
                                    30, 350,           # use fixed ylim, not ax.get_ylim()
                                    where=sick_series > 0.5,
                                    alpha=0.08, color=color,
                                    label=f"Sick detected ({label})")

            m_illness = metrics_for_window(df, illness_start, illness_end)
            m_post    = metrics_for_window(df, illness_end, None)
            title_parts.append(
                f"{label}: illness TIR={m_illness['TIR']:.1f}% "
                f"TAR={m_illness['TAR']:.1f}% TBR={m_illness['TBR']:.1f}% "
                f"| post TBR={m_post['TBR']:.1f}%"
            )

        ax.axhline(70,  color="red",    linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(180, color="orange", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvspan(illness_start, illness_end, alpha=0.1, color="red", label="Illness window")
        ax.set_ylabel("BG (mg/dL)")
        ax.set_ylim(30, 350)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title("\n".join(title_parts), fontsize=8)

    axes[-1].set_xlabel("Date")
    fig.tight_layout()

    safe = patient_name.replace("#", "_")
    path = os.path.join(save_path, f"bg_5conditions_{safe}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path

def save_tbr_comparison_plot(metrics_df, save_path):
    """

    Each patient is a line; conditions are x-axis ticks.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    cond_labels = [CONDITIONS[k]["label"] for k in CONDITIONS]
    x = list(range(len(cond_labels)))

    # Post-illness TBR (key metric for autotune danger)
    for patient in metrics_df.index:
        y = []
        for cond_key in CONDITIONS:
            col = f"{cond_key}_post_TBR"
            y.append(metrics_df.loc[patient, col] if col in metrics_df.columns else np.nan)
        if any(~np.isnan(y)):
            line, = ax.plot(x, y, marker="o", linewidth=1.2, alpha=0.75)
            ax.text(x[-1] + 0.05, y[-1], patient, fontsize=9,
                    va="center", color=line.get_color())

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=15, ha="right")
    ax.set_ylabel("Time Below Range (%)")
    ax.set_title(
        "Post-illness TBR (days 10–14)\n"
        #"Lower is better — B↑TBR shows autotune danger without sick detection"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 35)

  
    fig.tight_layout()

    path = os.path.join(save_path, "tbr_5conditions.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path

def save_illness_tir_comparison_plot(metrics_df, save_path):
    """
    Each patient is a line; conditions are x-axis ticks.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    cond_labels = [CONDITIONS[k]["label"] for k in CONDITIONS]
    x = list(range(len(cond_labels)))

    # Illness-window TIR
    for patient in metrics_df.index:
        y = []
        for cond_key in CONDITIONS:
            col = f"{cond_key}_illness_TIR"
            y.append(metrics_df.loc[patient, col] if col in metrics_df.columns else np.nan)
        if any(~np.isnan(y)):
            line, = ax.plot(x, y, marker="o", linewidth=1.2, alpha=0.75)
            ax.text(x[-1] + 0.05, y[-1], patient, fontsize=9,
                    va="center", color=line.get_color())

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=15, ha="right")
    ax.set_ylabel("Time In Range (%)")
    ax.set_title("Illness-window TIR (days 4–10)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(25, 60)


    fig.tight_layout()

    path = os.path.join(save_path, "tir_5conditions.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
def get_patient_groups_from_name(patients):
    groups = {}

    for p in patients:
        prefix = p.split("#")[0].lower()

        if prefix in ["child", "adolescent", "adult"]:
            groups[p] = prefix
        else:
            groups[p] = "unknown"  # fallback safety

    return groups

def save_illness_tir_comparison_plot_age(metrics_df, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    group_order = ["child", "adolescent", "adult"]
    groups = get_patient_groups_from_name(metrics_df.index)

    cond_labels = [CONDITIONS[k]["label"] for k in CONDITIONS]
    x = list(range(len(cond_labels)))

 
    colors = {
        "child": "tab:blue",
        "adolescent": "tab:orange",
        "adult": "tab:green",
    }

    for group in group_order:
        group_patients = [p for p in metrics_df.index if groups.get(p) == group]

        if len(group_patients) == 0:
            continue

        mean_vals = []

        for cond_key in CONDITIONS:
            col = f"{cond_key}_illness_TIR"

            if col in metrics_df.columns:
                vals = metrics_df.loc[group_patients, col].values
                mean_vals.append(np.nanmean(vals))
            else:
                mean_vals.append(np.nan)

        ax.plot(
            x,
            mean_vals,
            marker="o",
            linewidth=2.5,
            label=group.capitalize(),
            color=colors.get(group),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=15, ha="right")
    ax.set_ylabel("Mean Illness TIR (%)")
    ax.set_title("Illness-window TIR by age group (mean across patients)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(25, 65)

    ax.legend(fontsize=10)

    fig.tight_layout()

    path = os.path.join(save_path, "tir_5conditions_by_age_mean.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return path
def save_mean_tir_by_window(metrics_df, save_path):
    """
    Plot mean TIR for windows: pre, illness, post for all conditions.
    Adds one movable textbox per x-position listing all condition values.
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    windows = ["pre", "illness", "post"]
    window_labels = ["Pre", "Illness", "Post"]
    x = list(range(len(windows)))

    fig, ax = plt.subplots(figsize=(12, 5))

    cond_colors = {
        "A_Loop":     "tab:blue",
        "B_Autotune": "tab:red",
        "C_Autosens": "tab:orange",
        "D_Autotune_Autosens": "tab:green",
        "E_Full":     "tab:purple",
    }

    # Store values per window for textbox creation
    window_values = {w: [] for w in windows}

    # ---- plot lines ----
    for cond_key in CONDITIONS:
        vals = []
        for w in windows:
            col = f"{cond_key}_{w}_TIR"
            if col in metrics_df.columns:
                val = metrics_df[col].mean()
            else:
                val = np.nan

            vals.append(val)
            window_values[w].append((cond_key, val))

        ax.plot(
            x,
            vals,
            marker="o",
            linewidth=2.0,
            label=CONDITIONS[cond_key]["label"],
            color=cond_colors.get(cond_key),
        )

   # ---- add color-coded text boxes ----
    textbox_y_positions = {
        "pre": 75,
        "illness": 75,
        "post": 75,
    }
    textbox_x_positions = {
        "pre": 0,
        "illness": 1,
        "post": 2,
    }
    line_spacing = 2.2  # vertical spacing between lines

    for i, w in enumerate(windows):
        entries = [
            (cond_key, val)
            for cond_key, val in window_values[w]
            if not np.isnan(val)
        ]

        # Draw a background box (empty text, just for the frame)
        ax.text(
            textbox_x_positions[w],
            textbox_y_positions[w],
            "",  # empty, just to create the box
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                alpha=0.8,
                edgecolor="gray"
            )
        )

        # Now draw each line separately (colored)
        for j, (cond_key, val) in enumerate(entries):
            ax.text(
                textbox_x_positions[w],
                textbox_y_positions[w] - j * line_spacing,
                f"{val:.3f}%",
                color=cond_colors.get(cond_key),
                fontsize=9,
                ha="left",
                va="top",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(window_labels)
    ax.set_ylabel("Mean TIR (%)")
    ax.set_title("Mean TIR before / during / after illness — by condition")

    ax.grid(True, alpha=0.3)
    ax.set_ylim(30, 100)

    ax.legend(fontsize=9)

    fig.tight_layout()

    path = os.path.join(save_path, "mean_tir_by_window_5conditions.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved mean TIR by window: {path}")
    return path

def save_mean_tbr_by_window(metrics_df, save_path):
    """
    Plot mean TBR for windows: pre, illness, post for all conditions.
    Adds one movable textbox per x-position listing all condition values.
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    windows = ["pre", "illness", "post"]
    window_labels = ["Pre", "Illness", "Post"]
    x = list(range(len(windows)))

    fig, ax = plt.subplots(figsize=(12, 5))

    cond_colors = {
        "A_Loop":     "tab:blue",
        "B_Autotune": "tab:red",
        "C_Autosens": "tab:orange",
        "D_Autotune_Autosens": "tab:green",
        "E_Full":     "tab:purple",
    }

    # Store values per window for textbox creation
    window_values = {w: [] for w in windows}

    # ---- plot lines ----
    for cond_key in CONDITIONS:
        vals = []
        for w in windows:
            col = f"{cond_key}_{w}_TBR"
            if col in metrics_df.columns:
                val = metrics_df[col].mean()
            else:
                val = np.nan

            vals.append(val)
            window_values[w].append((cond_key, val))

        ax.plot(
            x,
            vals,
            marker="o",
            linewidth=2,
            label=CONDITIONS[cond_key]["label"],
            color=cond_colors.get(cond_key),
        )

   # ---- add color-coded text boxes ----
    textbox_y_positions = {
        "pre": 5,
        "illness": 3,
        "post": 2,
    }
    textbox_x_positions = {
        "pre": 0.2,
        "illness": 1,
        "post": 2,
    }
    line_spacing = 0.2  # vertical spacing between lines

    for i, w in enumerate(windows):
        entries = [
            (cond_key, val)
            for cond_key, val in window_values[w]
            if not np.isnan(val)
        ]

        # Draw a background box (empty text, just for the frame)
        ax.text(
            textbox_x_positions[w],
            textbox_y_positions[w],
            "",  # empty, just to create the box
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                alpha=0.8,
                edgecolor="gray"
            )
        )

        # Now draw each line separately (colored)
        for j, (cond_key, val) in enumerate(entries):
            ax.text(
                textbox_x_positions[w],
                textbox_y_positions[w] - j * line_spacing,
                f"{val:.3f}%",
                color=cond_colors.get(cond_key),
                fontsize=9,
                ha="left",
                va="top",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(window_labels)
    ax.set_ylabel("Mean TBR (%)")
    ax.set_title("Mean TBR before / during / after illness — by condition")

    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 6)

    ax.legend(fontsize=9)

    fig.tight_layout()

    path = os.path.join(save_path, "mean_tbr_by_window_5conditions.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved mean TBR by window: {path}")
    return path

def bg_trace_with_illness_shade(df, all_histories, all_illness_traces):
    """Plot all patients' BG traces for a given condition on top of each other
    and overlay the mean trace with the illness window shaded.

    Parameters
    - all_histories: dict mapping condition_key -> {patient_name: df}
    - condition_key: which condition to plot (default 'A_Loop')
    - save_path: directory to save the figure
    """

    # Accept either the new-style call:
    #   bg_trace_with_illness_shade(all_histories, condition_key, save_path)
    # or the legacy call signature used earlier.
    if isinstance(df, dict) and isinstance(all_histories, str):
        histories = df
        condition_key = all_histories
        save_path = all_illness_traces
    elif isinstance(df, dict) and (all_histories is None or isinstance(all_histories, dict)):
        histories = df
        condition_key = "A_Loop"
        # save_path may be passed as third arg
        if isinstance(all_illness_traces, str):
            save_path = all_illness_traces
    else:
        # fallback: treat `all_histories` parameter as the histories mapping
        histories = all_histories if isinstance(all_histories, dict) else {}
        condition_key = "A_Loop"
        # third arg could be save_path
        save_path = all_illness_traces if isinstance(all_illness_traces, str) else save_path

    # use the resolved histories mapping
    histories = histories.get(condition_key, {}) if isinstance(histories, dict) else {}
    if not histories:
        print(f"No histories found for condition: {condition_key}")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot individual patient traces faintly
    series_list = []
    for patient_name, df in histories.items():
        if df is None or "BG" not in df.columns:
            continue
        s = df["BG"].rename(patient_name)
        series_list.append(s)
        ax.plot(s.index, s.values, color="gray", linewidth=0.6, alpha=0.25)

    if not series_list:
        print(f"No BG series available for condition {condition_key}")
        return None

    # Combine on common index and compute mean
    combined = pd.concat(series_list, axis=1)
    mean_series = combined.mean(axis=1)
    ax.plot(mean_series.index, mean_series.values, color="tab:blue", linewidth=2.0, label="Mean BG")

    # Illness window shading (same defaults used across scripts)
    START_TIME = datetime(2018, 1, 1, 8, 0, 0)
    ILLNESS_START_DAY = 4
    ILLNESS_END_DAY = 10
    illness_start = pd.Timestamp(START_TIME + timedelta(days=ILLNESS_START_DAY))
    illness_end = pd.Timestamp(START_TIME + timedelta(days=ILLNESS_END_DAY))

    ax.axhline(70, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(180, color="orange", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvspan(illness_start, illness_end, alpha=0.12, color="red")

    ax.set_ylabel("BG (mg/dL)")
    ax.set_ylim(0, 600)
    ax.grid(True, alpha=0.2)
    ax.legend()

    fig.suptitle(f"All 30 patients — {CONDITIONS[condition_key]['label']} (overlaid)\nMean in blue and illness days shaded", fontsize=13)
    fig.tight_layout()

    safe = condition_key.replace("#", "_")
    out = os.path.join(save_path if save_path else os.getcwd(), f"bg_all_{safe}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Saved overlay BG traces: {out}")
    return out

def save_illness_profile_plot(save_path):
    SIM_DAYS = 14
    ILLNESS_START = 4
    ILLNESS_END = 10

    TARGET_REDUCTION_FACTOR = 0.2

    steps_per_day = int(24 * 60 / 5)
    total_steps = SIM_DAYS * steps_per_day

    illness_start_step = ILLNESS_START * steps_per_day
    illness_end_step = ILLNESS_END * steps_per_day

    ramp_up_steps = 1 * steps_per_day
    ramp_down_steps = 1 * steps_per_day
    steady_steps = (illness_end_step - illness_start_step) - ramp_up_steps - ramp_down_steps

    def illness_intensity(step_idx):
        if step_idx < illness_start_step or step_idx >= illness_end_step:
            return 0.0

        rel = step_idx - illness_start_step

        if ramp_up_steps > 0 and rel < ramp_up_steps:
            return rel / float(ramp_up_steps)

        plateau_end = ramp_up_steps + steady_steps

        if rel < plateau_end:
            return 1.0

        if ramp_down_steps <= 0:
            return 0.0

        down_rel = rel - plateau_end
        return max(0.0, 1.0 - down_rel / float(ramp_down_steps))

    step_axis = np.arange(total_steps)
    day_axis = step_axis / steps_per_day

    intensity_vals = np.array([illness_intensity(i) for i in step_axis])
    reduction_vals = 1.0 - (1.0 - TARGET_REDUCTION_FACTOR) * intensity_vals

    fig, ax1 = plt.subplots(figsize=(11, 4))

    ax1.plot(day_axis, intensity_vals, label="Illness intensity", linewidth=2)
    ax1.plot(day_axis, reduction_vals, label="Reduction factor", linewidth=2)

    ax1.axvspan(ILLNESS_START, ILLNESS_END, alpha=0.1, color="red")

    ax1.set_xlabel("Simulation day")
    ax1.set_ylabel("Unitless scale")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax1.legend(loc="upper right", fontsize=9)

    
    fig.suptitle(f"Illness profile for 14 days simulation, illness days shaded", fontsize=13)
    fig.tight_layout()
    path = os.path.join(save_path, "illness_profile.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return path

def save_autosens_ratio_vs_Vmx_plot(save_path, patient_name, all_illness_traces):

    import os
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # Illness window shading (same defaults used across scripts)
    START_TIME = datetime(2018, 1, 1, 8, 0, 0)
    ILLNESS_START_DAY = 4
    ILLNESS_END_DAY = 10
    AUTOSENS_LEARNING_END = 1
    AUTOTUNE_LEARNING_END = 7
    illness_start = pd.Timestamp(START_TIME + timedelta(days=ILLNESS_START_DAY))
    illness_end = pd.Timestamp(START_TIME + timedelta(days=ILLNESS_END_DAY))
    autosens_learning_end = pd.Timestamp(START_TIME + timedelta(days=AUTOSENS_LEARNING_END))
    autotune_learning_end = pd.Timestamp(START_TIME + timedelta(days=AUTOTUNE_LEARNING_END))

    colors = {
        "E_Full": "tab:purple",
        "D_Autotune_Autosens": "tab:green",
        "C_Autosens": "tab:orange",
    }

    for cond_key, color in colors.items():

        trace = all_illness_traces.get(cond_key, {}).get(patient_name)
        if trace is None:
            continue

        if "autosens_ratio" not in trace.columns:
            continue

        series = trace["autosens_ratio"].astype(float)
        ax.plot(series.index, series, color=color, linewidth=1.2,
                label=f"{CONDITIONS[cond_key]['label']} (autosens_ratio)")


    if "Vmx_relative" in trace.columns:
        Vmx = trace["Vmx_relative"].astype(float)
        Vmx_flipped = 2.0 - Vmx
        ax.plot(
            Vmx.index,
            Vmx_flipped,
            color="tab:blue",        
            linewidth=2.0,
            alpha=0.8,
            label=f"Relative insulin sensitivity (Vmx/baseline) inverted for comparison"
        )
    ax.axvline(
        autosens_learning_end,
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Autosens learning end (after 1 day)"
    )

    ax.axvline(
        autotune_learning_end,
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Autotune learning end (after 7 days)"
    )
    """
    ax.fill_betweenx(
        illness_df = all_illness_traces.get(cond_key, {}).get(patient_name)
        if trace is None:
            continue

        if "autosens_t_start" or "autosens_t_end" not in trace.columns:
            continue
        [ax.get_ylim()[0], ax.get_ylim()[1]],
        illness_df["autosens_t_start"],
        illness_df["autosens_t_end"],
        alpha=0.08, color="steelblue", label="autosens data window",
    )
    """
    #ax.plot(illness_df.index, illness_df["autosens_ratio"], label="autosens ratio")
    #ax.plot(illness_df.index, illness_df["Vmx_relative"],   label="Vmx relative")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # If index is time, convert illness window properly (optional fix below)
    ax.set_title(
        f"{patient_name} | Autosens vs Vmx during illness, illness days shaded"
    )
    ax.axvspan(illness_start, illness_end, alpha=0.12, color="red")
    ax.set_ylabel("Value (normalized)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    fig.tight_layout()

    safe = patient_name.replace("#", "_")
    path = os.path.join(save_path, f"autosens_vs_Vmx_{safe}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved: {path}")
    return path

def build_metrics_table(all_histories):
    """
    Build a per-patient metrics table across all conditions and time windows.
    all_histories: dict[condition_key → dict[patient_name → df]]
    all_report_results: dict[condition_key → dict[patient_name → df]]
    """
    rows = []
    patients = sorted(set.intersection(*[set(h.keys()) for h in all_histories.values()]))
    START_TIME             = datetime(2018, 1, 1, 8, 0, 0)

    ILLNESS_START_DAY      = 4
    ILLNESS_END_DAY        = 10
    
    ILLNESS_START_TIME  = START_TIME + timedelta(days=ILLNESS_START_DAY)
    ILLNESS_END_TIME    = START_TIME + timedelta(days=ILLNESS_END_DAY)

    windows = {
        "full":        (None,               None),
        "pre":         (None,               ILLNESS_START_TIME),
        "illness":     (ILLNESS_START_TIME,  ILLNESS_END_TIME),
        "post":        (ILLNESS_END_TIME,    None),
    }

    for patient in patients:
        row = {"Patient": patient}
        for cond_key in CONDITIONS:
            df = all_histories.get(cond_key, {}).get(patient)
            for win_name, (t0, t1) in windows.items():
                m = metrics_for_window(df, t0, t1)
                for metric, val in m.items():
                    row[f"{cond_key}_{win_name}_{metric}"] = val
        rows.append(row)

    return pd.DataFrame(rows).set_index("Patient").sort_index()

def main():
    save_path = os.path.join(
        os.path.dirname(__file__),
        "examples", "results", "postprocessing", "ALL_PATIENTS" 
    )
    histories_root = os.path.join(
        os.path.dirname(__file__),
        "examples", "results", "supersick_allPatients_02",
    )
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "examples", "results", "supersick_allPatients_02",
        "metrics_all_conditions.csv",
    )
    #fixing metrics table if not correct
    """
    PATIENT_NAMES = ["adult#003", "child#007","child#008","adolescent#007" ]
    all_histories      = {}
    for cond_key in CONDITIONS:
        all_histories[cond_key] = {}
        for patient in PATIENT_NAMES:
            histories= os.path.join(histories_root, cond_key, f"{patient}.csv")
            df = pd.read_csv(histories, index_col=0, parse_dates=True)
            all_histories[cond_key][patient] = df
    metrics_df = build_metrics_table(all_histories)
    metrics_df.to_csv(os.path.join(save_path, "metrics_all_conditions_new.csv"))
    """


    metrics_df = pd.read_csv(csv_path, index_col=0)
    print(metrics_df.head())
    print(metrics_df.columns)
    # Ensure the postprocessing save directory exists before writing any plots
    os.makedirs(save_path, exist_ok=True)

    plot_path = save_illness_tir_comparison_plot(metrics_df, save_path)
    print(f"Saved illness TIR comparison plot to: {plot_path}")
    plot_path = save_tbr_comparison_plot(metrics_df, save_path)
    print(f"Saved TBR comparison plot to: {plot_path}")
    plot_path = save_illness_profile_plot(save_path)
    print(f"Saved illness profile plot to: {plot_path}")

    # mean TIR by window (pre / illness / post)
    plot_path = save_mean_tir_by_window(metrics_df, save_path)
    plot_path = save_mean_tbr_by_window(metrics_df, save_path)
    plot_path = save_illness_tir_comparison_plot_age(metrics_df, save_path)

    # Load histories from the parent folder of the postprocessing save directory
    os.makedirs(save_path, exist_ok=True)
    
    
    all_histories, all_illness_traces = load_histories_from_root(histories_root, CONDITIONS)

    patients = get_all_patients()
    #patients = ["adult#003", "child#007","child#008","adolescent#007" ]
    """
    for patient in patients:
        plot_path = save_autosens_ratio_vs_Vmx_plot(save_path, patient, all_illness_traces)
        print(f"saved BG trace for all patients to {plot_path}")
    
    # Overlay BG traces for all patients in Condition A to show illness plausibility
    plot_path = bg_trace_with_illness_shade(all_histories, "A_Loop", save_path)
    """
    
    for patient in patients:
        plot_path = save_bg_4conditions_plot(save_path, patient, all_histories, all_illness_traces)
        print(f"saved BG trace for all patients to {plot_path}")
    

if __name__ == "__main__":
    main()