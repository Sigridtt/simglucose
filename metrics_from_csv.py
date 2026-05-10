import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# CONFIG
# ==========================
BASE_PATH = "./examples/results/illness_compare_all_New"
LOOP_PATH = os.path.join(BASE_PATH, "loop")
ADAPTIVE_PATH = os.path.join(BASE_PATH, "adaptive")
OUTPUT_PATH = BASE_PATH  # save directly here

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ==========================
# LOAD CSV FILES
# ==========================
def load_histories(path):
    histories = {}

    for file in os.listdir(path):

        # ONLY keep real patient files
        if not file.endswith(".csv"):
            continue

        if "_illness_trace" in file:
            continue

        if "report" in file or "combined" in file:
            continue

        if "metrics" in file or "summary" in file:
            continue

        full_path = os.path.join(path, file)

        df = pd.read_csv(full_path)

        # Normalize columns
        df.columns = [c.strip() for c in df.columns]

        #  Only accept files that contain BG
        if "BG" not in df.columns:
            print(f"[SKIP] Not a patient file: {file}")
            continue

        # Set time index
        if "Time" in df.columns:
            df["Time"] = pd.to_datetime(df["Time"])
            df = df.set_index("Time")

        patient_name = file.replace(".csv", "")
        histories[patient_name] = df

    return histories



loop_histories = load_histories(LOOP_PATH)
adaptive_histories = load_histories(ADAPTIVE_PATH)

patients = sorted(list(set(loop_histories.keys()) & set(adaptive_histories.keys())))

# ==========================
# FAST METRICS (NO report())
# ==========================
def compute_metrics(df, patient_name="unknown"):
    if df is None or df.empty:
        print(f"[WARN] Empty dataframe for {patient_name}")
        return None

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Debug print once if BG missing
    if "BG" not in df.columns:
        print(f"[ERROR] 'BG' column not found for {patient_name}")
        print(f"Available columns: {list(df.columns)}")
        return None

    bg = df["BG"]

    tir = 100.0 * ((bg >= 70) & (bg <= 180)).mean()
    tar = 100.0 * (bg > 180).mean()
    tbr = 100.0 * (bg < 70).mean()

    # Risk column safe handling
    risk = df["Risk"].mean() if "Risk" in df.columns else np.nan

    return {
        "TIR": tir,
        "TAR": tar,
        "TBR": tbr,
        "RiskIndex": risk
    }

# ==========================
# COMPUTE ALL METRICS
# ==========================
rows = []

for p in patients:
    loop_metrics = compute_metrics(loop_histories[p])
    adaptive_metrics = compute_metrics(adaptive_histories[p])

    if loop_metrics is None or adaptive_metrics is None:
        continue

    row = {
        "Patient": p,

        "Loop_TIR": loop_metrics["TIR"],
        "Adaptive_TIR": adaptive_metrics["TIR"],

        "Loop_TAR": loop_metrics["TAR"],
        "Adaptive_TAR": adaptive_metrics["TAR"],

        "Loop_TBR": loop_metrics["TBR"],
        "Adaptive_TBR": adaptive_metrics["TBR"],

        "Loop_RiskIndex": loop_metrics["RiskIndex"],
        "Adaptive_RiskIndex": adaptive_metrics["RiskIndex"],

        "Delta_TIR": adaptive_metrics["TIR"] - loop_metrics["TIR"],
        "Delta_TAR": adaptive_metrics["TAR"] - loop_metrics["TAR"],
        "Delta_TBR": adaptive_metrics["TBR"] - loop_metrics["TBR"],
        "Delta_RiskIndex": adaptive_metrics["RiskIndex"] - loop_metrics["RiskIndex"],
    }

    rows.append(row)

combined_df = pd.DataFrame(rows).set_index("Patient")

# Save CSV
combined_csv_path = os.path.join(OUTPUT_PATH, "combined_metrics_results.csv")
combined_df.to_csv(combined_csv_path)

# ==========================
# PLOT TIR BEFORE vs AFTER
# ==========================
plt.figure(figsize=(10, 6))

for p in combined_df.index:
    y = [combined_df.loc[p, "Loop_TIR"], combined_df.loc[p, "Adaptive_TIR"]]
    plt.plot([0, 1], y, marker="o", linewidth=1.5)
    plt.text(1.02, y[1], p, fontsize=8, va="center")

plt.xticks([0, 1], ["Loop", "Adaptive"])
plt.ylabel("Time In Range (70–180 mg/dL), %")
plt.title("TIR Before vs After (Fast Metrics from CSV)")
plt.grid(True)
plt.xlim(-0.1, 1.3)

plot_path = os.path.join(OUTPUT_PATH, "tir_comparison.png")
plt.savefig(plot_path, dpi=200)
plt.close()

# ==========================
# PRINT SUMMARY
# ==========================
print("\n=== RESULTS ===")
print(combined_df.round(3))

print("\nSaved files:")
print(f"- Combined CSV: {combined_csv_path}")
print(f"- Plot: {plot_path}")