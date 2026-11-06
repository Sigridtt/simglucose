"""Plot the fuzzy Auto Basal rules of adaptiveController780G (fuzzylogic_basal.m).

Panels
  (a) BGL membership functions  (Low / Mid / High)
  (b) BGL-rate membership functions (Falling / Steady / Rising)
  (c) Rule table: rule number and Auto Basal factor per BGL set x rate set
  (d) Resulting Auto Basal factor over BGL x rate, with the hard limits
Optionally, a worked example (EXAMPLE = (BGL, rate)) is marked in all panels;
EXAMPLE = None (default) gives the general figure without a specific point.

The numbers mirror fuzzylogic_basal.m and cfg.FuzzyBasal in
adaptiveController780G.m (local branch local/ifb-fuzzy-basal). Edit P below
if you change them there.

Usage:  python plot_fuzzy_basal.py            -> shows the figure
        python plot_fuzzy_basal.py out.png    -> also saves it
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# ---- parameters (same as cfg.FuzzyBasal defaults) ----
P = dict(
    BGLLow=4.5, BGLMid=8.0, BGLHigh=25.0,   # [mmol/L]
    SlopeFast=-0.5, SlopeRise=0.5,          # [mmol/L/5min]
    Rules=np.array([[0.0, 0.5, 1.0],        # BGL Low : Falling Steady Rising
                    [0.5, 1.0, 1.0],        # BGL Mid
                    [0.7, 1.0, 1.0]]),      # BGL High
)
EXAMPLE = None                              # e.g. (6.0, -0.25) to mark a worked example (BGL, rate)

# ---- colours ----
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (blue, orange, aqua)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)   # factor 0 (light) -> 1 (dark)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
HARD = "#bdbcb8"                            # "no insulin" areas


# ---- fuzzy logic, line by line as fuzzylogic_basal.m ----
def mu_bgl(b):
    b = min(b, P["BGLHigh"])
    if b < P["BGLLow"]:
        return np.array([1.0, 0.0, 0.0])    # only for plotting; hard limit applies here
    if b <= P["BGLMid"]:
        t = (b - P["BGLLow"]) / (P["BGLMid"] - P["BGLLow"])
        return np.array([1 - t, t, 0.0])
    t = (b - P["BGLMid"]) / (P["BGLHigh"] - P["BGLMid"])
    return np.array([0.0, 1 - t, t])


def mu_slope(s):
    s = min(s, P["SlopeRise"])
    if s <= P["SlopeFast"]:
        return np.array([1.0, 0.0, 0.0])    # only for plotting; hard limit applies here
    if s <= 0:
        t = (s - P["SlopeFast"]) / (0 - P["SlopeFast"])
        return np.array([1 - t, t, 0.0])
    t = s / P["SlopeRise"]
    return np.array([0.0, 1 - t, t])


def factor(b, s):
    if b < P["BGLLow"] or s <= P["SlopeFast"]:
        return 0.0
    W = np.outer(mu_bgl(b), mu_slope(s))    # product AND
    return float(np.clip((W * P["Rules"]).sum() / W.sum(), 0, 1))


def style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def membership_panel(ax, x, mu_fun, names, peaks, hard_lim, hard_side, xlabel, ex_val, title):
    M = np.array([mu_fun(v) for v in x])
    lo, hi = x[0], x[-1]
    hx = (lo, hard_lim) if hard_side == "left" else (hard_lim, hi)
    ax.axvspan(*hx, color=HARD, alpha=0.35, lw=0, hatch="//", ec="white")
    ax.text(np.mean(hx), 1.08, "hard limit:\nno insulin", ha="center", va="bottom",
            fontsize=8, color=INK2)
    for i, (n, pk) in enumerate(zip(names, peaks)):
        m = M[:, i].copy()
        m[(x < hard_lim) if hard_side == "left" else (x <= hard_lim)] = np.nan
        ax.plot(x, m, color=SERIES[i], lw=2, label=n)
        ax.annotate(n, (pk, 1.0), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)          # direct label, text ink
    mu_ex = mu_fun(ex_val) if ex_val is not None else np.zeros(3)
    if ex_val is not None:
        ax.axvline(ex_val, color=INK2, lw=1, ls="--")
    firing = [i for i, m in enumerate(mu_ex) if m > 0]
    for n_i, i in enumerate(firing):
        m = mu_ex[i]
        ax.plot(ex_val, m, "o", ms=8, color=SERIES[i], mec="white", mew=2, zorder=5)
        dy = 0 if len(firing) == 1 else (9 if n_i == 0 else -9)    # keep equal values apart
        ax.annotate(f"{names[i]} = {m:.2f}", (ex_val, m), xytext=(8, dy),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.35)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("degree of membership")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="center right", frameon=False, fontsize=8.5, labelcolor=INK2)
    style(ax, title)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), facecolor="#fcfcfb")
    for ax in axes.flat:
        ax.set_facecolor("#fcfcfb")
    bex, sex = EXAMPLE if EXAMPLE is not None else (None, None)
    ex_txt = lambda name, v: f"   (example {name} = {v:g})" if v is not None else ""

    # (a) BGL memberships
    membership_panel(axes[0, 0], np.linspace(0, 27, 1081), mu_bgl,
                     ["Low", "Mid", "High"], [P["BGLLow"], P["BGLMid"], P["BGLHigh"]],
                     P["BGLLow"], "left", "BGL [mmol/L]", bex,
                     "(a) BGL sets" + ex_txt("BGL", bex))

    # (b) rate memberships
    membership_panel(axes[0, 1], np.linspace(-1.0, 1.0, 801), mu_slope,
                     ["Falling", "Steady", "Rising"], [P["SlopeFast"], 0, P["SlopeRise"]],
                     P["SlopeFast"], "left", "BGL rate [mmol/L/5min]", sex,
                     "(b) Rate sets" + ex_txt("rate", sex))

    # (c) rule table
    ax = axes[1, 0]
    R = P["Rules"]
    W = np.outer(mu_bgl(bex), mu_slope(sex)) if EXAMPLE is not None else np.zeros((3, 3))
    ax.imshow(R, cmap=CMAP, vmin=0, vmax=1, origin="upper")
    for i in range(3):
        for j in range(3):
            num = 3 * i + j + 1
            dark = R[i, j] > 0.55
            txt = f"Rule {num}\n× {R[i, j]:g}\n({(R[i, j] - 1) * 100:+.0f} %)"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="white" if dark else INK)
            if W[i, j] > 0:
                ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False,
                                       ec=SERIES[1], lw=2.5))
                ax.text(j + 0.44, i + 0.44, f"w = {W[i, j]:.2f}", ha="right", va="bottom",
                        fontsize=8, color="white" if dark else INK)
    for v in (0.5, 1.5):
        ax.axhline(v, color="#fcfcfb", lw=2)
        ax.axvline(v, color="#fcfcfb", lw=2)
    ax.set_xticks(range(3), ["Falling\n(−0.5 … 0)", "Steady\n(−0.5 … 0.5)", "Rising\n(0 … ≥0.5)"])
    ax.set_yticks(range(3), ["Low\n(4.5 … 8)", "Mid\n(4.5 … 25)", "High\n(8 … ≥25)"])
    ax.set_xlabel("BGL rate set [mmol/L/5min]")
    ax.set_ylabel("BGL set [mmol/L]")
    style(ax, "(c) Rule table: Auto Basal factor")
    for s in ax.spines.values():
        s.set_visible(False)
    note = ("Rule strength w = BGL degree × rate degree; "
            "factor = Σ(w · rule value) / Σw.\n"
            "Hard limits (BGL < 4.5 or rate ≤ −0.5) give factor 0 before the table.")
    if EXAMPLE is not None:
        f_ex = factor(bex, sex)
        terms = " + ".join(f"{W[i, j]:.2f}·{R[i, j]:g}" for i in range(3) for j in range(3) if W[i, j] > 0)
        note = (f"Example: factor = ({terms}) / {W.sum():.2f} = {f_ex:.2f}; "
                "orange boxes = rules that fire.\n" + note)
    ax.annotate(note, (0, -0.30), xycoords="axes fraction", fontsize=8.5, color=INK2, va="top")

    # (d) factor map
    ax = axes[1, 1]
    bg = np.linspace(0, 25, 501)
    sl = np.linspace(-2, 2, 401)
    F = np.array([[factor(b, s) for b in bg] for s in sl])
    hard = np.array([[(b < P["BGLLow"]) or (s <= P["SlopeFast"]) for b in bg] for s in sl])
    Fm = np.ma.masked_where(hard, F)
    ax.add_patch(Rectangle((0, -2), 25, 4, color=HARD, alpha=0.35, lw=0, hatch="//", ec="white"))
    im = ax.pcolormesh(bg, sl, Fm, cmap=CMAP, vmin=0, vmax=1, shading="auto")
    cs = ax.contour(bg, sl, Fm, levels=[0.25, 0.5, 0.75, 0.95], colors="white", linewidths=1)
    ax.clabel(cs, fmt="%.2f", fontsize=8)
    for txt, (x, y) in [("No insulin", (2.25, 1.0)), ("No insulin", (2.25, -1.25)),
                        ("No insulin", (14.5, -1.25))]:
        ax.text(x, y, txt, ha="center", va="center", fontsize=9, color=INK)
    if EXAMPLE is not None:
        ax.plot(bex, sex, "o", ms=9, color=SERIES[1], mec="white", mew=2, zorder=5)
        ax.annotate(f"example: factor {factor(bex, sex):.2f}", (bex, sex), xytext=(11, 1.4),
                    textcoords="data", fontsize=8.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.25", fc="#fcfcfb", ec="none"),
                    arrowprops=dict(arrowstyle="-", color="#fcfcfb", lw=1))
    for v in (P["BGLLow"], P["BGLMid"]):
        ax.axvline(v, color=INK2, lw=0.6, ls=":")
    for v in (P["SlopeFast"], 0, P["SlopeRise"]):
        ax.axhline(v, color=INK2, lw=0.6, ls=":")
    ax.set_xlim(0, 25)
    ax.set_ylim(-2, 2)
    ax.set_xticks([0, 4.5, 8, 12, 16, 20, 25])
    ax.set_yticks([-2, -1, -0.5, 0, 0.5, 1, 2])
    ax.set_xlabel("BGL [mmol/L]")
    ax.set_ylabel("BGL rate [mmol/L/5min]")
    style(ax, "(d) Resulting Auto Basal factor")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Auto Basal factor (1 = unchanged, 0 = suspended)", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_visible(False)

    fig.suptitle("Fuzzy scaling of the Auto Basal (adaptiveController780G, fuzzylogic_basal.m)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97), h_pad=4)
    if len(sys.argv) > 1:
        fig.savefig(sys.argv[1], dpi=200, facecolor=fig.get_facecolor())
    plt.show()


if __name__ == "__main__":
    main()