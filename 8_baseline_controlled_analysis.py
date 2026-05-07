"""
8_baseline_controlled_analysis.py

Addresses the methodological concern:
  "Your baseline was already inaccurate — so manipulation effects might just
   be noise from already-failing cases."

Strategy: Split cities into baseline-correct vs baseline-wrong groups,
then measure HPR and Jaccard degradation *within each group separately*.

If manipulation raises HPR even in cities where the model was originally
correct, that's a direct causal rebuttal — the lying subagent corrupts
good decisions, not just noisy ones.

Outputs:
  data/analysis/baseline_controlled_hpr.png
  data/analysis/baseline_controlled_jaccard.png
  data/analysis/baseline_controlled_summary.csv
  data/analysis/baseline_controlled_stats.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu, fisher_exact

OUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "analysis")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results.csv")
STATS_PATH   = os.path.join(OUT_DIR, "baseline_controlled_stats.txt")

MANIPULATION_CONDITIONS = ["sabotage", "confident_liar", "gaslighting", "mitigation"]
CONDITION_LABELS = {
    "baseline":       "Baseline",
    "sabotage":       "Sabotage",
    "confident_liar": "Confident\nLiar",
    "gaslighting":    "Gaslighting",
    "mitigation":     "Mitigation",
}
COLORS = {
    "baseline_correct":  "#2ecc71",   # green  — cities model handled correctly
    "baseline_wrong":    "#e74c3c",   # red    — cities model already failed
}

os.makedirs(OUT_DIR, exist_ok=True)


# ── Data loading & preparation ────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(RESULTS_PATH)
    df["is_correct"] = df["is_correct"].astype(bool)
    df["jaccard_strict"] = df["jaccard_strict"].astype(float)
    df["nw_confidence"]  = df["nw_confidence"].astype(float)
    return df


def classify_cities(df):
    """
    For each city, determine if the *baseline Arch A* trial was correct.
    This is the most conservative definition: the simplest architecture
    on an unmanipulated map. Cities that fail even this are genuinely hard.
    """
    baseline_a = df[(df["condition"] == "baseline") & (df["arch_type"] == "A")].copy()
    city_status = baseline_a.set_index("city_id")["is_correct"].to_dict()
    return city_status  # city_id -> True/False


def add_group_label(df, city_status):
    df = df.copy()
    df["baseline_group"] = df["city_id"].map(city_status).map(
        {True: "baseline_correct", False: "baseline_wrong"}
    )
    return df.dropna(subset=["baseline_group"])


# ── HPR (Hallucination Path Rate) computation ─────────────────────────────────

def compute_hpr(df, condition, group):
    """
    HPR = fraction of trials where is_correct is False AND jaccard_strict < 0.1
    (near-zero Jaccard on a manipulation condition = routed through erased road).
    For baseline, HPR is expected ~0 by construction.
    """
    sub = df[(df["condition"] == condition) & (df["baseline_group"] == group) & (df["arch_type"] == "A")]
    if len(sub) == 0:
        return np.nan, 0
    hpr = ((~sub["is_correct"]) & (sub["jaccard_strict"] < 0.1)).mean()
    return hpr, len(sub)


# ── Jaccard delta computation ─────────────────────────────────────────────────

def compute_jaccard_delta(df, condition, group):
    """
    Mean strict Jaccard for (condition, group) minus mean for (baseline, group).
    Negative delta = degradation from baseline.
    """
    base = df[(df["condition"] == "baseline") & (df["baseline_group"] == group) & (df["arch_type"] == "A")]
    manip = df[(df["condition"] == condition) & (df["baseline_group"] == group) & (df["arch_type"] == "A")]
    if len(base) == 0 or len(manip) == 0:
        return np.nan
    return manip["jaccard_strict"].mean() - base["jaccard_strict"].mean()


# ── Statistical tests ─────────────────────────────────────────────────────────

def test_hpr_groups(df, condition):
    """
    Fisher's exact test: does HPR differ between baseline-correct and
    baseline-wrong cities under this condition?
    Also: Mann-Whitney U on jaccard_strict.
    """
    results = {}
    for group in ["baseline_correct", "baseline_wrong"]:
        sub = df[(df["condition"] == condition) & (df["baseline_group"] == group) & (df["arch_type"] == "A")]
        base_sub = df[(df["condition"] == "baseline") & (df["baseline_group"] == group) & (df["arch_type"] == "A")]
        results[group] = {"jaccard": sub["jaccard_strict"].values, "base_jaccard": base_sub["jaccard_strict"].values, "n": len(sub)}

    # Mann-Whitney: jaccard under condition, correct vs wrong cities
    u_stat, p_val = mannwhitneyu(
        results["baseline_correct"]["jaccard"],
        results["baseline_wrong"]["jaccard"],
        alternative="two-sided"
    )
    return u_stat, p_val, results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_hpr_by_group(df, stats_lines):
    """
    Grouped bar chart: for each manipulation condition,
    show HPR for baseline-correct vs baseline-wrong cities.
    """
    conditions = MANIPULATION_CONDITIONS
    x = np.arange(len(conditions))
    width = 0.35

    correct_hprs, wrong_hprs = [], []
    for cond in conditions:
        hc, _ = compute_hpr(df, cond, "baseline_correct")
        hw, _ = compute_hpr(df, cond, "baseline_wrong")
        correct_hprs.append(hc)
        wrong_hprs.append(hw)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_c = ax.bar(x - width/2, correct_hprs, width, label="Cities model got RIGHT at baseline",
                    color=COLORS["baseline_correct"], alpha=0.85, edgecolor="white")
    bars_w = ax.bar(x + width/2, wrong_hprs,   width, label="Cities model already got WRONG at baseline",
                    color=COLORS["baseline_wrong"],   alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=12)
    ax.set_ylabel("Hallucination Path Rate (HPR)", fontsize=12)
    ax.set_title(
        "Manipulation Effect on Baseline-Correct vs Baseline-Wrong Cities\n"
        "(Arch A only — controls for task difficulty)",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylim(0, min(1.0, max(correct_hprs + wrong_hprs) * 1.5 + 0.05))
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)

    # Annotate bar values
    for bar in bars_c:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.1%}",
                    ha="center", va="bottom", fontsize=9, color="#1a7a47")
    for bar in bars_w:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.1%}",
                    ha="center", va="bottom", fontsize=9, color="#922b1e")

    # Key finding annotation
    key_msg = (
        "Key: Green bars rising above zero means manipulation corrupts\n"
        "even cities the model originally handled correctly."
    )
    ax.text(0.02, 0.97, key_msg, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "baseline_controlled_hpr.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    stats_lines.append(f"\nHPR by group:")
    for i, cond in enumerate(conditions):
        stats_lines.append(f"  {cond:20s}  correct={correct_hprs[i]:.3f}  wrong={wrong_hprs[i]:.3f}")


def plot_jaccard_delta(df, stats_lines):
    """
    Line plot: Jaccard delta (vs baseline) for each condition,
    separately for baseline-correct and baseline-wrong cities.
    """
    conditions = MANIPULATION_CONDITIONS
    x = np.arange(len(conditions))

    correct_deltas, wrong_deltas = [], []
    for cond in conditions:
        correct_deltas.append(compute_jaccard_delta(df, cond, "baseline_correct"))
        wrong_deltas.append(compute_jaccard_delta(df, cond, "baseline_wrong"))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", label="Baseline (no change)")
    ax.plot(x, correct_deltas, "o-", color=COLORS["baseline_correct"], linewidth=2.5,
            markersize=9, label="Cities model got RIGHT at baseline", zorder=3)
    ax.plot(x, wrong_deltas,   "s--", color=COLORS["baseline_wrong"],  linewidth=2.5,
            markersize=9, label="Cities model already got WRONG at baseline", zorder=3)

    for i, (cd, wd) in enumerate(zip(correct_deltas, wrong_deltas)):
        ax.annotate(f"{cd:+.3f}", (x[i], cd), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color="#1a7a47")
        ax.annotate(f"{wd:+.3f}", (x[i], wd), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=9, color="#922b1e")

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=12)
    ax.set_ylabel("Jaccard Strict Delta vs Baseline", fontsize=12)
    ax.set_title(
        "Route Quality Degradation vs Baseline\n"
        "Separated by whether model was correct at baseline (Arch A)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "baseline_controlled_jaccard.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    stats_lines.append(f"\nJaccard delta by group:")
    for i, cond in enumerate(conditions):
        stats_lines.append(f"  {cond:20s}  correct={correct_deltas[i]:+.4f}  wrong={wrong_deltas[i]:+.4f}")


# ── Summary CSV ───────────────────────────────────────────────────────────────

def build_summary(df):
    rows = []
    for cond in ["baseline"] + MANIPULATION_CONDITIONS:
        for group in ["baseline_correct", "baseline_wrong"]:
            for arch in ["A", "B"]:
                sub = df[(df["condition"] == cond) & (df["baseline_group"] == group) & (df["arch_type"] == arch)]
                if len(sub) == 0:
                    continue
                hpr_val, _ = compute_hpr(df, cond, group) if arch == "A" else (np.nan, 0)
                rows.append({
                    "condition":       cond,
                    "baseline_group":  group,
                    "arch_type":       arch,
                    "n":               len(sub),
                    "pct_correct":     sub["is_correct"].mean(),
                    "jaccard_strict":  sub["jaccard_strict"].mean(),
                    "jaccard_loose":   sub["jaccard_loose"].mean(),
                    "mean_nw_conf":    sub["nw_confidence"].mean(),
                    "hpr_arch_a":      hpr_val if arch == "A" else np.nan,
                })
    return pd.DataFrame(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    df = load_data()

    print("Classifying cities by baseline performance...")
    city_status = classify_cities(df)
    n_correct = sum(city_status.values())
    n_wrong   = len(city_status) - n_correct
    print(f"  Baseline-correct cities: {n_correct} / {len(city_status)}")
    print(f"  Baseline-wrong cities:   {n_wrong}  / {len(city_status)}")

    df = add_group_label(df, city_status)

    stats_lines = [
        "=" * 60,
        "BASELINE-CONTROLLED ANALYSIS",
        "=" * 60,
        f"Total cities: {len(city_status)}",
        f"Baseline-correct (Arch A): {n_correct}",
        f"Baseline-wrong  (Arch A): {n_wrong}",
    ]

    # Statistical tests per manipulation condition
    stats_lines.append("\nMann-Whitney U: Jaccard under manipulation (correct vs wrong cities)")
    stats_lines.append(f"{'Condition':<20} {'U-stat':>8} {'p-value':>10}  {'Significant?':>12}")
    stats_lines.append("-" * 55)
    for cond in MANIPULATION_CONDITIONS:
        u, p, _ = test_hpr_groups(df, cond)
        sig = "YES ***" if p < 0.05 else "no"
        stats_lines.append(f"{cond:<20} {u:>8.1f} {p:>10.4f}  {sig:>12}")

    # Core rebuttal numbers
    stats_lines.append("\n" + "=" * 60)
    stats_lines.append("CORE REBUTTAL: Does manipulation hurt baseline-CORRECT cities?")
    stats_lines.append("=" * 60)
    for cond in MANIPULATION_CONDITIONS:
        hpr, n = compute_hpr(df, cond, "baseline_correct")
        delta   = compute_jaccard_delta(df, cond, "baseline_correct")
        stats_lines.append(
            f"{cond:<20}  HPR={hpr:.3f}  Jaccard_delta={delta:+.4f}  (n={n})"
        )

    stats_lines.append("\n" + "=" * 60)
    stats_lines.append("COMPARISON: Same metrics for baseline-WRONG cities")
    stats_lines.append("=" * 60)
    for cond in MANIPULATION_CONDITIONS:
        hpr, n = compute_hpr(df, cond, "baseline_wrong")
        delta   = compute_jaccard_delta(df, cond, "baseline_wrong")
        stats_lines.append(
            f"{cond:<20}  HPR={hpr:.3f}  Jaccard_delta={delta:+.4f}  (n={n})"
        )

    # Print all stats
    for line in stats_lines:
        print(line)

    # Save stats text
    with open(STATS_PATH, "w") as f:
        f.write("\n".join(stats_lines))
    print(f"\nSaved: {STATS_PATH}")

    # Plots
    print("\nGenerating plots...")
    plot_hpr_by_group(df, stats_lines)
    plot_jaccard_delta(df, stats_lines)

    # Summary CSV
    summary = build_summary(df)
    csv_path = os.path.join(OUT_DIR, "baseline_controlled_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
