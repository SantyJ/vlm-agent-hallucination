# This file reads the results file and prints summary numbers
# It also saves a short text summary to disk
import os
import pandas as pd
from scipy.stats import mannwhitneyu

# Settings used during the analysis
RESULTS_PATH = os.path.join("data", "results.csv")
SUMMARY_PATH = os.path.join("data", "metrics_summary.txt")
CONDITIONS = [
    "baseline",
    "sabotage",
    "confident_liar",
    "gaslighting",
    "mitigation",
]
ARCHS = ["A", "B"]

# Reads the results file and cleans up the columns
df = pd.read_csv(RESULTS_PATH)
df["is_correct"] = (
    df["is_correct"].astype(str).str.lower().map({"true": 1, "false": 0})
)
df["nw_confidence"] = pd.to_numeric(df["nw_confidence"], errors="coerce")
df["jaccard_strict"] = pd.to_numeric(df["jaccard_strict"], errors="coerce")
df["jaccard_loose"] = pd.to_numeric(df["jaccard_loose"], errors="coerce")

# Helper that prints a line and also keeps it for the summary file
lines = []
def emit(line=""):
    print(line)
    lines.append(line)

# Show how often each condition was right per variant
emit("=" * 60)
emit("ACCURACY BY CONDITION AND ARCHITECTURE")
emit("=" * 60)
for arch in ARCHS:
    emit(f"\nArchitecture {arch}:")
    sub = df[df["arch_type"] == arch]
    for cond in CONDITIONS:
        rows = sub[sub["condition"] == cond]
        if len(rows) == 0:
            emit(f"  {cond:18s}: no data")
            continue
        acc = rows["is_correct"].mean()
        strict = rows["jaccard_strict"].mean()
        loose = rows["jaccard_loose"].mean()
        emit(
            f"  {cond:18s}: acc={acc:.3f}  strict={strict:.3f}  "
            f"loose={loose:.3f}  (n={len(rows)})"
        )

# How much each condition lowered accuracy compared to baseline
emit("")
emit("=" * 60)
emit("HALLUCINATION PROPAGATION RATE (HPR)")
emit("HPR = baseline_accuracy - condition_accuracy")
emit("=" * 60)
for arch in ARCHS:
    emit(f"\nArchitecture {arch}:")
    sub = df[df["arch_type"] == arch]
    base_rows = sub[sub["condition"] == "baseline"]
    if len(base_rows) == 0:
        emit("  no baseline data")
        continue
    base = base_rows["is_correct"].mean()
    for cond in CONDITIONS:
        if cond == "baseline":
            continue
        rows = sub[sub["condition"] == cond]
        if len(rows) == 0:
            emit(f"  {cond:18s}: no data")
            continue
        cond_acc = rows["is_correct"].mean()
        hpr = base - cond_acc
        emit(
            f"  {cond:18s}: HPR={hpr:+.3f}  "
            f"(base={base:.3f}, cond={cond_acc:.3f})"
        )

# Whether wrong answers came with higher NW confidence
emit("")
emit("=" * 60)
emit("CONFIDENCE BIAS SCORE (CBS)")
emit("CBS = mean(nw_conf | wrong) - mean(nw_conf | correct)")
emit("=" * 60)
wrong = df[df["is_correct"] == 0]["nw_confidence"].dropna()
correct = df[df["is_correct"] == 1]["nw_confidence"].dropna()
if len(wrong) > 0 and len(correct) > 0:
    cbs = wrong.mean() - correct.mean()
    emit(f"  mean(wrong)   = {wrong.mean():.2f}  (n={len(wrong)})")
    emit(f"  mean(correct) = {correct.mean():.2f}  (n={len(correct)})")
    emit(f"  CBS           = {cbs:+.2f}")
    if cbs > 0:
        emit("  positive: wrong decisions came with higher NW confidence")
    else:
        emit("  non-positive: confidence does not track wrongness here")
else:
    emit("  insufficient data")

# Run a statistical test on baseline versus confident liar
emit("")
emit("=" * 60)
emit("MANN-WHITNEY U: baseline vs confident_liar (is_correct)")
emit("=" * 60)
b = df[df["condition"] == "baseline"]["is_correct"].dropna()
c = df[df["condition"] == "confident_liar"]["is_correct"].dropna()
if len(b) >= 2 and len(c) >= 2:
    stat, p = mannwhitneyu(b, c, alternative="two-sided")
    emit(f"  baseline n        = {len(b)}, mean = {b.mean():.3f}")
    emit(f"  confident_liar n  = {len(c)}, mean = {c.mean():.3f}")
    emit(f"  U statistic       = {stat:.3f}")
    emit(f"  p-value           = {p:.4f}")
    emit(f"  significant @0.05 = {p < 0.05}")
else:
    emit("  insufficient data for test")

# Compare variant A and variant B for each condition
emit("")
emit("=" * 60)
emit("ARCHITECTURE COMPARISON: A vs B per condition")
emit("=" * 60)
for cond in CONDITIONS:
    a_rows = df[(df["condition"] == cond) & (df["arch_type"] == "A")]
    b_rows = df[(df["condition"] == cond) & (df["arch_type"] == "B")]
    if len(a_rows) == 0 or len(b_rows) == 0:
        emit(f"  {cond:18s}: insufficient data")
        continue
    acc_a = a_rows["is_correct"].mean()
    acc_b = b_rows["is_correct"].mean()
    if acc_a == 0:
        improv_str = "inf%" if acc_b > 0 else "0.0%"
    else:
        improv = (acc_b - acc_a) / acc_a * 100
        improv_str = f"{improv:+.1f}%"
    emit(
        f"  {cond:18s}: A={acc_a:.3f}, B={acc_b:.3f}, "
        f"B vs A improvement: {improv_str}"
    )

# Compare strict and loose Jaccard scores per condition
emit("")
emit("=" * 60)
emit("STRICT vs LOOSE JACCARD BY CONDITION")
emit("strict uses full waypoint list, loose uses landmark names only")
emit("=" * 60)
for cond in CONDITIONS:
    rows = df[df["condition"] == cond]
    if len(rows) == 0:
        emit(f"  {cond:18s}  no data")
        continue
    strict = rows["jaccard_strict"].mean()
    loose = rows["jaccard_loose"].mean()
    gap = loose - strict
    emit(
        f"  {cond:18s}  strict={strict:.3f}  loose={loose:.3f}  "
        f"gap={gap:+.3f}  (n={len(rows)})"
    )

# Same comparison split by architecture
emit("")
emit("STRICT vs LOOSE JACCARD BY CONDITION AND ARCHITECTURE")
emit("=" * 60)
for arch in ARCHS:
    emit(f"\nArchitecture {arch}:")
    sub = df[df["arch_type"] == arch]
    for cond in CONDITIONS:
        rows = sub[sub["condition"] == cond]
        if len(rows) == 0:
            emit(f"  {cond:18s}  no data")
            continue
        strict = rows["jaccard_strict"].mean()
        loose = rows["jaccard_loose"].mean()
        gap = loose - strict
        emit(
            f"  {cond:18s}  strict={strict:.3f}  loose={loose:.3f}  "
            f"gap={gap:+.3f}  (n={len(rows)})"
        )

# Save everything that was printed to a text file
os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines))
emit("")
emit(f"summary written to {SUMMARY_PATH}")
