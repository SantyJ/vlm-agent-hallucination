"""
9_data_quality_checks.py

Audits the generated city data and ground-truth labels for quality issues
that could explain low baseline accuracy independent of model capability.

Checks performed:
  1. Ground-truth path revalidation   — recompute shortest paths from graph,
                                        compare to stored metadata
  2. Label uniqueness                 — duplicate node labels within a city
  3. Label existence in paths         — path names reference real nodes
  4. Start/end visibility in quadrants— pixel coords, are they actually in the
                                        425×425 quadrant images the model sees?
  5. Scoring threshold sensitivity    — how many baseline trials flip correct
                                        at threshold 0.3 / 0.4 / 0.5 / 0.6?
  6. Path length distribution         — longer paths → lower achievable Jaccard
  7. Equivalent paths analysis        — many equivalents = ambiguous ground truth

Outputs:
  data/analysis/quality_report.txt
  data/analysis/quality_path_lengths.png
  data/analysis/quality_visibility.png
  data/analysis/quality_threshold_sensitivity.png
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glob import glob

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cities")
OUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "analysis")
RESULTS_PATH= os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results.csv")
REPORT_PATH = os.path.join(OUT_DIR, "quality_report.txt")

IMG_SIZE    = 800
Q_SIZE      = 425
OVERLAP     = 50
OFFSET      = Q_SIZE - OVERLAP   # 375

os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_metadata(city_id):
    path = os.path.join(DATA_DIR, f"city_{city_id:03d}_meta.json")
    with open(path) as f:
        return json.load(f)


def rebuild_graph(meta):
    """
    Reconstruct a simple weighted graph from stored node coordinates.
    We store landmark positions and shortest_path_nodes, so we can verify
    the path is connected and has the right length. We cannot fully reconstruct
    the original graph (edge list not stored), but we CAN verify:
    - path_length == len(shortest_path_nodes) - 1
    - all_shortest_paths_names[0] == shortest_path_names
    - path nodes are unique
    """
    issues = []
    sp_nodes = meta.get("shortest_path_nodes", [])
    sp_names  = meta.get("shortest_path_names", [])
    all_names = meta.get("all_shortest_paths_names", [])

    if len(sp_nodes) != len(sp_names):
        issues.append(f"  Node count mismatch: {len(sp_nodes)} coords vs {len(sp_names)} names")

    stored_len = meta.get("path_length", -1)
    actual_len = len(sp_nodes) - 1 if sp_nodes else -1
    if stored_len != actual_len:
        issues.append(f"  path_length={stored_len} but len(nodes)-1={actual_len}")

    # Check first all_shortest_paths_names matches shortest_path_names
    if all_names and all_names[0] != sp_names:
        issues.append(f"  all_shortest_paths_names[0] != shortest_path_names")

    # Check path nodes are unique (no cycles)
    if len(sp_nodes) != len(set(map(tuple, sp_nodes))):
        issues.append(f"  shortest_path_nodes contains duplicate coordinates (cycle?)")

    return issues


# ── Check 1: Ground-truth path consistency ────────────────────────────────────

def check_gt_consistency(meta):
    """Verify internal consistency of stored ground-truth paths."""
    issues = []
    issues.extend(rebuild_graph(meta))

    # All equivalent paths must have same length
    all_sp = meta.get("all_shortest_paths_names", [])
    if all_sp:
        lengths = [len(p) for p in all_sp]
        if len(set(lengths)) > 1:
            issues.append(f"  Inequivalent path lengths: {set(lengths)} — some 'equivalent' paths are different lengths")

    # Start and end must appear in every path
    start = meta.get("start")
    end   = meta.get("end")
    for i, p in enumerate(all_sp):
        if p and p[0] != start:
            issues.append(f"  Path {i}: starts with '{p[0]}' not '{start}'")
        if p and p[-1] != end:
            issues.append(f"  Path {i}: ends with '{p[-1]}' not '{end}'")

    return issues


# ── Check 2: Label uniqueness ─────────────────────────────────────────────────

def check_label_uniqueness(meta):
    """No two nodes should share a label within a city."""
    issues = []
    # All labels that appear in paths
    all_labels = []
    for path in meta.get("all_shortest_paths_names", []):
        all_labels.extend(path)
    # Also check landmark keys
    landmarks = list(meta.get("landmarks", {}).keys())

    seen = {}
    for lbl in all_labels:
        seen[lbl] = seen.get(lbl, 0) + 1

    # Labels appearing in more than one distinct path position aren't duplicates,
    # they're just on multiple paths — that's fine. Real duplicate = same label
    # used for two different physical nodes. We detect this via landmarks dict.
    lm_names = list(meta.get("landmarks", {}).keys())
    if len(lm_names) != len(set(lm_names)):
        dupes = [n for n in lm_names if lm_names.count(n) > 1]
        issues.append(f"  Duplicate landmark names: {set(dupes)}")

    return issues


# ── Check 3: Label existence in paths ────────────────────────────────────────

def check_label_existence(meta):
    """All labels in paths should be either a landmark name or iXX/dXX format."""
    import re
    issues = []
    valid_landmark_names = set(meta.get("landmarks", {}).keys())
    node_pattern = re.compile(r'^[id]\d{2}$')

    for path in meta.get("all_shortest_paths_names", []):
        for lbl in path:
            if lbl not in valid_landmark_names and not node_pattern.match(lbl):
                issues.append(f"  Unexpected label format: '{lbl}'")

    return issues


# ── Check 4: Start/end visibility in quadrant images ─────────────────────────

def node_to_pixel(node_coords):
    """Mirror of city_generator's node_to_pixel."""
    margin    = 50
    img_size  = 800
    cell_size = (img_size - 2 * margin) / 9
    x, y = node_coords
    return (margin + x * cell_size, margin + y * cell_size)


def pixel_in_quadrant(px, py, quadrant):
    """Check if pixel (px, py) falls within the given quadrant crop."""
    crops = {
        'NW': (0,      0,      Q_SIZE,  Q_SIZE),
        'NE': (OFFSET, 0,      IMG_SIZE,Q_SIZE),
        'SW': (0,      OFFSET, Q_SIZE,  IMG_SIZE),
        'SE': (OFFSET, OFFSET, IMG_SIZE,IMG_SIZE),
    }
    x0, y0, x1, y1 = crops[quadrant]
    return x0 <= px < x1 and y0 <= py < y1


def check_endpoint_visibility(meta):
    """
    Verify that start and end landmarks are visible in at least one quadrant.
    Also checks that all path waypoints collectively span the quadrants.
    """
    issues = []
    landmarks = meta.get("landmarks", {})   # name -> [x, y]
    start = meta.get("start")
    end   = meta.get("end")

    for name, role in [(start, "start"), (end, "end")]:
        if name not in landmarks:
            issues.append(f"  {role} landmark '{name}' not found in landmarks dict")
            continue
        coords = landmarks[name]
        px, py = node_to_pixel(coords)
        visible_in = [q for q in ["NW", "NE", "SW", "SE"] if pixel_in_quadrant(px, py, q)]
        if not visible_in:
            issues.append(f"  {role} '{name}' at pixel ({px:.0f},{py:.0f}) NOT visible in any quadrant!")
        # Note but don't flag as error — overlap means nodes can be in 2 quadrants
        elif len(visible_in) == 1:
            pass  # normal
        # visible in 2 is fine (overlap zone)

    return issues, {name: [q for q in ["NW","NE","SW","SE"]
                            if pixel_in_quadrant(*node_to_pixel(landmarks[name]), q)]
                    for name in [start, end] if name in landmarks}


# ── Check 5: Scoring threshold sensitivity ───────────────────────────────────

def threshold_sensitivity(df_baseline):
    """
    For each threshold in [0.3, 0.4, 0.5, 0.6, 0.7], compute % correct (Arch A baseline).
    Shows how sensitive 'accuracy' is to the chosen threshold.
    """
    results = {}
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pct = (df_baseline["jaccard_loose"] > t).mean()
        results[t] = pct
    return results


# ── Check 6: Path lengths vs Jaccard score ───────────────────────────────────

def analyze_path_length_effect(df_baseline, metas):
    """
    Longer ground truth paths → harder to achieve high Jaccard even if close.
    Plot path_length vs mean jaccard_strict for baseline Arch A.
    """
    city_len = {m["city_id"]: m["path_length"] for m in metas}
    df = df_baseline[df_baseline["arch_type"] == "A"].copy()
    df["path_length"] = df["city_id"].map(city_len)
    return df[["city_id", "path_length", "jaccard_strict", "jaccard_loose", "is_correct"]]


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_path_length_effect(df_len, report_lines):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter: path length vs jaccard_strict
    ax = axes[0]
    jitter = np.random.uniform(-0.15, 0.15, len(df_len))
    ax.scatter(df_len["path_length"] + jitter, df_len["jaccard_strict"],
               alpha=0.4, s=20, color="#3498db")
    # Trend line
    coeffs = np.polyfit(df_len["path_length"].fillna(0), df_len["jaccard_strict"], 1)
    xs = np.linspace(df_len["path_length"].min(), df_len["path_length"].max(), 100)
    ax.plot(xs, np.polyval(coeffs, xs), "r--", linewidth=1.5, label=f"trend (slope={coeffs[0]:.3f})")
    ax.set_xlabel("Ground Truth Path Length (# edges)", fontsize=11)
    ax.set_ylabel("Jaccard Strict (baseline Arch A)", fontsize=11)
    ax.set_title("Path Length vs Route Accuracy", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Box plot by path length bucket
    ax = axes[1]
    df_len["len_bucket"] = pd.cut(df_len["path_length"], bins=[0,4,6,8,12,20],
                                   labels=["≤4","5-6","7-8","9-12",">12"])
    groups = [df_len[df_len["len_bucket"] == b]["jaccard_strict"].dropna()
              for b in ["≤4","5-6","7-8","9-12",">12"]]
    ax.boxplot(groups, labels=["≤4","5-6","7-8","9-12",">12"], patch_artist=True,
               boxprops=dict(facecolor="#aed6f1"))
    ax.set_xlabel("Path Length Bucket", fontsize=11)
    ax.set_ylabel("Jaccard Strict", fontsize=11)
    ax.set_title("Jaccard by Path Length Bucket", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Effect of Ground Truth Path Length on Scoring", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "quality_path_lengths.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # Stats
    corr = df_len["path_length"].corr(df_len["jaccard_strict"])
    report_lines.append(f"\n[CHECK 6] Path length vs Jaccard correlation: r = {corr:.4f}")
    report_lines.append(f"  Mean path length: {df_len['path_length'].mean():.2f}  "
                        f"(min={df_len['path_length'].min()}, max={df_len['path_length'].max()})")
    grp_means = df_len.groupby("len_bucket")["jaccard_strict"].mean()
    for bucket, mean in grp_means.items():
        report_lines.append(f"  Length {bucket}: mean Jaccard = {mean:.3f}")


def plot_threshold_sensitivity(thresholds, report_lines):
    fig, ax = plt.subplots(figsize=(8, 5))
    ts = list(thresholds.keys())
    vs = [thresholds[t] * 100 for t in ts]
    bars = ax.bar([str(t) for t in ts], vs, color="#5dade2", edgecolor="white", alpha=0.85)
    ax.axhline(50, color="red", linestyle="--", linewidth=1, label="50% line")
    for bar, v in zip(bars, vs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=11)
    ax.set_xlabel("Jaccard Loose Threshold for 'Correct'", fontsize=12)
    ax.set_ylabel("% Trials Marked Correct (Baseline Arch A)", fontsize=12)
    ax.set_title("Threshold Sensitivity: How the Accuracy Number Changes\nwith Different 'Correct' Cutoffs",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(vs) * 1.2)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "quality_threshold_sensitivity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    report_lines.append("\n[CHECK 5] Threshold sensitivity (% correct, baseline Arch A):")
    for t, v in thresholds.items():
        flag = " ← current" if t == 0.5 else ""
        report_lines.append(f"  threshold={t}  →  {v*100:.1f}% correct{flag}")


def plot_visibility_heatmap(visibility_data, report_lines):
    """
    For each city, show which quadrant(s) the start/end landmarks fall in.
    Summarise as a count matrix.
    """
    start_quad_counts = {"NW": 0, "NE": 0, "SW": 0, "SE": 0}
    end_quad_counts   = {"NW": 0, "NE": 0, "SW": 0, "SE": 0}
    both_same_quad    = 0
    neither_visible   = 0

    for cid, (sv, ev) in visibility_data.items():
        for q in sv: start_quad_counts[q] += 1
        for q in ev: end_quad_counts[q]   += 1
        # Start and end in same quadrant = easy; different = must integrate
        if set(sv) & set(ev):
            both_same_quad += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    quads = ["NW", "NE", "SW", "SE"]

    for ax, counts, title in [
        (axes[0], start_quad_counts, "Start Landmark Quadrant Distribution"),
        (axes[1], end_quad_counts,   "End Landmark Quadrant Distribution"),
    ]:
        vals = [counts[q] for q in quads]
        bars = ax.bar(quads, vals, color=["#3498db","#e74c3c","#2ecc71","#f39c12"],
                      edgecolor="white", alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, str(v),
                    ha="center", va="bottom", fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of cities", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Where Start/End Landmarks Fall in Quadrant Images\n"
                 "(nodes in overlap zone counted in both quadrants)", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "quality_visibility.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    report_lines.append(f"\n[CHECK 4] Start/end landmark quadrant visibility:")
    report_lines.append(f"  Start quadrant distribution: {start_quad_counts}")
    report_lines.append(f"  End   quadrant distribution: {end_quad_counts}")
    report_lines.append(f"  Cities where start & end share a quadrant: {both_same_quad} / {len(visibility_data)}")
    report_lines.append(f"  (Note: overlap zone = 50px; nodes there appear in 2 quadrants)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    report_lines = [
        "=" * 65,
        "DATA QUALITY AUDIT REPORT",
        "=" * 65,
    ]

    # Load all metadata
    meta_files = sorted(glob(os.path.join(DATA_DIR, "city_*_meta.json")))
    metas = []
    for mf in meta_files:
        with open(mf) as f:
            metas.append(json.load(f))
    print(f"Loaded {len(metas)} city metadata files.")
    report_lines.append(f"Cities audited: {len(metas)}")

    # ── Check 1: GT consistency ───────────────────────────────────────────────
    report_lines.append("\n" + "─" * 65)
    report_lines.append("[CHECK 1] Ground-truth path internal consistency")
    report_lines.append("─" * 65)
    gt_issues = {}
    for meta in metas:
        issues = check_gt_consistency(meta)
        if issues:
            gt_issues[meta["city_id"]] = issues
    if gt_issues:
        for cid, issues in gt_issues.items():
            report_lines.append(f"  City {cid:03d}:")
            report_lines.extend(issues)
    else:
        report_lines.append("  PASS — all stored paths are internally consistent.")
    print(f"[1] GT consistency: {len(gt_issues)} cities with issues")

    # ── Check 2: Label uniqueness ─────────────────────────────────────────────
    report_lines.append("\n[CHECK 2] Label uniqueness within each city")
    lbl_issues = {}
    for meta in metas:
        issues = check_label_uniqueness(meta)
        if issues:
            lbl_issues[meta["city_id"]] = issues
    if lbl_issues:
        for cid, issues in lbl_issues.items():
            report_lines.append(f"  City {cid:03d}: {issues}")
    else:
        report_lines.append("  PASS — no duplicate landmark names detected.")
    print(f"[2] Label uniqueness: {len(lbl_issues)} cities with issues")

    # ── Check 3: Label existence ──────────────────────────────────────────────
    report_lines.append("\n[CHECK 3] Label format validity in paths")
    fmt_issues = {}
    for meta in metas:
        issues = check_label_existence(meta)
        if issues:
            fmt_issues[meta["city_id"]] = issues
    if fmt_issues:
        for cid, issues in fmt_issues.items():
            report_lines.append(f"  City {cid:03d}: {issues}")
    else:
        report_lines.append("  PASS — all path labels match expected formats.")
    print(f"[3] Label format: {len(fmt_issues)} cities with issues")

    # ── Check 4: Endpoint visibility ──────────────────────────────────────────
    report_lines.append("\n[CHECK 4] Start/end landmark quadrant visibility")
    visibility_issues = {}
    visibility_data   = {}
    for meta in metas:
        issues, vis = check_endpoint_visibility(meta)
        visibility_data[meta["city_id"]] = (
            vis.get(meta.get("start"), []),
            vis.get(meta.get("end"),   []),
        )
        if issues:
            visibility_issues[meta["city_id"]] = issues
    if visibility_issues:
        for cid, issues in visibility_issues.items():
            report_lines.append(f"  City {cid:03d}:")
            report_lines.extend(issues)
    else:
        report_lines.append("  PASS — all start/end landmarks visible in at least one quadrant.")
    print(f"[4] Endpoint visibility: {len(visibility_issues)} cities with issues")
    plot_visibility_heatmap(visibility_data, report_lines)

    # ── Check 5: Scoring threshold sensitivity ────────────────────────────────
    report_lines.append("\n[CHECK 5] Scoring threshold sensitivity")
    df = pd.read_csv(RESULTS_PATH)
    df["is_correct"]    = df["is_correct"].astype(bool)
    df["jaccard_strict"]= df["jaccard_strict"].astype(float)
    df["jaccard_loose"] = df["jaccard_loose"].astype(float)
    df_baseline = df[df["condition"] == "baseline"]
    thresholds = threshold_sensitivity(df_baseline)
    plot_threshold_sensitivity(thresholds, report_lines)

    # ── Check 6: Path length effect ───────────────────────────────────────────
    report_lines.append("\n[CHECK 6] Path length vs scoring")
    df_len = analyze_path_length_effect(df_baseline, metas)
    plot_path_length_effect(df_len, report_lines)

    # ── Check 7: Equivalent paths analysis ───────────────────────────────────
    report_lines.append("\n[CHECK 7] Equivalent paths analysis")
    num_equiv = [m["num_equivalent_paths"] for m in metas]
    report_lines.append(f"  Min equivalent paths: {min(num_equiv)}")
    report_lines.append(f"  Max equivalent paths: {max(num_equiv)}")
    report_lines.append(f"  Mean: {np.mean(num_equiv):.1f}  Median: {np.median(num_equiv):.0f}")
    report_lines.append(f"  Cities with >10 equivalent paths: "
                        f"{sum(1 for n in num_equiv if n > 10)}")
    report_lines.append(f"  Note: scoring uses best-of-all-equivalents Jaccard, so")
    report_lines.append(f"  more equivalent paths = more lenient scoring, not stricter.")

    # ── Check 8: Spot-check worst baseline cities ─────────────────────────────
    report_lines.append("\n[CHECK 8] Worst baseline cities (Jaccard < 0.1, Arch A)")
    worst = df[(df["condition"] == "baseline") & (df["arch_type"] == "A")
               & (df["jaccard_strict"] < 0.1)].sort_values("jaccard_strict")
    report_lines.append(f"  Count: {len(worst)} trials out of "
                        f"{len(df[(df['condition']=='baseline') & (df['arch_type']=='A')])}")
    # Spot-check: look at their ground truth path lengths
    city_len = {m["city_id"]: m["path_length"] for m in metas}
    city_equiv= {m["city_id"]: m["num_equivalent_paths"] for m in metas}
    worst_sample = worst.head(10)
    for _, row in worst_sample.iterrows():
        cid = row["city_id"]
        report_lines.append(
            f"  City {cid:03d}: jaccard_strict={row['jaccard_strict']:.3f}  "
            f"jaccard_loose={row['jaccard_loose']:.3f}  "
            f"path_len={city_len.get(cid,'?')}  "
            f"equiv_paths={city_equiv.get(cid,'?')}"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    report_lines.append("\n" + "=" * 65)
    report_lines.append("SUMMARY")
    report_lines.append("=" * 65)
    total_issues = len(gt_issues) + len(lbl_issues) + len(fmt_issues) + len(visibility_issues)
    report_lines.append(f"  Structural data issues found: {total_issues} cities")
    report_lines.append(f"  Ground truth paths: {'PROBLEMS FOUND' if gt_issues else 'clean'}")
    report_lines.append(f"  Label uniqueness:   {'PROBLEMS FOUND' if lbl_issues else 'clean'}")
    report_lines.append(f"  Label formats:      {'PROBLEMS FOUND' if fmt_issues else 'clean'}")
    report_lines.append(f"  Endpoint visibility:{'PROBLEMS FOUND' if visibility_issues else 'clean'}")
    report_lines.append(f"")
    report_lines.append(f"  Threshold at 0.3: {thresholds[0.3]*100:.1f}% correct  (generous)")
    report_lines.append(f"  Threshold at 0.5: {thresholds[0.5]*100:.1f}% correct  (current)")
    report_lines.append(f"  Threshold at 0.7: {thresholds[0.7]*100:.1f}% correct  (strict)")
    report_lines.append(f"")
    report_lines.append(f"  Path length correlation with Jaccard: see quality_path_lengths.png")
    report_lines.append(f"  If correlation is strongly negative, low baseline accuracy is")
    report_lines.append(f"  partly a scoring artefact of long paths, not model failure.")

    # Write report
    for line in report_lines:
        print(line)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nFull report saved to: {REPORT_PATH}")
    print(f"Plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
