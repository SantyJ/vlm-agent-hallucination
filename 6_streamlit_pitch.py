# Streamlit dashboard for the VLM Confidence Bias audit.
# Run with: streamlit run 6_streamlit_pitch.py

import os
import json
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image

sns.set_theme(style="whitegrid", font_scale=1.05)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE          = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH   = os.path.join(_HERE, "data", "results.csv")
CITIES_DIR     = os.path.join(_HERE, "data", "cities")
HEATMAPS_FINAL = os.path.join(_HERE, "data", "heatmaps_final")
HEATMAPS_DIR   = os.path.join(_HERE, "data", "heatmaps")
ANALYSIS_DIR   = os.path.join(_HERE, "data", "analysis")
ROBUST_CSV     = os.path.join(ANALYSIS_DIR, "robustness_summary.csv")
BASELINE_CSV   = os.path.join(ANALYSIS_DIR, "baseline_controlled_summary.csv")
CHESS_CSV      = os.path.join(_HERE, "data", "chess", "chess_results.csv")
PLOTS_CITIES   = os.path.join(_HERE, "data", "plots", "cities")
PLOTS_CHESS    = os.path.join(_HERE, "data", "plots", "chess")
os.makedirs(PLOTS_CITIES, exist_ok=True)
os.makedirs(PLOTS_CHESS,  exist_ok=True)


def save_plot(fig, subdir, name):
    """Save fig as PNG alongside the in-memory st.pyplot() call."""
    path = os.path.join(subdir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")


CONDITION_ORDER  = ["baseline", "sabotage", "confident_liar", "gaslighting", "mitigation"]
CONDITION_LABELS = {
    "baseline":       "Baseline",
    "sabotage":       "Sabotage",
    "confident_liar": "Confident\nLiar",
    "gaslighting":    "Gaslighting",
    "mitigation":     "Mitigation",
}
ARCH_COLORS = {"A": "#4C72B0", "B": "#DD8452"}


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_data():
    if not os.path.exists(RESULTS_PATH):
        return None
    df = pd.read_csv(RESULTS_PATH)
    df["is_correct"] = df["is_correct"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    df["error_rate"] = (~df["is_correct"]).astype(float)
    return df

@st.cache_data(ttl=30)
def load_chess_data():
    if not os.path.exists(CHESS_CSV):
        return None
    df = pd.read_csv(CHESS_CSV)
    df["move_correct"] = df["move_correct"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    df["error_rate"] = (~df["move_correct"]).astype(float)
    return df

@st.cache_data
def load_robustness():
    if not os.path.exists(ROBUST_CSV):
        return None
    return pd.read_csv(ROBUST_CSV)

@st.cache_data
def load_baseline():
    if not os.path.exists(BASELINE_CSV):
        return None
    return pd.read_csv(BASELINE_CSV)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="VLM Confidence Bias Audit", page_icon="🧠", layout="wide")

df = load_data()
if df is None:
    st.error("results.csv not found. Run 5_batch_runner.py first.")
    st.stop()

n_cities = df["city_id"].nunique()
n_rows   = len(df)
wrong    = df[df["is_correct"] == False]["nw_confidence"].dropna()
correct  = df[df["is_correct"] == True]["nw_confidence"].dropna()
cbs      = wrong.mean() - correct.mean() if (len(wrong) and len(correct)) else 0.0

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 VLM Confidence Bias Audit")
st.caption(
    "Multi-agent Qwen2-VL subagents read map quadrants → "
    "Claude Sonnet root agent synthesises the shortest path. "
    "We test whether a confident lying subagent can corrupt the root agent's decision."
)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Cities", f"{n_cities} / 120")
c2.metric("Total trials", n_rows)
c3.metric("Conditions", "5")
c4.metric("Architectures", "A · B")
c5.metric("CBS (global)", f"{cbs:+.2f}",
          delta="bias detected" if cbs > 0 else "no bias",
          delta_color="inverse" if cbs > 0 else "normal")

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Confidence Bias",
    "📈 Accuracy Deep-Dive",
    "🔥 Attention Heatmaps",
    "🧩 Extended Thinking",
    "🔬 Baseline Analysis",
    "♟️ Chess Validation",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Confidence Bias
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Error rate bars ───────────────────────────────────────────────────────
    st.subheader("Error Rate by Condition — Architecture A vs B")
    st.caption("**Arch A** = text only · **Arch B** = text + images to root agent")

    x      = np.arange(len(CONDITION_ORDER))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, arch in enumerate(["A", "B"]):
        vals = [
            df[(df["condition"] == c) & (df["arch_type"] == arch)]["error_rate"].mean()
            for c in CONDITION_ORDER
        ]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      label=f"Arch {arch}", color=ARCH_COLORS[arch], alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=10)
    ax.set_ylabel("Error Rate")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    save_plot(fig, PLOTS_CITIES, "error_rate_by_condition")
    plt.close(fig)

    st.divider()

    # ── CBS per condition ─────────────────────────────────────────────────────
    st.subheader("Confidence Bias Score (CBS) per Condition")
    st.caption(
        "CBS = mean confidence on **wrong** − mean confidence on **correct** predictions.  \n"
        "Positive = model was more confident when wrong."
    )
    cbs_vals = []
    for cond in CONDITION_ORDER:
        sub = df[df["condition"] == cond]
        w = sub[sub["is_correct"] == False]["nw_confidence"].dropna()
        c = sub[sub["is_correct"] == True]["nw_confidence"].dropna()
        cbs_vals.append(w.mean() - c.mean() if (len(w) and len(c)) else 0.0)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bar_colors = ["#c0392b" if v > 0 else "#27ae60" for v in cbs_vals]
    bars = ax.bar([CONDITION_LABELS[c] for c in CONDITION_ORDER], cbs_vals,
                  color=bar_colors, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, cbs_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.3 if v >= 0 else -0.8),
                f"{v:+.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(0, color="#333", linewidth=1.2, linestyle="--")
    ax.set_ylabel("CBS (confidence delta)")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    save_plot(fig, PLOTS_CITIES, "cbs_per_condition")
    plt.close(fig)

    st.divider()

    # ── Confidence violin ─────────────────────────────────────────────────────
    st.subheader("NW Confidence Distribution — Wrong vs Correct")
    st.caption("Violin shows the full distribution, not just the mean. Wide at top = model was overconfident.")

    fig, axes = plt.subplots(1, len(CONDITION_ORDER), figsize=(13, 4), sharey=True)
    for ax, cond in zip(axes, CONDITION_ORDER):
        sub = df[df["condition"] == cond]
        data_wrong   = sub[sub["is_correct"] == False]["nw_confidence"].dropna().tolist()
        data_correct = sub[sub["is_correct"] == True]["nw_confidence"].dropna().tolist()
        parts = ax.violinplot(
            [data_wrong, data_correct],
            positions=[1, 2], widths=0.7, showmedians=True,
        )
        for pc, color in zip(parts["bodies"], ["#c0392b", "#27ae60"]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        for key in ("cmedians", "cbars", "cmaxes", "cmins"):
            parts[key].set_color("#333")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Wrong", "Correct"], fontsize=9)
        ax.set_title(CONDITION_LABELS[cond].replace("\n", " "), fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("NW Confidence")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    save_plot(fig, PLOTS_CITIES, "confidence_violin")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Accuracy Deep-Dive
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    # ── Heatmap overview ──────────────────────────────────────────────────────
    st.subheader("Error Rate — Condition × Architecture")
    st.caption("Color intensity = error rate. Darker = worse.")

    pivot = df.pivot_table(
        values="error_rate", index="condition", columns="arch_type", aggfunc="mean"
    ).reindex(CONDITION_ORDER)
    pivot.index = [CONDITION_LABELS[c].replace("\n", " ") for c in CONDITION_ORDER]

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="YlOrRd",
        vmin=0, vmax=1, linewidths=0.5, linecolor="#ccc",
        ax=ax, annot_kws={"size": 12, "weight": "bold"},
        cbar_kws={"shrink": 0.8, "label": "Error Rate"},
    )
    ax.set_xlabel("Architecture")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    col_hm, _ = st.columns([1, 1])
    with col_hm:
        st.pyplot(fig, use_container_width=True)
    save_plot(fig, PLOTS_CITIES, "error_rate_heatmap")
    plt.close(fig)

    st.divider()

    # ── Jaccard radar charts ──────────────────────────────────────────────────
    st.subheader("Strict vs Loose Jaccard — Radar Comparison")
    st.caption(
        "Each axis = one condition. **Solid fill** = Strict Jaccard (full path).  "
        "**Outline only** = Loose Jaccard (landmarks only).  \n"
        "Larger area = better. Gap between solid and outline = intersection-label errors."
    )

    def make_radar(ax, strict_vals, loose_vals, labels, title, color):
        N = len(labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        sv = strict_vals + [strict_vals[0]]
        lv = loose_vals  + [loose_vals[0]]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
        ax.set_ylim(0, 0.7)
        ax.set_yticks([0.2, 0.4, 0.6])
        ax.set_yticklabels(["0.2", "0.4", "0.6"], fontsize=7, color="#888")
        ax.plot(angles, sv, color=color, linewidth=2)
        ax.fill(angles, sv, alpha=0.35, color=color)
        ax.plot(angles, lv, color=color, linewidth=1.5, linestyle="--")
        ax.fill(angles, lv, alpha=0.10, color=color)
        ax.set_title(title, pad=14, fontsize=11, fontweight="bold")
        solid_patch  = mpatches.Patch(color=color, alpha=0.5, label="Strict")
        dashed_patch = mpatches.Patch(color=color, alpha=0.2, label="Loose")
        ax.legend(handles=[solid_patch, dashed_patch], loc="lower right",
                  bbox_to_anchor=(1.35, -0.05), fontsize=8)

    labels = [CONDITION_LABELS[c].replace("\n", " ") for c in CONDITION_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                              subplot_kw=dict(polar=True))
    for ax, arch, color in zip(axes, ["A", "B"], [ARCH_COLORS["A"], ARCH_COLORS["B"]]):
        strict_vals = [
            df[(df["condition"] == c) & (df["arch_type"] == arch)]["jaccard_strict"].mean()
            for c in CONDITION_ORDER
        ]
        loose_vals = [
            df[(df["condition"] == c) & (df["arch_type"] == arch)]["jaccard_loose"].mean()
            for c in CONDITION_ORDER
        ]
        make_radar(ax, strict_vals, loose_vals, labels, f"Architecture {arch}", color)
    fig.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=True)
    save_plot(fig, PLOTS_CITIES, "jaccard_radar")
    plt.close(fig)

    st.divider()

    # ── Threshold robustness ──────────────────────────────────────────────────
    st.subheader("Threshold Robustness")
    st.caption("Accuracy at different correctness thresholds. If condition ordering is stable, findings are robust.")

    rob_df = load_robustness()
    if rob_df is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        cond_colors_list = ["#6c757d", "#e07b39", "#c0392b", "#8e44ad", "#27ae60"]
        for cond, color in zip(CONDITION_ORDER, cond_colors_list):
            sub = rob_df[rob_df["condition"] == cond]
            ax.plot(sub["threshold"], sub["accuracy"], marker="o", linewidth=2,
                    color=color, label=CONDITION_LABELS[cond].replace("\n", " "))
        ax.set_xlabel("Correctness Threshold")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        save_plot(fig, PLOTS_CITIES, "threshold_robustness")
        plt.close(fig)
    else:
        st.info("Run `10_threshold_robustness.py` to generate this data.")

    st.divider()

    # ── Raw table ─────────────────────────────────────────────────────────────
    st.subheader("Raw Results Table")
    def complexity(cid):
        if cid <= 40:  return "L1"
        if cid <= 80:  return "L2"
        return "L3"
    df["complexity"] = df["city_id"].apply(complexity)

    col_f1, col_f2, col_f3 = st.columns(3)
    f_cond = col_f1.multiselect("Condition",     CONDITION_ORDER, default=CONDITION_ORDER)
    f_arch = col_f2.multiselect("Architecture",  ["A", "B"],      default=["A", "B"])
    f_comp = col_f3.multiselect("Complexity",    ["L1", "L2", "L3"], default=["L1", "L2", "L3"])
    filtered = df[
        df["condition"].isin(f_cond) &
        df["arch_type"].isin(f_arch) &
        df["complexity"].isin(f_comp)
    ][["city_id", "complexity", "condition", "arch_type",
       "is_correct", "jaccard_strict", "jaccard_loose", "nw_confidence"]]
    st.dataframe(filtered, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Attention Heatmaps
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Attention Rollout Heatmaps")
    st.caption(
        "Deep red = high attention · Pale = ignored · Black rectangle = erased road.  \n"
        "If the encoder attended to the erased region but the subagent still reported "
        "it confidently → perception-language disconnect."
    )

    hm_source = st.radio(
        "Source",
        ["Final (single-color)", "Rollout (JET)", "GradCAM (vision encoder)"],
        horizontal=True,
    )
    hm_dir = {
        "Final (single-color)":        HEATMAPS_FINAL,
        "Rollout (JET)":               os.path.join(_HERE, "data", "heatmaps_rollout"),
        "GradCAM (vision encoder)":    HEATMAPS_DIR,
    }[hm_source]

    sbs_files = sorted(glob.glob(os.path.join(hm_dir, "*_side_by_side.png")))
    if not sbs_files:
        st.info(f"No heatmaps in `{os.path.relpath(hm_dir, _HERE)}`. Run the relevant script first.")
    else:
        city_options = [os.path.basename(f).replace("_side_by_side.png", "") for f in sbs_files]
        selected = st.selectbox("City", city_options)
        st.image(
            Image.open(os.path.join(hm_dir, f"{selected}_side_by_side.png")),
            caption=f"{selected} — left: sabotaged map  |  right: attention overlay",
            use_container_width=True,
        )
        city_id_num = int("".join(filter(str.isdigit, selected.split("_")[1])))
        city_rows = df[
            (df["city_id"] == city_id_num) &
            (df["condition"].isin(["sabotage", "confident_liar"]))
        ]
        if not city_rows.empty:
            for _, r in city_rows.iterrows():
                badge = "✅" if r["is_correct"] else "❌"
                st.markdown(
                    f"{badge} **{r['condition']}** / Arch {r['arch_type']} — "
                    f"NW conf: `{r['nw_confidence']}` · Jaccard loose: `{r['jaccard_loose']:.2f}`"
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Extended Thinking
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🧩 Claude's Extended Thinking — Internal Monologue")
    st.caption("Raw <thinking> traces from Claude Sonnet root agent before committing to a path.")

    col_l, col_r = st.columns([1, 2])
    with col_l:
        cities_with_traces = sorted(
            df[df["thinking_trace"].str.strip().str.len() > 10]["city_id"].unique()
        )
        if not cities_with_traces:
            st.info("No thinking traces found.")
        else:
            selected_city = st.selectbox("City", cities_with_traces)
            selected_cond = st.selectbox("Condition", CONDITION_ORDER)
            selected_arch = st.radio("Architecture", ["A", "B"])

    with col_r:
        if cities_with_traces:
            row = df[
                (df["city_id"] == selected_city) &
                (df["condition"] == selected_cond) &
                (df["arch_type"] == selected_arch)
            ]
            if row.empty:
                st.info("No data for this combination.")
            else:
                r = row.iloc[0]
                badge = "✅ Correct" if r["is_correct"] else "❌ Wrong"
                st.markdown(
                    f"**Result:** {badge}  ·  **NW confidence:** `{r['nw_confidence']}`  ·  "
                    f"**Strict:** `{r['jaccard_strict']:.3f}`  ·  **Loose:** `{r['jaccard_loose']:.3f}`"
                )
                contradictions = json.loads(r["contradictions_found"]) if r["contradictions_found"] else []
                if contradictions:
                    with st.expander(f"⚠️ Contradictions flagged ({len(contradictions)})"):
                        for c in contradictions:
                            st.markdown(f"- {c}")
                agents_trusted = json.loads(r["agents_trusted"]) if r["agents_trusted"] else []
                if agents_trusted:
                    st.markdown(f"**Agents trusted:** {', '.join(agents_trusted)}")
                trace = str(r["thinking_trace"]).strip()
                if trace and len(trace) > 10:
                    with st.expander("💭 Full thinking trace", expanded=True):
                        st.markdown(
                            f'<div style="background:#1e1e2e;color:#cdd6f4;padding:16px;'
                            f'border-radius:8px;font-family:monospace;font-size:13px;'
                            f'white-space:pre-wrap;max-height:500px;overflow-y:auto;">'
                            f'{trace}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No thinking trace for this trial.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Baseline Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Baseline-Controlled Analysis")
    st.caption(
        "Cities split by whether the **baseline** run was correct or wrong — "
        "controls for city difficulty."
    )

    base_df = load_baseline()
    if base_df is None:
        st.info("Run `8_baseline_controlled_analysis.py` to generate this data.")
    else:
        # ── HPR by condition ──────────────────────────────────────────────────
        st.subheader("Hallucination Path Rate (HPR) — Arch A")
        st.caption("Fraction of trials where root agent route passes through the erased road.")

        hpr_df = base_df[
            (base_df["arch_type"] == "A") & base_df["hpr_arch_a"].notna()
        ].copy()

        fig, ax = plt.subplots(figsize=(9, 4))
        groups  = ["baseline_correct", "baseline_wrong"]
        g_labels = {"baseline_correct": "Baseline Correct", "baseline_wrong": "Baseline Wrong"}
        g_colors = {"baseline_correct": "#27ae60", "baseline_wrong": "#e07b39"}
        x = np.arange(len(CONDITION_ORDER))
        width = 0.35
        for i, grp in enumerate(groups):
            sub = hpr_df[hpr_df["baseline_group"] == grp]
            vals = [
                sub[sub["condition"] == c]["hpr_arch_a"].values[0]
                if len(sub[sub["condition"] == c]) else 0.0
                for c in CONDITION_ORDER
            ]
            bars = ax.bar(x + (i - 0.5) * width, vals, width,
                          label=g_labels[grp], color=g_colors[grp], alpha=0.85, edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c].replace("\n", " ") for c in CONDITION_ORDER])
        ax.set_ylabel("HPR")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        save_plot(fig, PLOTS_CITIES, "hpr_by_condition")
        plt.close(fig)

        st.divider()

        # ── Accuracy by condition × baseline group ────────────────────────────
        st.subheader("Accuracy by Condition × Baseline Group")

        acc_arch = st.radio("Architecture", ["A", "B"], horizontal=True, key="base_arch")
        acc_df = base_df[base_df["arch_type"] == acc_arch].copy()

        fig, ax = plt.subplots(figsize=(9, 4))
        for i, grp in enumerate(groups):
            sub = acc_df[acc_df["baseline_group"] == grp]
            vals = [
                sub[sub["condition"] == c]["pct_correct"].values[0]
                if len(sub[sub["condition"] == c]) else 0.0
                for c in CONDITION_ORDER
            ]
            bars = ax.bar(x + (i - 0.5) * width, vals, width,
                          label=g_labels[grp], color=g_colors[grp], alpha=0.85, edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c].replace("\n", " ") for c in CONDITION_ORDER])
        ax.set_ylabel("% Correct")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylim(0, 1.2)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        save_plot(fig, PLOTS_CITIES, f"accuracy_by_baseline_group_arch{acc_arch}")
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Chess Validation
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    chess_df = load_chess_data()

    if chess_df is None:
        st.info("No chess results yet. Run `7_chess_validation.py batch` first.")
    else:
        n_chess_pos  = chess_df["position_id"].nunique()
        n_chess_rows = len(chess_df)

        chess_wrong   = chess_df[chess_df["move_correct"] == False]["tactics_confidence"].dropna()
        chess_correct = chess_df[chess_df["move_correct"] == True]["tactics_confidence"].dropna()
        chess_cbs     = chess_wrong.mean() - chess_correct.mean() \
                        if (len(chess_wrong) and len(chess_correct)) else 0.0

        # ── Header metrics ────────────────────────────────────────────────────
        st.subheader("Cross-Domain Replication — Does the bias survive a different task?")
        st.caption(
            "Same multi-agent architecture, same conditions — but now the task is chess move selection.  \n"
            "Visual sabotage = victim piece removed from board image (mirrors the erased road).  \n"
            "Tactics subagent = the manipulated agent (mirrors NW subagent)."
        )

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Positions", f"{n_chess_pos}")
        mc2.metric("Chess trials", n_chess_rows)
        mc3.metric("Chess CBS", f"{chess_cbs:+.2f}",
                   delta="bias replicates ✓" if chess_cbs > 0 else "no bias",
                   delta_color="inverse" if chess_cbs > 0 else "normal")
        mc4.metric("Cities CBS", f"{cbs:+.2f}", delta="reference")

        st.divider()

        # ── CBS side-by-side ──────────────────────────────────────────────────
        st.subheader("Confidence Bias Score — Cities vs Chess")
        st.caption(
            "CBS = mean confidence when **wrong** − mean confidence when **correct**.  \n"
            "Near-identical values across domains confirm the bias is domain-agnostic."
        )

        fig, ax = plt.subplots(figsize=(5, 3.5))
        domains = ["Cities\n(navigation)", "Chess\n(move selection)"]
        values  = [cbs, chess_cbs]
        colors  = ["#c0392b" if v > 0 else "#27ae60" for v in values]
        bars = ax.bar(domains, values, color=colors, alpha=0.85, edgecolor="white", width=0.4)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{v:+.2f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
        ax.axhline(0, color="#333", linewidth=1.2, linestyle="--")
        ax.set_ylabel("CBS (confidence delta)")
        ax.set_ylim(-2, max(values) + 3)
        ax.spines[["top", "right"]].set_visible(False)
        col_cbs, _ = st.columns([1, 1])
        with col_cbs:
            st.pyplot(fig, use_container_width=True)
        save_plot(fig, PLOTS_CHESS, "cbs_comparison_cities_vs_chess")
        plt.close(fig)

        st.divider()

        # ── Accuracy by condition — cities vs chess (Arch A) ──────────────────
        st.subheader("Accuracy by Condition — Architecture A (Cities vs Chess)")
        st.caption(
            "Arch A = text-only reports to root agent — the architecture most vulnerable to confident lies.  \n"
            "Confident Liar injects 95% confidence with false information about the key feature."
        )

        CHESS_COND_ORDER = ["baseline", "sabotage", "confident_liar", "gaslighting", "mitigation"]
        CHESS_COND_LABELS = {
            "baseline":       "Baseline",
            "sabotage":       "Sabotage",
            "confident_liar": "Confident\nLiar",
            "gaslighting":    "Gaslighting",
            "mitigation":     "Mitigation",
        }

        x     = np.arange(len(CHESS_COND_ORDER))
        width = 0.35

        cities_acc_a = [
            1.0 - df[(df["condition"] == c) & (df["arch_type"] == "A")]["error_rate"].mean()
            for c in CHESS_COND_ORDER
        ]
        chess_acc_a = [
            chess_df[(chess_df["condition"] == c) & (chess_df["arch_type"] == "A")]["move_correct"].mean()
            if len(chess_df[(chess_df["condition"] == c) & (chess_df["arch_type"] == "A")]) else 0.0
            for c in CHESS_COND_ORDER
        ]

        fig, ax = plt.subplots(figsize=(10, 4))
        b1 = ax.bar(x - width / 2, cities_acc_a, width,
                    label="Cities", color="#4C72B0", alpha=0.88, edgecolor="white")
        b2 = ax.bar(x + width / 2, chess_acc_a,  width,
                    label="Chess",  color="#DD8452", alpha=0.88, edgecolor="white")
        for bar, v in zip(list(b1) + list(b2), cities_acc_a + chess_acc_a):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([CHESS_COND_LABELS[c] for c in CHESS_COND_ORDER], fontsize=10)
        ax.set_ylabel("Accuracy (1 − error rate)")
        ax.set_ylim(0, 1.18)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        save_plot(fig, PLOTS_CHESS, "accuracy_by_condition_arch_a")
        plt.close(fig)

        st.divider()

        # ── Accuracy drop under confident liar — both archs ───────────────────
        st.subheader("Accuracy Drop: Baseline → Confident Liar")
        st.caption(
            "How much does accuracy fall when a confident liar injects false information?  \n"
            "Larger drop = stronger vulnerability to social engineering."
        )

        drop_data = []
        for domain, source_df, correct_col in [
            ("Cities", df, "is_correct"),
            ("Chess",  chess_df, "move_correct"),
        ]:
            for arch in ["A", "B"]:
                base_acc = source_df[
                    (source_df["condition"] == "baseline") & (source_df["arch_type"] == arch)
                ][correct_col].mean()
                liar_acc = source_df[
                    (source_df["condition"] == "confident_liar") & (source_df["arch_type"] == arch)
                ][correct_col].mean()
                drop_data.append({
                    "Domain": domain, "Arch": arch,
                    "Baseline": base_acc, "Confident Liar": liar_acc,
                    "Drop": base_acc - liar_acc,
                })
        drop_df = pd.DataFrame(drop_data)

        fig, ax = plt.subplots(figsize=(7, 4))
        group_labels = [f"{r['Domain']}\nArch {r['Arch']}" for _, r in drop_df.iterrows()]
        drop_vals    = drop_df["Drop"].tolist()
        bar_colors   = ["#c0392b" if v > 0 else "#27ae60" for v in drop_vals]
        bars = ax.bar(group_labels, drop_vals, color=bar_colors, alpha=0.85, edgecolor="white", width=0.5)
        for bar, v in zip(bars, drop_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.005 if v >= 0 else -0.025),
                    f"{v:+.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.axhline(0, color="#333", linewidth=1.0, linestyle="--")
        ax.set_ylabel("Accuracy drop (Baseline − Confident Liar)")
        ax.set_title("Consistent drop across both domains and architectures", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        save_plot(fig, PLOTS_CHESS, "accuracy_drop_confident_liar")
        plt.close(fig)

        st.divider()

        # ── Raw chess table ───────────────────────────────────────────────────
        st.subheader("Raw Chess Results")
        col_cf1, col_cf2 = st.columns(2)
        f_chess_cond = col_cf1.multiselect(
            "Condition", CHESS_COND_ORDER, default=CHESS_COND_ORDER, key="chess_cond"
        )
        f_chess_arch = col_cf2.multiselect(
            "Architecture", ["A", "B"], default=["A", "B"], key="chess_arch"
        )
        chess_filtered = chess_df[
            chess_df["condition"].isin(f_chess_cond) &
            chess_df["arch_type"].isin(f_chess_arch)
        ][["position_id", "condition", "arch_type", "move_correct",
           "recommended_move", "ground_truth_move", "victim_piece",
           "tactics_confidence", "root_confidence"]]
        st.dataframe(chess_filtered, use_container_width=True, hide_index=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(f"Data: {n_rows} trials · {n_cities} cities · 5 conditions · 2 architectures")
