# VLM Agent Hallucination Audit

Can a single confident-but-lying subagent corrupt a multi-agent system's decision-making? This project audits **confidence bias** and **hallucination propagation** in vision-language model (VLM) pipelines, using synthetic city navigation and chess as evaluation domains.

---

## Overview

Modern multi-agent AI systems delegate perception to specialized subagents and rely on a root agent to synthesize their reports. This creates a trust vulnerability: if a subagent reports false information with high confidence, does the root agent get misled?

We test this with a controlled experiment:

- **120 procedurally generated city maps** across 3 complexity levels
- **4 subagents** (one per map quadrant) report landmarks and road conditions to a root agent
- The root agent recommends a shortest path, scored against Dijkstra ground truth
- We inject adversarial conditions (sabotage, confident lying, gaslighting) to measure how much a dishonest subagent degrades system performance

---

## Experiment Design

### Conditions (5 total)

| Condition | Description |
|---|---|
| `baseline` | All subagents report truthfully |
| `sabotage` | One road is visually erased from the image |
| `confident_liar` | One subagent confidently reports a non-existent road (100% confidence) |
| `gaslighting` | One subagent contradicts the others with fabricated information |
| `mitigation` | Root agent prompted to be skeptical of outlier subagent reports |

### Architectures (2 total)

| Architecture | Root Agent Input |
|---|---|
| **A** | Text reports from subagents only |
| **B** | Text reports + the original map image |

### Scale

- **Cities**: 120 cities × 5 conditions × 2 architectures = **1,200 trials**
- **Chess**: 30 positions × 5 conditions × 2 architectures = **300 trials** (cross-domain replication)

---

## Key Findings

### Accuracy by Condition (Architecture A / B)

| Condition | Arch A | Arch B |
|---|---|---|
| Baseline | 20.0% | 5.8% |
| Sabotage | 11.7% | 5.8% |
| Confident Liar | **8.3%** | **2.5%** |
| Gaslighting | 11.7% | 5.0% |
| Mitigation | 15.8% | 5.0% |

### Confidence Bias Score (CBS)

Wrong decisions came with **significantly higher subagent confidence** than correct ones:

- Mean confidence when wrong: **98.22**
- Mean confidence when correct: **91.47**
- **CBS = +6.75** — models are more confident when they're wrong

### Hallucination Propagation Rate (HPR)

Architecture A was substantially more vulnerable to adversarial subagents than Architecture B (which also sees the image directly):

- Confident Liar HPR: **+11.7%** (Arch A) vs **+3.3%** (Arch B)
- Statistical significance: Mann-Whitney U, **p = 0.0045**

### Architecture Comparison

Giving the root agent direct image access (Arch B) did not improve accuracy — it actually reduced it by ~60–70% across all conditions. Architecture B becomes overloaded with visual complexity and relies more heavily on strict path matching (higher strict Jaccard, lower loose Jaccard).

---

## Cross-Domain Validation: Chess

To test whether confidence bias is task-agnostic, we replicated the experiment on chess move selection using 30 board positions. Three Qwen2-VL subagents (Material, Tactics, Strategy) analyze the board independently and report to a root agent.

The same confidence bias pattern held: wrong moves were recommended with higher confidence, and `confident_liar` injections degraded accuracy in Architecture A more than Architecture B.

---

## Attention Visualization

We implemented GradCAM and attention rollout on the Qwen2-VL vision encoder to investigate *where* the model looks when it hallucinates. When a road is visually erased (sabotage), the model still attends to the region where the road used to be — suggesting a **perception-language disconnect** where visual attention does not fully gate linguistic claims.

---

## Repo Structure

```
vlm-agent-hallucination/
├── 1_city_generator.py          # Procedural city map generation (NetworkX + Pillow)
├── 2_qwen_multi_gpu.py          # Qwen2-VL-7B multi-GPU inference (4x GPU parallelism)
├── 3_gradcam_hooks.py           # GradCAM attention visualization
├── 3a–3d_*.py                   # GradCAM / attention rollout variants
├── 4_langgraph_flow.py          # Multi-agent workflow (LangGraph + Claude)
├── 5_batch_runner.py            # Main experiment orchestrator (1,200 trials)
├── 6_streamlit_pitch.py         # Interactive results dashboard
├── 7_chess_validation.py        # Cross-domain chess replication (300 trials)
├── 8_baseline_controlled_analysis.py  # Controlled analysis on baseline-correct subset
├── 8_statistical_analysis.py   # CBS, HPR, Mann-Whitney U, Jaccard scoring
├── 9_data_quality_checks.py    # Ground-truth audit (7 checks)
├── 10_threshold_robustness.py  # Robustness across Jaccard thresholds 0.3–0.7
├── data/
│   ├── cities/                  # 120 generated city PNGs + metadata JSON
│   ├── chess/boards/            # 30 rendered chess positions
│   ├── results.csv              # Full trial results (1,200+ rows)
│   ├── metrics_summary.txt      # Computed accuracy, CBS, HPR, Jaccard stats
│   ├── analysis/                # Baseline-controlled and robustness outputs
│   ├── heatmaps_final/          # GradCAM overlay images
│   ├── heatmaps_rollout/        # Attention rollout visualizations
│   └── plots/                   # Figures for cities and chess domains
└── requirements.txt             # Python dependencies (LangGraph + Anthropic API)
```

---

## Models Used

| Role | Model |
|---|---|
| Visual subagents (quadrant analysis) | Qwen2-VL-7B-Instruct (self-hosted, 4x GPU) |
| Root agent synthesizer | Claude Sonnet (via Anthropic API) |
| Subagents in LangGraph flow | Claude Haiku (via Anthropic API) |

---

## Setup

**Prerequisites:** Python 3.10+, CUDA-capable GPUs for Qwen inference (or swap for API-based VLM).

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install GPU dependencies (platform-specific)

The Qwen2-VL inference requires `torch`, `transformers`, and `qwen-vl-utils`. The `torch` installation varies by platform:

**Linux / Windows (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate qwen-vl-utils
```

**Mac (Apple Silicon, MPS):**
```bash
pip install torch torchvision
pip install transformers accelerate qwen-vl-utils
```

**CPU only (no GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate qwen-vl-utils
```

> **Note:** The full experiment was run with 4× RTX 4090 GPUs. CPU inference is extremely slow and not recommended for the full 1,200-trial run.

### 3. Download the Qwen2-VL model

The visual subagents use [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) (~16 GB). Download it with:

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir ./models/qwen2vl2
```

Then update `MODEL_PATH` in `2_qwen_multi_gpu.py` to point to your local path:

```python
MODEL_PATH = "./models/qwen2vl2"  # adjust if needed
```

### 4. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY=your-key-here
```

**Run order:**

```bash
# 1. Generate city maps
python 1_city_generator.py

# 2. Run Qwen subagent inference (multi-GPU)
python 2_qwen_multi_gpu.py

# 3. Run the full LangGraph experiment (1,200 trials)
python 5_batch_runner.py

# 4. Compute statistics
python 8_statistical_analysis.py

# 5. Launch interactive dashboard
streamlit run 6_streamlit_pitch.py
```

For chess cross-domain validation:

```bash
python 7_chess_validation.py
```

---

## Metrics

- **Accuracy**: Exact match of recommended path against Dijkstra ground truth
- **Strict Jaccard**: Intersection-over-union on full waypoint sequences
- **Loose Jaccard**: Intersection-over-union on landmark names only
- **CBS (Confidence Bias Score)**: `mean_confidence(wrong) − mean_confidence(correct)`
- **HPR (Hallucination Propagation Rate)**: `baseline_accuracy − condition_accuracy`

---

## Reference

This project is informed by:

> *Language-Instructed Vision Models* — see `3452_Language_Instructed_Visio.pdf`
