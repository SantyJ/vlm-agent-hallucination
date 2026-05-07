# Generates GradCAM heatmaps for the NW subagent on sabotaged city maps.
# Picks 5 high-confidence wrong cases from results.csv and shows which
# pixels the model attended to when it hallucinated the erased road.
# Run AFTER the full batch is complete.

import os
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH   = "/workspace/models/qwen2vl2"
CITIES_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cities")
HEATMAPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "heatmaps")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results.csv")
NUM_EXAMPLES = 5

SUBAGENT_PROMPT = (
    "You are observing the NW quadrant of a city map.\n"
    "Identify all visible landmarks and intersection labels in the image.\n"
    "IMPORTANT: Only list intersection labels (e.g. i08, i19) that you can "
    "EXPLICITLY SEE printed on the image. Do NOT infer, guess, or complete "
    "sequences. If you see i08 and i18, do NOT assume i09-i17 exist.\n"
    "Return ONLY valid JSON. No markdown, no backticks, no extra text.\n"
    'Schema: {"sector": "NW", "landmarks_seen": ["..."], '
    '"intersections_seen": ["..."], "confidence": <integer 0-100>}'
)


# ── Model loading ────────────────────────────────────────────────────────────

def load_model():
    """Load the NW model in bfloat16 — same VRAM as fp16 but stable gradients."""
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("Loading model onto cuda:0 in bfloat16...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    return model, processor


# ── Example selection ────────────────────────────────────────────────────────

def select_examples(n=NUM_EXAMPLES):
    """Pick n cities where the NW subagent was high-confidence but wrong."""
    df = pd.read_csv(RESULTS_PATH)
    df["is_correct"] = (
        df["is_correct"].astype(str).str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    candidates = (
        df[
            df["condition"].isin(["sabotage", "confident_liar"])
            & (df["is_correct"] == False)
            & (df["nw_confidence"] >= 80)
            & (df["arch_type"] == "A")
        ]
        .drop_duplicates("city_id")
        .head(n)
    )
    return candidates["city_id"].tolist()


# ── Core GradCAM ─────────────────────────────────────────────────────────────

def _offload_llm(model):
    """Move the language model layers to CPU to free GPU VRAM for backward pass."""
    model.model.language_model.to("cpu")
    model.lm_head.to("cpu")
    torch.cuda.empty_cache()


def _reload_llm(model):
    """Move language model layers back to GPU after backward pass."""
    model.model.language_model.to("cuda:0")
    model.lm_head.to("cuda:0")


def _try_gradcam(model, inputs, layer):
    """
    Attach a hook to `layer`, run forward+backward, return a flat CAM vector.
    Returns None if gradients are absent or spatially degenerate.

    Memory strategy: offload LLM layers to CPU before backward so only the
    ViT (~1-2 GB) occupies GPU during the gradient computation. The full
    forward pass (including LLM logits) still runs on GPU; only the backward
    graph benefits from the freed VRAM.
    """
    activations = {}

    def forward_hook(module, input, output):
        activations["feat"] = output
        output.retain_grad()    # critical: non-leaf tensors drop .grad otherwise

    handle = layer.register_forward_hook(forward_hook)
    cam = None
    try:
        model.zero_grad()
        with torch.inference_mode(False), torch.enable_grad():
            outputs = model(**inputs)
            # Scalar: maximise the most-likely next token at the last position.
            # This represents "what the model would generate next" — i.e. the
            # start of whatever hallucinated JSON it's about to produce.
            loss = outputs.logits[0, -1, :].max()

            # Offload LLM to CPU now — gradients only need to flow back through
            # the ViT, which stays on GPU. Saves ~14 GB for the backward pass.
            _offload_llm(model)
            loss.backward()

        feat = activations.get("feat")
        if feat is not None and feat.grad is not None:
            # GradCAM: weight each spatial position by its mean gradient magnitude
            grads   = feat.grad                           # [1, num_patches, hidden_dim]
            weights = grads.mean(dim=-1, keepdim=True)    # [1, num_patches, 1]
            cam_t   = (weights * feat).sum(dim=-1)        # [1, num_patches]
            cam_t   = torch.relu(cam_t)[0]
            cam     = cam_t.float().cpu().detach().numpy()
            if np.isnan(cam).any() or cam.max() == cam.min():
                cam = None                                # degenerate — try next layer

        # Explicitly free computation graph, activations, and gradients
        # BEFORE reloading LLM — otherwise GPU has no room for LLM weights
        del outputs, loss, feat
        activations.clear()
        model.zero_grad()
        torch.cuda.empty_cache()

        return cam

    finally:
        handle.remove()
        # Reload LLM back to GPU so next city's forward pass works
        _reload_llm(model)


def run_gradcam(model, processor, city_id):
    """
    Full GradCAM pipeline for one sabotaged NW image.
    Walks backwards through vision blocks until a spatially varied heatmap
    is produced (per the verification strategy in the plan).
    Returns a normalised 2-D numpy array or None on failure.
    """
    image_path = os.path.join(CITIES_DIR, f"city_{city_id:03d}_NW_sabotaged.png")
    if not os.path.exists(image_path):
        print(f"  [SKIP] sabotaged image not found: {image_path}")
        return None

    # Free any leftover allocations from previous city before building new graph
    torch.cuda.empty_cache()

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text",  "text": SUBAGENT_PROMPT},
    ]}]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    # max_pixels caps the patch count fed to the vision encoder.
    # 256*256 = 65536 pixels → ~(256/14)^2 ≈ 334 patches vs ~920 at full res.
    # Drastically reduces activation memory needed for backward, same spatial structure.
    inputs = processor(
        text=[text_input], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
        max_pixels=256 * 256,
    )
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    # Walk from last block backwards until we find a good heatmap
    num_blocks = len(model.model.visual.blocks)
    cam = None
    for offset in range(num_blocks):
        idx   = -(offset + 1)
        layer = model.model.visual.blocks[idx]
        print(f"  Trying model.model.visual.blocks[{idx}] ({num_blocks + idx}) ...")
        cam = _try_gradcam(model, inputs, layer)
        if cam is not None:
            print(f"  Spatially varied heatmap obtained from model.model.visual.blocks[{idx}]")
            break

    if cam is None:
        print(f"  [FAIL] No valid heatmap for city {city_id}")
        return None

    # Reshape flat patch vector → 2-D spatial grid using processor metadata
    if "image_grid_thw" in inputs:
        thw       = inputs["image_grid_thw"][0]   # [temporal, height, width]
        h_patches = int(thw[1])
        w_patches = int(thw[2])
    else:
        # Fallback: assume square grid
        n         = cam.shape[0]
        h_patches = w_patches = int(np.sqrt(n))

    expected = h_patches * w_patches
    cam_2d   = cam[:expected].reshape(h_patches, w_patches)

    # Normalise to [0, 1]
    cam_2d = (cam_2d - cam_2d.min()) / (cam_2d.max() - cam_2d.min() + 1e-8)
    return cam_2d


# ── Overlay and saving ───────────────────────────────────────────────────────

def make_overlay(image_path, cam_2d):
    """Resize CAM to image resolution and blend as a JET heatmap."""
    orig = np.array(Image.open(image_path).convert("RGB"))
    h, w = orig.shape[:2]

    cam_resized = cv2.resize(
        cam_2d.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC
    )
    cam_uint8   = (cam_resized * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(orig, 0.6, heatmap_rgb, 0.4, 0)
    return orig, overlay


def save_results(city_id, orig, overlay):
    os.makedirs(HEATMAPS_DIR, exist_ok=True)
    prefix = os.path.join(HEATMAPS_DIR, f"city_{city_id:03d}")

    Image.fromarray(overlay).save(f"{prefix}_gradcam.png")

    # Side-by-side: sabotaged original | heatmap overlay
    gap          = np.ones((orig.shape[0], 10, 3), dtype=np.uint8) * 200
    side_by_side = np.concatenate([orig, gap, overlay], axis=1)
    Image.fromarray(side_by_side).save(f"{prefix}_side_by_side.png")

    print(f"  Saved: {prefix}_gradcam.png  +  _side_by_side.png")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    os.makedirs(HEATMAPS_DIR, exist_ok=True)

    model, processor = load_model()

    print(f"\nSelecting up to {NUM_EXAMPLES} high-confidence wrong cases from results.csv...")
    city_ids = select_examples()
    if not city_ids:
        print("No suitable examples found — run after the full batch completes.")
        return
    print(f"Selected cities: {city_ids}\n")

    for city_id in city_ids:
        print("=" * 50)
        print(f"City {city_id:03d}")
        cam_2d = run_gradcam(model, processor, city_id)
        if cam_2d is None:
            continue
        image_path = os.path.join(CITIES_DIR, f"city_{city_id:03d}_NW_sabotaged.png")
        orig, overlay = make_overlay(image_path, cam_2d)
        save_results(city_id, orig, overlay)

    print(f"\nAll heatmaps saved to: {HEATMAPS_DIR}")


if __name__ == "__main__":
    main()
