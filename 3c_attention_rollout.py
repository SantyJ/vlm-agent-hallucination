"""
3c_attention_rollout.py  —  FAST, no backward pass

Attention rollout heatmaps for the NW subagent on sabotaged city maps.
No gradients needed — extracts attention weights directly from ViT blocks
during a standard forward pass.

Method (Abnar & Zuidema, 2020):
  1. For each transformer block, extract the mean attention matrix across heads
  2. Add identity (residual connection) and re-normalise each row
  3. Multiply all layer matrices together — this "rolls out" how attention
     from the final layer traces back to the input patches
  4. Take the row corresponding to the [CLS] token (or mean over all tokens)
     as the spatial importance map

Why this works here:
  GradCAM needs backward pass → OOM on 24 GB with 7B model.
  Attention rollout is pure forward pass → fits in <2 GB, ~2s per city.
  Both answer "which patches did the model focus on?" for the sabotaged map.

Outputs (separate dir from GradCAM):
  data/heatmaps_rollout/city_XXX_rollout_gradcam.png
  data/heatmaps_rollout/city_XXX_rollout_side_by_side.png
"""

import os
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH    = "/workspace/models/qwen2vl2"
CITIES_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cities")
HEATMAPS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "heatmaps_rollout")
RESULTS_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results.csv")
NUM_EXAMPLES  = 5

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

os.makedirs(HEATMAPS_DIR, exist_ok=True)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
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


# ── Example selection ─────────────────────────────────────────────────────────

def select_examples(n=NUM_EXAMPLES):
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


# ── Attention rollout ─────────────────────────────────────────────────────────

def compute_attention_rollout(model, inputs):
    """
    Extract attention matrices from all ViT blocks by hooking the qkv linear
    output and manually computing Q @ K.T / scale + softmax per layer.

    Qwen2VL uses Flash Attention which does not expose attention weights.
    We bypass this by capturing the raw QKV projection output and computing
    the full attention matrix ourselves in float32.

    hidden_states shape going into each block: [num_patches, hidden_dim]
    qkv output: [num_patches, 3 * hidden_dim]
    After reshape: [3, num_patches, num_heads, head_dim]
    → Q, K each: [num_patches, num_heads, head_dim]

    Rollout:
      For each layer:
        A = mean_heads(softmax(Q @ K.T / sqrt(head_dim)))  # [N, N]
        A = A + I;  A = A / row_sum                        # residual + renorm
      rollout = A_0 @ A_1 @ ... @ A_L
      importance = rollout.mean(dim=0)
    """
    visual      = model.model.visual
    num_heads   = visual.blocks[0].attn.num_heads
    qkv_outputs = []   # one tensor per block: [N, 3*H*D]

    hooks = []
    for block in visual.blocks:
        def qkv_hook(module, input, output, store=qkv_outputs):
            store.append(output.detach().float().cpu())
        hooks.append(block.attn.qkv.register_forward_hook(qkv_hook))

    try:
        with torch.no_grad():
            _ = model.model.visual(inputs["pixel_values"], inputs["image_grid_thw"])
    finally:
        for h in hooks:
            h.remove()

    if not qkv_outputs:
        print("  [WARN] No QKV outputs captured")
        return None

    thw         = inputs["image_grid_thw"][0]
    h_patches   = int(thw[1])
    w_patches   = int(thw[2])
    num_patches = h_patches * w_patches

    rollout = None
    for qkv in qkv_outputs:
        # qkv: [N, 3 * num_heads * head_dim]
        N = qkv.shape[0]
        if N != num_patches:
            continue   # skip if shape unexpected

        hidden = qkv.shape[1] // 3
        head_dim = hidden // num_heads

        # [N, 3, num_heads, head_dim] → unbind on dim 1
        qkv_r = qkv.reshape(N, 3, num_heads, head_dim)
        Q = qkv_r[:, 0]   # [N, num_heads, head_dim]
        K = qkv_r[:, 1]   # [N, num_heads, head_dim]

        # Compute attention per head: [num_heads, N, N]
        scale = head_dim ** -0.5
        Q_t = Q.permute(1, 0, 2)   # [num_heads, N, head_dim]
        K_t = K.permute(1, 0, 2)   # [num_heads, N, head_dim]
        attn = torch.bmm(Q_t, K_t.transpose(1, 2)) * scale  # [num_heads, N, N]
        attn = torch.softmax(attn, dim=-1)

        # Mean over heads → [N, N]
        a = attn.mean(dim=0)

        # Residual + row-normalise
        a = a + torch.eye(N)
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        rollout = a if rollout is None else rollout @ a

    if rollout is None:
        return None

    importance = rollout.mean(dim=0).numpy()   # [num_patches]
    if np.isnan(importance).any() or importance.max() == importance.min():
        return None

    return importance, h_patches, w_patches


# ── Full pipeline per city ────────────────────────────────────────────────────

def run_rollout(model, processor, city_id):
    image_path = os.path.join(CITIES_DIR, f"city_{city_id:03d}_NW_sabotaged.png")
    if not os.path.exists(image_path):
        print(f"  [SKIP] sabotaged image not found: {image_path}")
        return None

    torch.cuda.empty_cache()

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text",  "text": SUBAGENT_PROMPT},
    ]}]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    result = compute_attention_rollout(model, inputs)
    if result is None:
        print(f"  [FAIL] Could not compute rollout for city {city_id}")
        return None

    importance, h_patches, w_patches = result
    print(f"  Rollout computed: {h_patches}×{w_patches} patch grid, "
          f"{len(importance)} values")

    cam_2d = importance.reshape(h_patches, w_patches)
    cam_2d = (cam_2d - cam_2d.min()) / (cam_2d.max() - cam_2d.min() + 1e-8)
    return cam_2d


# ── Overlay and saving ────────────────────────────────────────────────────────

def make_overlay(image_path, cam_2d):
    orig = np.array(Image.open(image_path).convert("RGB"))
    h, w = orig.shape[:2]
    cam_resized = cv2.resize(cam_2d.astype(np.float32), (w, h),
                             interpolation=cv2.INTER_CUBIC)
    cam_uint8   = (cam_resized * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay     = cv2.addWeighted(orig, 0.6, heatmap_rgb, 0.4, 0)
    return orig, overlay


def save_results(city_id, orig, overlay):
    prefix = os.path.join(HEATMAPS_DIR, f"city_{city_id:03d}_rollout")
    Image.fromarray(overlay).save(f"{prefix}_gradcam.png")
    gap          = np.ones((orig.shape[0], 10, 3), dtype=np.uint8) * 200
    side_by_side = np.concatenate([orig, gap, overlay], axis=1)
    Image.fromarray(side_by_side).save(f"{prefix}_side_by_side.png")
    print(f"  Saved: {prefix}_gradcam.png  +  _side_by_side.png")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    model, processor = load_model()

    print(f"\nSelecting up to {NUM_EXAMPLES} high-confidence wrong cases...")
    city_ids = select_examples()
    if not city_ids:
        print("No suitable examples found.")
        return
    print(f"Selected cities: {city_ids}\n")

    for city_id in city_ids:
        print("=" * 50)
        print(f"City {city_id:03d}  [attention rollout]")
        cam_2d = run_rollout(model, processor, city_id)
        if cam_2d is None:
            continue
        image_path = os.path.join(CITIES_DIR, f"city_{city_id:03d}_NW_sabotaged.png")
        orig, overlay = make_overlay(image_path, cam_2d)
        save_results(city_id, orig, overlay)

    print(f"\nAll rollout heatmaps saved to: {HEATMAPS_DIR}")


if __name__ == "__main__":
    main()
