"""
3d_attention_rollout_final.py

Attention rollout heatmaps with clean single-color visualization.

Key differences from 3c:
  - Single color (white → deep red) instead of JET rainbow
  - Percentile normalization + gamma correction for better contrast
  - Auto-detects erased road rectangle from black pixels
  - Draws a yellow outline + "Erased Road" label on the overlay
  - Saves to data/heatmaps_final/

The story in one image:
  Pale = model ignored this region
  Deep red = model attended here
  Yellow box = where the road was erased
  → Erased road is pale: model never looked there, hallucinated from context
"""

import os
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH   = "/workspace/models/qwen2vl2"
CITIES_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cities")
HEATMAPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "heatmaps_final")
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

os.makedirs(HEATMAPS_DIR, exist_ok=True)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("Loading model onto cuda:0 in bfloat16...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0",
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


# ── Attention rollout (same QKV hook approach as 3c) ─────────────────────────

def compute_attention_rollout(model, inputs):
    visual    = model.model.visual
    num_heads = visual.blocks[0].attn.num_heads
    qkv_outs  = []

    hooks = []
    for block in visual.blocks:
        def qkv_hook(module, input, output, store=qkv_outs):
            store.append(output.detach().float().cpu())
        hooks.append(block.attn.qkv.register_forward_hook(qkv_hook))

    try:
        with torch.no_grad():
            _ = visual(inputs["pixel_values"], inputs["image_grid_thw"])
    finally:
        for h in hooks:
            h.remove()

    if not qkv_outs:
        return None

    thw         = inputs["image_grid_thw"][0]
    h_patches   = int(thw[1])
    w_patches   = int(thw[2])
    num_patches = h_patches * w_patches

    rollout = None
    for qkv in qkv_outs:
        N = qkv.shape[0]
        if N != num_patches:
            continue
        hidden   = qkv.shape[1] // 3
        head_dim = hidden // num_heads
        qkv_r    = qkv.reshape(N, 3, num_heads, head_dim)
        Q = qkv_r[:, 0].permute(1, 0, 2)   # [heads, N, head_dim]
        K = qkv_r[:, 1].permute(1, 0, 2)
        scale = head_dim ** -0.5
        attn  = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) * scale, dim=-1)
        a     = attn.mean(dim=0)            # [N, N]
        a     = a + torch.eye(N)
        a     = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        rollout = a if rollout is None else rollout @ a

    if rollout is None:
        return None

    importance = rollout.mean(dim=0).numpy()
    if np.isnan(importance).any() or importance.max() == importance.min():
        return None
    return importance, h_patches, w_patches


# ── Erased road rectangle detection ──────────────────────────────────────────

def detect_erased_rect(image_path, row_threshold=50, col_threshold=10):
    """
    Auto-detect the black erased-road rectangle from pixel values.
    Returns (row_min, row_max, col_min, col_max) or None.
    """
    img   = np.array(Image.open(image_path).convert("RGB"))
    black = (img[:, :, 0] < 20) & (img[:, :, 1] < 20) & (img[:, :, 2] < 20)
    dense_rows = np.where(black.sum(axis=1) > row_threshold)[0]
    dense_cols = np.where(black.sum(axis=0) > col_threshold)[0]
    if len(dense_rows) == 0 or len(dense_cols) == 0:
        return None
    return int(dense_rows.min()), int(dense_rows.max()), \
           int(dense_cols.min()), int(dense_cols.max())


# ── Single-color heatmap overlay ──────────────────────────────────────────────

def make_trizone_overlay(image_path, cam_2d):
    """
    Clean heatmap: Gaussian blur → threshold at 70th percentile → single-color overlay.

    Steps:
      1. Resize CAM to image resolution
      2. Gaussian blur (15x15) — merges scattered noise into coherent regions
      3. Zero out everything below 70th percentile — only strongest activations remain
      4. Normalise remaining values to [0, 1]
      5. Map to white → deep red (single hue, varying shade)
      6. Blend with original — transparent where zeroed, red where active
    """
    orig = np.array(Image.open(image_path).convert("RGB"))
    h, w = orig.shape[:2]

    cam_resized = cv2.resize(cam_2d.astype(np.float32), (w, h),
                             interpolation=cv2.INTER_CUBIC)

    # Step 1: blur to merge noise into coherent blobs
    cam_resized = cv2.GaussianBlur(cam_resized, (15, 15), 0)

    # Step 2: zero out everything below 70th percentile
    threshold = np.percentile(cam_resized, 70)
    cam_resized[cam_resized < threshold] = 0

    # Step 3: normalise remaining values
    if cam_resized.max() > 0:
        cam_resized = cam_resized / cam_resized.max()

    # Step 4: build red heatmap (white=0, deep red=1)
    r_ch = np.ones_like(cam_resized)
    g_ch = 1.0 - cam_resized * 0.88
    b_ch = 1.0 - cam_resized * 0.88
    heatmap = (np.stack([r_ch, g_ch, b_ch], axis=-1) * 255).astype(np.uint8)

    # Step 5: blend — use cam as alpha so zeroed areas are fully transparent
    alpha = cam_resized[:, :, np.newaxis]           # [H, W, 1]
    overlay = (orig * (1 - alpha * 0.75) + heatmap * alpha * 0.75).astype(np.uint8)

    return orig, overlay


# ── Save ──────────────────────────────────────────────────────────────────────

def save_results(city_id, orig, overlay):
    prefix = os.path.join(HEATMAPS_DIR, f"city_{city_id:03d}")
    Image.fromarray(overlay).save(f"{prefix}_heatmap.png")
    gap          = np.ones((orig.shape[0], 12, 3), dtype=np.uint8) * 240
    side_by_side = np.concatenate([orig, gap, overlay], axis=1)
    Image.fromarray(side_by_side).save(f"{prefix}_side_by_side.png")
    print(f"  Saved: {prefix}_heatmap.png  +  _side_by_side.png")


# ── Full pipeline per city ────────────────────────────────────────────────────

def run_city(model, processor, city_id):
    image_path = os.path.join(CITIES_DIR, f"city_{city_id:03d}_NW_sabotaged.png")
    if not os.path.exists(image_path):
        print(f"  [SKIP] image not found: {image_path}")
        return

    torch.cuda.empty_cache()

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text",  "text": SUBAGENT_PROMPT},
    ]}]
    text_input   = processor.apply_chat_template(
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
        print(f"  [FAIL] rollout failed for city {city_id}")
        return

    importance, h_patches, w_patches = result
    cam_2d = importance.reshape(h_patches, w_patches)
    cam_2d = (cam_2d - cam_2d.min()) / (cam_2d.max() - cam_2d.min() + 1e-8)

    orig, overlay = make_trizone_overlay(image_path, cam_2d)
    save_results(city_id, orig, overlay)


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
        print(f"City {city_id:03d}")
        run_city(model, processor, city_id)

    print(f"\nAll heatmaps saved to: {HEATMAPS_DIR}")


if __name__ == "__main__":
    main()
