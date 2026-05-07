"""
3b_gradcam_checkpointing.py  —  SLOW but end-to-end variant

GradCAM using the full model with gradient checkpointing enabled.
Trades compute time for memory: activations are NOT stored during forward
pass — they are recomputed on-the-fly during backward. This halves peak
activation memory at the cost of ~2-3x longer backward pass.

Expected time: 3-8 minutes per city (vs ~10-15s for 3a).
Run this only if you need the loss signal tied to the actual hallucinated
token ("which patches drove the next-token prediction"). Otherwise use
3a_gradcam_visual_encoder.py.

Why gradient checkpointing saves memory:
  Normally PyTorch stores ALL intermediate activations during forward pass
  so backward can use them. With checkpointing, only checkpoint boundaries
  are stored; everything in between is recomputed during backward. For a
  32-layer transformer, this reduces activation memory from O(layers) to
  O(sqrt(layers)).
"""

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
HEATMAPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "heatmaps_ckpt")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results.csv")
NUM_EXAMPLES = 1

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
    # Enable gradient checkpointing on both the vision encoder and language model.
    # This prevents storing intermediate activations during forward pass —
    # they are recomputed during backward instead. Reduces peak activation
    # memory from ~8-12 GB to ~2-4 GB, at the cost of ~2-3x slower backward.
    print("Enabling gradient checkpointing on vision encoder and language model...")
    model.model.visual.gradient_checkpointing_enable()
    model.model.language_model.gradient_checkpointing_enable()
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


# ── GradCAM (full model, gradient checkpointing) ─────────────────────────────

def _try_gradcam(model, inputs, layer):
    """
    Attach hook, run full model forward+backward with checkpointing.
    Loss = max logit at final position (start of hallucinated JSON output).
    Returns flat CAM vector or None.
    """
    activations = {}

    def forward_hook(module, input, output):
        activations["feat"] = output
        output.retain_grad()

    handle = layer.register_forward_hook(forward_hook)
    cam = None
    try:
        model.zero_grad()
        torch.cuda.empty_cache()

        with torch.inference_mode(False), torch.enable_grad():
            outputs = model(**inputs)
            loss    = outputs.logits[0, -1, :].max()
            loss.backward()

        feat = activations.get("feat")
        if feat is not None and feat.grad is not None:
            grads   = feat.grad
            weights = grads.mean(dim=-1, keepdim=True)
            cam_t   = (weights * feat).sum(dim=-1)
            if cam_t.dim() > 1:
                cam_t = cam_t[0]
            cam_t = torch.relu(cam_t)
            cam   = cam_t.float().cpu().detach().numpy()
            if np.isnan(cam).any() or cam.max() == cam.min():
                cam = None

        del outputs, loss, feat
        activations.clear()
        model.zero_grad()
        torch.cuda.empty_cache()
        return cam

    finally:
        handle.remove()


def run_gradcam(model, processor, city_id):
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
        max_pixels=256 * 256,   # cap patches to reduce activation footprint
    )
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    num_blocks = len(model.model.visual.blocks)
    cam = None
    for offset in range(num_blocks):
        idx   = -(offset + 1)
        layer = model.model.visual.blocks[idx]
        print(f"  Trying visual.blocks[{idx}] ({num_blocks + idx}) ...")
        cam = _try_gradcam(model, inputs, layer)
        if cam is not None:
            print(f"  Heatmap obtained from visual.blocks[{idx}]")
            break

    if cam is None:
        print(f"  [FAIL] No valid heatmap for city {city_id}")
        return None

    if "image_grid_thw" in inputs:
        thw       = inputs["image_grid_thw"][0]
        h_patches = int(thw[1])
        w_patches = int(thw[2])
    else:
        n = cam.shape[0]
        h_patches = w_patches = int(np.sqrt(n))

    expected = h_patches * w_patches
    cam_2d   = cam[:expected].reshape(h_patches, w_patches)
    cam_2d   = (cam_2d - cam_2d.min()) / (cam_2d.max() - cam_2d.min() + 1e-8)
    return cam_2d


# ── Overlay and saving ────────────────────────────────────────────────────────

def make_overlay(image_path, cam_2d):
    orig = np.array(Image.open(image_path).convert("RGB"))
    h, w = orig.shape[:2]
    cam_resized = cv2.resize(cam_2d.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    cam_uint8   = (cam_resized * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay     = cv2.addWeighted(orig, 0.6, heatmap_rgb, 0.4, 0)
    return orig, overlay


def save_results(city_id, orig, overlay, suffix="ckpt"):
    os.makedirs(HEATMAPS_DIR, exist_ok=True)
    prefix = os.path.join(HEATMAPS_DIR, f"city_{city_id:03d}_{suffix}")
    Image.fromarray(overlay).save(f"{prefix}_gradcam.png")
    gap          = np.ones((orig.shape[0], 10, 3), dtype=np.uint8) * 200
    side_by_side = np.concatenate([orig, gap, overlay], axis=1)
    Image.fromarray(side_by_side).save(f"{prefix}_side_by_side.png")
    print(f"  Saved: {prefix}_gradcam.png  +  _side_by_side.png")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(HEATMAPS_DIR, exist_ok=True)
    model, processor = load_model()

    print(f"\nSelecting up to {NUM_EXAMPLES} high-confidence wrong cases...")
    city_ids = select_examples()
    if not city_ids:
        print("No suitable examples found.")
        return
    print(f"Selected cities: {city_ids}\n")

    for city_id in city_ids:
        print("=" * 50)
        print(f"City {city_id:03d}  [full model + gradient checkpointing]")
        cam_2d = run_gradcam(model, processor, city_id)
        if cam_2d is None:
            continue
        image_path = os.path.join(CITIES_DIR, f"city_{city_id:03d}_NW_sabotaged.png")
        orig, overlay = make_overlay(image_path, cam_2d)
        save_results(city_id, orig, overlay, suffix="ckpt")

    print(f"\nAll heatmaps saved to: {HEATMAPS_DIR}")


if __name__ == "__main__":
    main()
