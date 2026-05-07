"""
3a_gradcam_visual_encoder.py  —  FAST variant

GradCAM using only the vision encoder (ViT), not the full LLM.
Fits comfortably in 24 GB VRAM. Typically ~10-15s per city.

Why this works:
  The full forward+backward through a 7B LLM requires holding the entire
  computation graph in GPU memory (~8-12 GB of activations) alongside the
  model weights (~14 GB) — more than 24 GB total. Running only the ViT
  (~600 MB weights, small activations) keeps everything under 3 GB.

What the heatmap shows:
  Which image patches produced the strongest activations in the visual
  representation — i.e. where the model "looked" when encoding the image.
  Standard approach for ViT GradCAM in VLM interpretability work.

Run this first. Fall back to 3b_gradcam_checkpointing.py only if you need
the end-to-end loss signal tied to the actual hallucinated token.
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


# ── GradCAM (vision encoder only) ────────────────────────────────────────────

def _try_gradcam(visual, pixel_values, grid_thw, layer):
    """
    Hook layer, run only the ViT forward+backward, return flat CAM vector.
    Loss = sum of all output features (maximises total patch activation).
    """
    activations = {}

    def forward_hook(module, input, output):
        activations["feat"] = output
        output.retain_grad()

    handle = layer.register_forward_hook(forward_hook)
    cam = None
    try:
        visual.zero_grad()
        with torch.inference_mode(False), torch.enable_grad():
            out  = visual(pixel_values, grid_thw)
            # visual() may return a tensor or a ModelOutput dataclass
            feat_out = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            loss = feat_out.sum()
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

        del out, feat_out, loss, feat
        activations.clear()
        visual.zero_grad()
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
    )
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    visual     = model.model.visual
    num_blocks = len(visual.blocks)
    cam = None
    for offset in range(num_blocks):
        idx   = -(offset + 1)
        layer = visual.blocks[idx]
        print(f"  Trying visual.blocks[{idx}] ({num_blocks + idx}) ...")
        cam = _try_gradcam(visual, inputs["pixel_values"], inputs["image_grid_thw"], layer)
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


def save_results(city_id, orig, overlay, suffix="ve"):
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
        print(f"City {city_id:03d}  [vision-encoder GradCAM]")
        cam_2d = run_gradcam(model, processor, city_id)
        if cam_2d is None:
            continue
        image_path = os.path.join(CITIES_DIR, f"city_{city_id:03d}_NW_sabotaged.png")
        orig, overlay = make_overlay(image_path, cam_2d)
        save_results(city_id, orig, overlay, suffix="ve")

    print(f"\nAll heatmaps saved to: {HEATMAPS_DIR}")


if __name__ == "__main__":
    main()
