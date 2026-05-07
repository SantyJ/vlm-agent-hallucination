"""
2_qwen_multi_gpu.py

Qwen2-VL multi-GPU inference engine.
Loads one independent model instance per GPU (4x RTX 4090).
Analyzes all 4 city map quadrants in parallel using ThreadPoolExecutor.
Each thread is fully self-contained with explicit CUDA device context.
"""

import os
import json
import time
import logging
import torch

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_PATH = "/workspace/models/qwen2vl2"
LOGS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
SECTORS    = ["NW", "NE", "SW", "SE"]
GPU_MAP    = {"NW": 0, "NE": 1, "SW": 2, "SE": 3}

SUBAGENT_PROMPT = (
    "You are observing the {sector} quadrant of a city map.\n"
    "Identify all visible landmarks and intersection labels in the image.\n"
    "IMPORTANT: Only list intersection labels (e.g. i08, i19) that you can "
    "EXPLICITLY SEE printed on the image. Do NOT infer, guess, or complete "
    "sequences. If you see i08 and i18, do NOT assume i09-i17 exist.\n"
    "Return ONLY valid JSON. No markdown, no backticks, no extra text.\n"
    'Schema: {{"sector": "{sector}", "landmarks_seen": ["..."], '
    '"intersections_seen": ["..."], "confidence": <integer 0-100>}}'
)

# ─────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)


def _make_logger(name: str, filename: str) -> logging.Logger:
    """Creates a named logger that writes to its own file AND master.log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger  # already configured

    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Per-agent file handler
    fh = logging.FileHandler(os.path.join(LOGS_DIR, filename), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Master log handler (shared across all loggers)
    mh = logging.FileHandler(os.path.join(LOGS_DIR, "master.log"), encoding="utf-8")
    mh.setFormatter(fmt)
    logger.addHandler(mh)

    # Also echo to stdout
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# Create per-GPU loggers
GPU_LOGGERS = {
    "NW": _make_logger("gpu_nw", "gpu_0_nw.log"),
    "NE": _make_logger("gpu_ne", "gpu_1_ne.log"),
    "SW": _make_logger("gpu_sw", "gpu_2_sw.log"),
    "SE": _make_logger("gpu_se", "gpu_3_se.log"),
}
ROOT_LOGGER = _make_logger("root_agent", "root_agent.log")


# ─────────────────────────────────────────────
# Main Engine
# ─────────────────────────────────────────────
class QwenMultiGPUAgent:
    """
    Loads one Qwen2-VL-7B instance onto each of the 4 GPUs at init time.
    Provides analyze_all_quadrants() for true parallel inference.
    """

    def __init__(self):
        ROOT_LOGGER.info("[SYSTEM] Initialising QwenMultiGPUAgent...")
        ROOT_LOGGER.info("[SYSTEM] Loading processor from %s", MODEL_PATH)
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH)

        self.models = {}
        for sector, gpu_id in GPU_MAP.items():
            device = f"cuda:{gpu_id}"
            log    = GPU_LOGGERS[sector]
            log.info("[SYSTEM] [GPU-%d/%s] Loading model onto %s ...", gpu_id, sector, device)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float16,
                device_map=device,
            )
            model.eval()
            self.models[sector] = model
            log.info("[SYSTEM] [GPU-%d/%s] Model loaded successfully.", gpu_id, sector)

        ROOT_LOGGER.info("[SYSTEM] All 4 GPU models loaded. Engine ready.")

    # ──────────────────────────────────────────
    def _infer_one(self, sector: str, image_path: str, city_id: int,
                   system_prompt: str = None) -> dict:
        """
        Fully self-contained inference call for one sector.
        Sets CUDA device explicitly at the very top — no shared tensors.
        """
        gpu_id = GPU_MAP[sector]
        log    = GPU_LOGGERS[sector]

        # ── CUDA safety: set device context for this thread ──────────────
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"

        tag = f"[CITY-{city_id:03d}] [GPU-{gpu_id}/{sector}]"
        log.info("%s Starting inference | image: %s", tag, os.path.basename(image_path))

        prompt_text = SUBAGENT_PROMPT.format(sector=sector)
        if system_prompt:
            prompt_text = system_prompt + "\n\n" + prompt_text

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text",  "text": prompt_text},
                ],
            }
        ]

        try:
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            # Move all tensors to this thread's dedicated GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}

            t_start = time.time()
            with torch.no_grad():
                generated_ids = self.models[sector].generate(
                    **inputs, max_new_tokens=512
                )
            elapsed = time.time() - t_start

            # Decode back to Python strings immediately (no tensor leaks)
            trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs["input_ids"], generated_ids)
            ]
            raw_text = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            log.info("%s Inference completed in %.2fs", tag, elapsed)
            log.info("%s Raw output: %s", tag, raw_text[:300])

            result = self._parse_json(raw_text, sector)
            log.info("%s Parsed JSON: %s", tag, json.dumps(result))
            return result

        except Exception as exc:
            log.error("%s ERROR during inference: %s", tag, exc, exc_info=True)
            return {
                "sector": sector,
                "landmarks_seen": [],
                "roads_seen": [],
                "confidence": 0,
                "parse_error": True,
                "error": str(exc),
            }

    # ──────────────────────────────────────────
    @staticmethod
    def _parse_json(raw: str, sector: str) -> dict:
        """Strips markdown fences and parses JSON, with a safe fallback."""
        text = raw.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "sector": sector,
                "landmarks_seen": [],
                "roads_seen": [],
                "confidence": 0,
                "parse_error": True,
                "raw": text[:200],
            }

    # ──────────────────────────────────────────
    def analyze_all_quadrants(
        self,
        quadrant_paths: dict,
        city_id: int,
        system_prompts: dict = None,
    ) -> list:
        """
        Runs all 4 quadrant inferences sequentially (GPU 0 → 1 → 2 → 3).

        Each model stays resident on its own GPU between calls, so there is no
        reload cost.  A ThreadPoolExecutor approach looks parallel but every
        model.generate() is a Python token-by-token loop, so all 4 threads
        fight for the GIL on every step — measured at ~33s vs ~16s sequential.
        Sequential execution is ~2× faster for this workload.
        """
        system_prompts = system_prompts or {}
        ROOT_LOGGER.info(
            "[CITY-%03d] [ROOT] Running quadrant inferences sequentially (GPU 0→1→2→3)...",
            city_id,
        )

        results = {}
        for sector in SECTORS:
            results[sector] = self._infer_one(
                sector,
                quadrant_paths[sector],
                city_id,
                system_prompts.get(sector),
            )

        ROOT_LOGGER.info(
            "[CITY-%03d] [ROOT] All quadrant inferences complete.", city_id
        )
        return [results[s] for s in SECTORS]


# ─────────────────────────────────────────────
# Standalone Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    CITIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cities")
    city_id    = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    quadrant_paths = {
        s: os.path.join(CITIES_DIR, f"city_{city_id:03d}_{s}.png")
        for s in SECTORS
    }

    # Check all images exist
    for s, p in quadrant_paths.items():
        if not os.path.exists(p):
            print(f"[ERROR] Missing image: {p}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print(f"  Qwen Multi-GPU Standalone Test — City {city_id:03d}")
    print("=" * 60 + "\n")

    agent   = QwenMultiGPUAgent()
    reports = agent.analyze_all_quadrants(quadrant_paths, city_id=city_id)

    print("\n" + "=" * 60)
    print("  FINAL OUTPUTS")
    print("=" * 60)
    for r in reports:
        print(json.dumps(r, indent=2))
    print(f"\nLogs written to: {LOGS_DIR}")
