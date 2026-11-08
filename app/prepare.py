"""Precompute the study targets on the GPU server (before recruitment).

For every prompt, generates targets for several seeds from one rough face
sketch, runs the landmark confidence check, and saves the targets that pass as
guide packages (status: pending review). It also prints the confidence
distribution, to calibrate ghost.CONF_MIN on generated targets in the pilot.

  python prepare.py "a young woman with short hair" "an old man with a beard"
  python prepare.py --prompts prompts.txt --seeds 6 --keep 2
  python prepare.py --dry-run ...          # only report confidence, save nothing

Then review: python package.py list / python package.py review <id> ...
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "..", "shadowdraw-demo", "demo")
sys.path.insert(0, DEMO)

from engine import CANVAS, ShadowEngine  # noqa: E402

import generate  # noqa: E402
import ghost  # noqa: E402
import package  # noqa: E402


def default_sketch():
    """A plain frontal face scribble (head, eyes, nose, mouth) in the 300 frame."""
    im = Image.new("L", (CANVAS, CANVAS), 0)
    d = ImageDraw.Draw(im)
    d.ellipse([72, 52, 228, 268], outline=255, width=2)
    d.ellipse([100, 120, 136, 136], outline=255, width=2)
    d.ellipse([164, 120, 200, 136], outline=255, width=2)
    d.line([(150, 140), (140, 185), (160, 185)], fill=255, width=2)
    d.line([(124, 222), (176, 222)], fill=255, width=2)
    return np.asarray(im) > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prompt", nargs="*")
    ap.add_argument("--prompts", help="text file, one prompt per line")
    ap.add_argument("--sketch", help="PNG rough sketch (dark strokes on white)")
    ap.add_argument("--seeds", type=int, default=6, help="seeds tried per prompt")
    ap.add_argument("--keep", type=int, default=1, help="packages kept per prompt")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    prompts = list(a.prompt)
    if a.prompts:
        with open(a.prompts, encoding="utf-8") as f:
            prompts += [ln.strip() for ln in f if ln.strip()]
    if not prompts:
        ap.error("give at least one prompt")
    if not generate.available():
        sys.exit("diffusers/torch not installed: run this on the GPU server")

    if a.sketch:
        im = Image.open(a.sketch).convert("L").resize((CANVAS, CANVAS))
        sketch = np.asarray(im) < 128
    else:
        sketch = default_sketch()
    engine = ShadowEngine(os.path.join(DEMO, "data", "face_db.npz"))
    _, info = engine.shadow(sketch)
    if not info:
        sys.exit("the sketch does not match any face layout")
    dx, dy = info[0]["d"]
    print(f"device: {generate.device_name()}  layout d=({dx}, {dy})  "
          f"CONF_MIN={ghost.CONF_MIN}")

    mins, saved = [], []
    for prompt in prompts:
        kept = 0
        for seed in range(a.seeds):
            t0 = time.time()
            gray = generate.generate(sketch, prompt, seed)
            edges = ghost.image_edges(gray)
            lm, conf = ghost.find_landmarks(edges, dx, dy)
            low = min(conf.values())
            mins.append(low)
            ok = low >= ghost.CONF_MIN
            line = (f"{prompt[:34]!r:36} seed {seed}  {time.time() - t0:4.1f}s  "
                    + " ".join(f"{k}={v:.2f}" for k, v in conf.items())
                    + ("  ok" if ok else "  LOW"))
            if ok and kept < a.keep and not a.dry_run:
                layers = ghost.build_layers(edges, dx, dy, gray, lm)
                pid = package.save(layers, gray, {
                    "prompt": prompt, "seed": seed, "source": "generated",
                    "confidence": conf, "generator": "SD1.5+ControlNet+LCM",
                    "device": generate.device_name()})
                saved.append(pid)
                kept += 1
                line += f"  -> packages/{pid}"
                time.sleep(1)  # package ids are per second
            print(line, flush=True)
            if kept >= a.keep and not a.dry_run:
                break

    m = np.array(mins)
    print(f"\n{len(m)} targets, weakest-landmark confidence: "
          f"p10={np.percentile(m, 10):.2f} median={np.median(m):.2f} "
          f"pass@{ghost.CONF_MIN}={np.mean(m >= ghost.CONF_MIN):.0%}")
    if saved:
        print(f"saved {len(saved)} packages (pending review): python package.py list")


if __name__ == "__main__":
    main()
