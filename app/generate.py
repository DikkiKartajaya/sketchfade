"""B · target generator: rough sketch + prompt -> hidden target face image.

Stable Diffusion 1.5 + ControlNet (scribble) + LCM-LoRA, as in the proposal
(slide 7). The sketch steers the layout the way DeepFaceDrawing /
AniFaceDrawing do, but the result stays hidden: it is only used to build the
construction layers. LCM needs 4 denoising steps, so it also runs on a CPU.

  python generate.py      # downloads the models once (~3 GB) and runs a test
"""

import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET = "lllyasviel/control_v11p_sd15_scribble"
LCM_LORA = "latent-consistency/lcm-lora-sdv1-5"
RES = 512
STEPS = 4
PROMPT = ("portrait photo of {}, front view, centered face, looking at the camera, "
          "plain white background, soft even light, black and white")

_pipe = None


def available():
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def device_name():
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def load():
    """Build the pipeline once (first call downloads the weights)."""
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import (ControlNetModel, LCMScheduler,
                               StableDiffusionControlNetPipeline)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        controlnet = ControlNetModel.from_pretrained(CONTROLNET, dtype=dtype)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            BASE, controlnet=controlnet, dtype=dtype, variant="fp16",
            safety_checker=None, requires_safety_checker=False)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(LCM_LORA)
        pipe.fuse_lora()
        pipe.set_progress_bar_config(disable=True)
        _pipe = pipe.to(device)
    return _pipe


def generate(sketch, prompt, seed):
    """sketch: (H, W) bool rough strokes (white on black, as ControlNet expects).
    Returns the target as a (H, W) float32 grayscale image in [0, 1]."""
    import torch

    h, w = sketch.shape
    cond = (Image.fromarray(sketch.astype(np.uint8) * 255)
            .filter(ImageFilter.MaxFilter(3))  # thicker scribble lines
            .resize((RES, RES), Image.BILINEAR).convert("RGB"))
    img = load()(PROMPT.format(prompt or "a person"), image=cond,
                 num_inference_steps=STEPS, guidance_scale=1.0,
                 generator=torch.Generator().manual_seed(seed)).images[0]
    gray = img.convert("L").resize((w, h), Image.BILINEAR)
    return np.asarray(gray, np.float32) / 255.0


if __name__ == "__main__":
    # self-test: a rough face scribble -> target_test.png
    im = Image.new("L", (300, 300), 0)
    d = ImageDraw.Draw(im)
    d.ellipse([72, 52, 228, 268], outline=255, width=2)
    d.ellipse([100, 120, 136, 136], outline=255, width=2)
    d.ellipse([164, 120, 200, 136], outline=255, width=2)
    d.line([(150, 140), (140, 185), (160, 185)], fill=255, width=2)
    d.line([(124, 222), (176, 222)], fill=255, width=2)
    t0 = time.time()
    load()
    print(f"model loaded in {time.time() - t0:.0f}s")
    t0 = time.time()
    out = generate(np.asarray(im) > 0, "a young woman with short hair", seed=1)
    print(f"generated in {time.time() - t0:.1f}s")
    Image.fromarray((out * 255).astype(np.uint8)).save("target_test.png")
    print("saved target_test.png")
