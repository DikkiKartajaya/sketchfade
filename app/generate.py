"""B · target generator: rough sketch + prompt -> hidden target face image.

Stable Diffusion 1.5 + ControlNet (scribble) + LCM-LoRA, as in the proposal
(slide 7). The sketch steers the layout the way DeepFaceDrawing /
AniFaceDrawing do, but the result stays hidden: it is only used to build the
construction layers. LCM needs 4 denoising steps, so it also runs on a CPU.

Drawing styles (as in Chen et al. 2025, "Interactive Drawing Guidance for Anime
Illustrations with Diffusion Model") only change the prompt, so every style
uses the same weights and memory. The default is a pencil drawing: a photo's
skin, hair and stubble texture turns into noisy contours and blotchy tone
layers, while a drawing already has clear outlines and simple value masses.

Device: a CUDA GPU with at least MIN_VRAM_GB that can run a kernel, else Apple
MPS, else the CPU.

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
# The sketch sets the layout, not hard edges: at full strength over all steps
# ControlNet turns the closed head outline into a hole or frame (hood, mask,
# torn paper) around a squeezed face. Half strength for the first half of the
# steps keeps eyes, nose and mouth where they were drawn.
CN_SCALE, CN_END = 0.5, 0.5
CFG = 1.5  # LCM-LoRA works with 1-2; above 1 the prompt and NEGATIVE count (2 UNet passes)
NEGATIVE = "hood, hole, frame, mask, torn paper, cropped, deformed face, blurry"
MIN_VRAM_GB = 4  # SD 1.5 + ControlNet in fp16 at 512 px needs ~3.5 GB
STYLES = {  # name: (prompt, extra negative prompt); the first is the default
    "pencil": ("pencil drawing of {}, portrait, front view, centered face, looking at the "
               "viewer, clean confident outlines, simple soft shading, white background",
               "paper sheet, border, photo of a drawing, table, pencil, hand, photo, "
               "photorealistic, skin texture, noise"),
    "realistic": ("portrait photo of {}, front view, centered face, looking at the "
                  "camera, plain white background, soft even light, black and white", ""),
    "anime": ("anime illustration of {}, front view, centered face, looking at the "
              "viewer, plain white background, clean lineart, cel shading, monochrome", ""),
}
DEFAULT_STYLE = next(iter(STYLES))

_pipe = None
_device = None


def available():
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def device():
    """'cuda' for an adequate NVIDIA GPU, 'mps' on Apple silicon, else 'cpu'."""
    global _device
    if _device is None:
        import torch
        _device = "cpu"
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 2**30
            try:  # a torch build without kernels for this GPU fails here
                ok = gb >= MIN_VRAM_GB and (torch.ones(1, device="cuda") + 1).item() == 2
            except RuntimeError:
                ok = False
            if ok:
                _device = "cuda"
        elif torch.backends.mps.is_available():
            _device = "mps"
    return _device


def device_name():
    import torch
    if device() == "cuda":
        p = torch.cuda.get_device_properties(0)
        return f"{p.name} ({p.total_memory / 2**30:.0f} GB)"
    if torch.cuda.is_available():
        return "cpu (CUDA GPU found but unusable: too little VRAM or no kernels in this torch build)"
    if torch.version.cuda is None and not torch.backends.mps.is_available():
        return "cpu (CPU-only torch build)"
    return device()


def load():
    """Build the pipeline once (first call downloads the weights)."""
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import (ControlNetModel, LCMScheduler,
                               StableDiffusionControlNetPipeline)

        dev = device()
        dtype = torch.float16 if dev == "cuda" else torch.float32
        controlnet = ControlNetModel.from_pretrained(CONTROLNET, dtype=dtype)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            BASE, controlnet=controlnet, dtype=dtype, variant="fp16",
            safety_checker=None, requires_safety_checker=False)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(LCM_LORA)
        pipe.fuse_lora()
        pipe.set_progress_bar_config(disable=True)
        _pipe = pipe.to(dev)
    return _pipe


def generate(sketch, prompt, seed, style=DEFAULT_STYLE):
    """sketch: (H, W) bool rough strokes (white on black, as ControlNet expects).
    style: a key of STYLES.
    Returns the target as a (H, W) float32 grayscale image in [0, 1]."""
    import torch

    h, w = sketch.shape
    cond = (Image.fromarray(sketch.astype(np.uint8) * 255)
            .filter(ImageFilter.MaxFilter(3))  # thicker scribble lines
            .resize((RES, RES), Image.BILINEAR).convert("RGB"))
    text, negative = STYLES[style]
    img = load()(text.format(prompt or "a person"), image=cond, num_inference_steps=STEPS,
                 guidance_scale=CFG, negative_prompt=", ".join(filter(None, [NEGATIVE, negative])),
                 controlnet_conditioning_scale=CN_SCALE, control_guidance_end=CN_END,
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
    print(f"model loaded on {device_name()} in {time.time() - t0:.0f}s")
    for style in STYLES:
        t0 = time.time()
        out = generate(np.asarray(im) > 0, "a young woman with short hair", 1, style)
        print(f"{style}: generated in {time.time() - t0:.1f}s")
        Image.fromarray((out * 255).astype(np.uint8)).save(f"target_test_{style}.png")
        print(f"saved target_test_{style}.png")
