"""B · Construction Ghost: turn a hidden target face into construction layers.

The target is a generated face image (generate.py: SD + ControlNet + LCM) or,
when no generator is installed, the top aligned ShadowDraw database face
(edges only, so no shading layer).
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 300  # ShadowDraw database frame
PARTS = ["head", "eyes", "nose", "mouth"]
HEAD, EYES, NOSE, MOUTH = range(4)
LAYERS = ["block-in", "skeleton", "contours", "shading"]
SHADE_FRAC = 0.25  # darkest fraction of the face that becomes the shading layer

# Canonical face layout in the 300x300 database frame, measured from the mean
# database edge image (the faces are pre-aligned crops).
CX, CY, RX, RY = 150, 160, 78, 108
EYE_Y, NOSE_Y, MOUTH_Y = 128, 185, 222


def _peak_row(target, y, x, half_h, half_w):
    """Row with the most edge pixels near y (snaps a guide line to the target)."""
    y0, x0 = max(0, y - half_h), max(0, x - half_w)
    rows = target[y0 : y + half_h, x0 : x + half_w].sum(axis=1)
    return y0 + int(np.argmax(rows)) if rows.size and rows.max() > 0 else y


def _blur(gray, sigma):
    im = Image.fromarray((gray * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sigma))
    return np.asarray(im, np.float32) / 255.0


def image_edges(gray, frac=0.12):
    """Thin edges of a grayscale target: Canny without hysteresis."""
    g = _blur(gray, 1.5)
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    # non-maximum suppression along the dominant gradient axis
    m = np.pad(mag, 1)
    peak = np.where(np.abs(gx) >= np.abs(gy),
                    (mag >= m[1:-1, :-2]) & (mag >= m[1:-1, 2:]),
                    (mag >= m[:-2, 1:-1]) & (mag >= m[2:, 1:-1]))
    return peak & (mag > np.percentile(mag, 100 * (1 - frac)))


def build_layers(target, dx, dy, gray=None):
    """target: (300, 300) bool edges, already aligned to the drawing.
    (dx, dy): ShadowDraw displacement of the sketch; canonical point p sits at p - d.
    gray: the target image in [0, 1] for the shading layer (None: no shading).

    Returns (4, 300, 300) int8, one label image per layer:
    0 = no guide, k + 1 = guide pixel belonging to part k."""
    cx, cy = CX - dx, CY - dy
    eye = _peak_row(target, EYE_Y - dy, cx, 12, 60)
    nose = _peak_row(target, NOSE_Y - dy, cx, 12, 20)
    mouth = _peak_row(target, MOUTH_Y - dy, cx, 12, 35)

    # 1 block-in: big simple shapes
    block = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(block)
    d.ellipse([cx - RX, cy - RY, cx + RX, cy + RY], outline=HEAD + 1, width=2)
    for ex in (cx - 32, cx + 32):
        d.ellipse([ex - 18, eye - 8, ex + 18, eye + 8], outline=EYES + 1, width=2)
    d.line([(cx, eye + 10), (cx - 14, nose), (cx + 14, nose), (cx, eye + 10)],
           fill=NOSE + 1, width=2)
    d.ellipse([cx - 26, mouth - 7, cx + 26, mouth + 7], outline=MOUTH + 1, width=2)

    # 2 skeleton: center axis + proportion lines
    skel = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(skel)
    d.line([(cx, cy - RY), (cx, cy + RY)], fill=HEAD + 1, width=2)
    d.line([(cx - RX, eye), (cx + RX, eye)], fill=EYES + 1, width=2)
    d.line([(cx - 20, nose), (cx + 20, nose)], fill=NOSE + 1, width=2)
    d.line([(cx - 30, mouth), (cx + 30, mouth)], fill=MOUTH + 1, width=2)

    # 3 contours: the target's own edges inside the face, labeled by region
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    inside = ((xx - cx) / (RX * 1.1)) ** 2 + ((yy - cy) / (RY * 1.1)) ** 2 <= 1
    part = np.full((SIZE, SIZE), HEAD, np.int8)
    part[(abs(yy - eye) < 18) & (abs(xx - cx) < 60)] = EYES
    part[(yy >= eye + 18) & (yy < nose + 10) & (abs(xx - cx) < 22)] = NOSE
    part[(abs(yy - mouth) < 15) & (abs(xx - cx) < 38)] = MOUTH
    contours = np.where(target & inside, part + 1, 0)

    # 4 shading: the darkest value masses inside the face
    shading = np.zeros((SIZE, SIZE), np.int8)
    if gray is not None:
        tone = _blur(gray, 2.0)
        dark = tone < np.percentile(tone[inside], 100 * SHADE_FRAC)
        shading = np.where(dark & inside, part + 1, 0)

    return np.stack([np.asarray(block), np.asarray(skel), contours,
                     shading]).astype(np.int8)
