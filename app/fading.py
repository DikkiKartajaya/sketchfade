"""A · Fading Shadow: help that steps back as the user improves.

  1 draw a stroke  ->  2 measure error vs. guide  ->  3 update skill per part
  ->  4 choose help level per part (each part fades on its own)
"""

import json
import os

import numpy as np

from ghost import LAYERS, PARTS

SHADOW, LINES, CHECK, NONE = range(4)
LEVEL_NAMES = ["shadow + correction", "construction lines only",
               "check on request", "no help"]
THRESHOLDS = (0.4, 0.6, 0.8)  # skill at which a part steps down one level
RATE = 0.2  # EMA rate of the skill estimate
ERR_GOOD = 4.0  # mean px error (300 frame) that counts as a perfect stroke
ERR_BAD = 20.0  # mean px error that counts as a failed stroke; also the cap
REACH, PULL = 15.0, 0.5  # stroke correction: snap radius and strength


def match_stroke(pts, labels):
    """Score one stroke against the current layer's guide.

    pts: (N, 2) xy in the 300 frame; labels: (300, 300) layer label image.
    Returns (part index, mean error in px, nearest guide point per stroke point)."""
    ys, xs = np.nonzero(labels)
    if xs.size == 0:
        return 0, ERR_BAD, pts
    if xs.size > 6000:  # filled shading regions: subsample to bound memory
        ys, xs = ys[:: xs.size // 6000 + 1], xs[:: xs.size // 6000 + 1]
    g = np.stack([xs, ys], 1).astype(np.float32)
    d2 = ((pts[:, None, :] - g[None]) ** 2).sum(-1)
    idx = d2.argmin(1)
    dist = np.sqrt(d2[np.arange(len(pts)), idx])
    part = int(np.bincount(labels[ys[idx], xs[idx]] - 1).argmax())
    return part, float(np.minimum(dist, ERR_BAD).mean()), g[idx]


def correct(pts, nearest, strength=PULL):
    """PortraitSketch-style correction: pull nearby points toward the guide."""
    d = np.linalg.norm(nearest - pts, axis=1, keepdims=True)
    return pts + np.where(d < REACH, strength, 0.0) * (nearest - pts)


def ghost(layers, layer, levels, check=False):
    """What the user sees for the current layer, part by part (alpha in [0, 1])."""
    out = np.zeros(layers.shape[1:], np.float32)
    full = 0.45 if LAYERS[layer] == "shading" else 1.0  # filled areas: lighter
    for p, lvl in enumerate(levels):
        if lvl == SHADOW or (check and lvl == CHECK):
            out[layers[layer] == p + 1] = full
        elif lvl == LINES:
            # construction lines only: block-in (+ skeleton once past layer 1)
            for l in range(min(layer, 1) + 1):
                out[layers[l] == p + 1] = 0.6
    return out


class Skill:
    """Per-part skill estimate in [0, 1], persisted so it carries across sessions."""

    def __init__(self, path):
        self.path = path
        self.s = {p: 0.0 for p in PARTS}
        if os.path.exists(path):
            with open(path) as f:
                self.s.update(json.load(f))

    def update(self, part, err):
        q = np.clip((ERR_BAD - err) / (ERR_BAD - ERR_GOOD), 0, 1)
        self.s[PARTS[part]] += RATE * (float(q) - self.s[PARTS[part]])
        with open(self.path, "w") as f:
            json.dump(self.s, f, indent=1)

    def levels(self):
        return [sum(self.s[p] >= t for t in THRESHOLDS) for p in PARTS]
