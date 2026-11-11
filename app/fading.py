"""A · Fading Shadow: help that steps back as unassisted performance improves.

Final proposal (Lecture 4), Algorithm 2:

  practice with help  ->  unassisted probe at a fixed checkpoint (both groups)
  ->  two-way error + coverage per part  ->  success / failure per part
  ->  2 successes in a row lower that part's level, 2 failures raise it
      (adaptive group only; the fixed group stays at L3)

Assisted strokes never change the level: an assisted trace does not show
learning, so only the unassisted redraw decides the help.
"""

import json
import os
import time

import numpy as np

from ghost import FILLED, LAYERS, PARTS

L0, L1, L2, L3 = range(4)
LEVEL_NAMES = ["L0 independent", "L1 on request", "L2 reduced", "L3 strong"]
LEVEL_HELP = ["side reference only", "brief Check only", "guide 0.25",
              "guide 0.60 + correction"]
GUIDE_ALPHA = {L3: 0.60, L2: 0.25}  # overlay opacity per level
CHECK_ALPHA = 0.60  # L1: what Check reveals (for CHECK_MS on the client)
GROUPS = ("adaptive", "fixed")

# probe outcome (errors are fractions of the 300 px frame, so 0.03 = 9 px)
FRAME = 300.0
ERR_OK, ERR_FAIL = 0.03, 0.06
COV_OK, COV_FAIL = 0.80, 0.50
TOL = 0.02  # a guide pixel is covered if a drawn pixel is within 6 px
CAP = 0.10  # per-point distance cap (30 px), so one stray line cannot dominate
STREAK = 2  # consecutive successes / failures needed to change a level

# per-stroke matching during practice (part assignment, correction, logging)
REACH, PULL = 15.0, 0.5  # L3 correction: snap radius (px) and strength
MAX_PTS = 4000  # subsample point sets above this to bound memory


def _sub(p, n=MAX_PTS):
    return p if len(p) <= n else p[:: len(p) // n + 1]


def nn_dist(a, b):
    """For each point in a (N, 2), the distance to the nearest point in b (M, 2)."""
    if len(a) == 0:
        return np.zeros(0, np.float32)
    if len(b) == 0:
        return np.full(len(a), np.inf, np.float32)
    b = _sub(b).astype(np.float32)
    out = np.empty(len(a), np.float32)
    for i in range(0, len(a), 512):
        d2 = ((a[i : i + 512, None, :] - b[None]) ** 2).sum(-1)
        out[i : i + 512] = np.sqrt(d2.min(1))
    return out


def guide_mask(labels):
    """Pixels to draw for one layer. Filled layers (shading, shadow) count by their outline."""
    if not (labels > 0).any():
        return labels
    filled = labels > 0
    inner = filled.copy()
    inner[1:-1, 1:-1] &= (filled[:-2, 1:-1] & filled[2:, 1:-1]
                          & filled[1:-1, :-2] & filled[1:-1, 2:])
    edge = filled & ~inner if inner.sum() > 0.3 * filled.sum() else filled
    return np.where(edge, labels, 0)


def match_stroke(pts, labels):
    """Score one practice stroke against the current layer (one-way, for logging).

    pts: (N, 2) xy in the 300 frame; labels: (300, 300) layer label image.
    Returns (part index, mean error in px, nearest guide point per stroke point)."""
    ys, xs = np.nonzero(labels)
    if xs.size == 0:
        return 0, CAP * FRAME, pts
    step = xs.size // 6000 + 1  # filled shading regions: subsample
    ys, xs = ys[::step], xs[::step]
    g = np.stack([xs, ys], 1).astype(np.float32)
    idx = np.empty(len(pts), np.int64)
    for i in range(0, len(pts), 256):
        idx[i : i + 256] = ((pts[i : i + 256, None, :] - g[None]) ** 2).sum(-1).argmin(1)
    dist = np.linalg.norm(g[idx] - pts, axis=1)
    part = int(np.bincount(labels[ys[idx], xs[idx]] - 1, minlength=len(PARTS)).argmax())
    return part, float(np.minimum(dist, CAP * FRAME).mean()), g[idx]


def correct(pts, nearest, strength=PULL):
    """PortraitSketch-style correction (L3 only): pull nearby points to the guide."""
    d = np.linalg.norm(nearest - pts, axis=1, keepdims=True)
    return pts + np.where(d < REACH, strength, 0.0) * (nearest - pts)


def outcome(err, cov):
    if err <= ERR_OK and cov >= COV_OK:
        return "success"
    if err >= ERR_FAIL or cov < COV_FAIL:
        return "failure"
    return "neutral"


def score_probe(drawn, labels):
    """Performance estimator: two-way error + coverage per part for one probe.

    drawn: (300, 300) bool, the unassisted redraw; labels: the probed layer.
    Two-way error = mean of (drawn -> guide of that part) and (guide -> any
    drawn pixel), each point capped at CAP, as a fraction of the frame.
    Coverage = share of that part's guide pixels with a drawn pixel within TOL.
    Drawn pixels belong to the part of their nearest guide pixel."""
    guide = guide_mask(labels)
    gy, gx = np.nonzero(guide)
    g = np.stack([gx, gy], 1).astype(np.float32)
    gpart = guide[gy, gx] - 1
    dy, dx = np.nonzero(drawn)
    d = np.stack([dx, dy], 1).astype(np.float32)
    d = _sub(d)
    # part of each drawn pixel = part of its nearest guide pixel
    dpart = np.full(len(d), -1)
    if len(g) and len(d):
        gs = _sub(np.arange(len(g)))
        for i in range(0, len(d), 512):
            d2 = ((d[i : i + 512, None, :] - g[gs][None]) ** 2).sum(-1)
            dpart[i : i + 512] = gpart[gs[d2.argmin(1)]]
    cap = CAP * FRAME
    out = {}
    for p, name in enumerate(PARTS):
        gp = g[gpart == p]
        if len(gp) == 0:
            continue  # this layer has no guide for this part (nothing to judge)
        dp = d[dpart == p]
        g2d = np.minimum(nn_dist(gp, d), cap)
        if len(dp):
            d2g = np.minimum(nn_dist(dp, gp), cap)
            err = 0.5 * (d2g.mean() + g2d.mean()) / FRAME
        else:
            err = CAP
        cov = float((g2d <= TOL * FRAME).mean())
        out[name] = {"err": round(float(err), 4), "cov": round(cov, 3),
                     "n_drawn": int(len(dp)), "n_guide": int(len(gp)),
                     "outcome": outcome(err, cov)}
    return out


def ghost(layers, layer, levels, check=False):
    """Overlay alpha for the current layer, part by part (alpha in [0, 1]).
    L3: guide at 0.60, L2: guide at 0.25, L1: nothing unless Check, L0: nothing."""
    out = np.zeros(layers.shape[1:], np.float32)
    shade = 0.6 if LAYERS[layer] in FILLED else 1.0  # filled areas: lighter
    for p, lvl in enumerate(levels):
        a = GUIDE_ALPHA.get(lvl, CHECK_ALPHA if check and lvl == L1 else 0.0)
        if a:
            out[layers[layer] == p + 1] = a * shade
    return out


class Controller:
    """Fading controller for one participant, persisted across the three days."""

    def __init__(self, path, participant, group):
        self.path, self.participant = path, participant
        self.group = group
        self.level = {p: L3 for p in PARTS}
        self.succ = {p: 0 for p in PARTS}
        self.fail = {p: 0 for p in PARTS}
        if path and os.path.exists(path):
            with open(path) as f:
                saved = json.load(f)
            self.group = saved["group"]  # the assignment never changes
            for k in ("level", "succ", "fail"):
                getattr(self, k).update(saved[k])
        if self.group not in GROUPS:
            raise ValueError(f"group must be one of {GROUPS}")
        self.save()

    def save(self):
        if self.path is None:
            return
        with open(self.path, "w") as f:
            json.dump({"participant": self.participant, "group": self.group,
                       "level": self.level, "succ": self.succ, "fail": self.fail,
                       "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)

    def levels(self):
        return [self.level[p] for p in PARTS]

    def update(self, results):
        """Apply one probe's outcomes. Returns [(part, old, new)] level changes.
        Streaks are consecutive: a neutral or opposite outcome breaks them.
        The fixed group keeps its streaks for the log but never changes level."""
        changes = []
        for part, r in results.items():
            o = r["outcome"]
            self.succ[part] = self.succ[part] + 1 if o == "success" else 0
            self.fail[part] = self.fail[part] + 1 if o == "failure" else 0
            if self.group != "adaptive":
                continue
            old = self.level[part]
            if self.succ[part] >= STREAK and old > L0:
                self.level[part] = old - 1
            elif self.fail[part] >= STREAK and old < L3:
                self.level[part] = old + 1
            if self.level[part] != old:
                self.succ[part] = self.fail[part] = 0
                changes.append((part, old, self.level[part]))
        self.save()
        return changes
