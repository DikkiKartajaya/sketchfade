"""BiCE descriptor (Zitnick, ECCV 2010) with the low-dimensional parameters
used by ShadowDraw (Lee et al., SIGGRAPH 2011, sec 3.1):

  n_theta = 4 edge orientations,
  n_perp  = 18 positions perpendicular to the edge,
  n_tan   = 6 positions tangent to the edge,
  => 4 * 18 * 6 = 432 binary bits, top 20% of buckets set to 1.

Input patches are line/edge drawings (white ink on black, float in [0, 1]).
Orientation and magnitude maps are computed once per full image with
`orientation_maps`, then per-patch descriptors are extracted from crops, which
is much faster than per-patch filtering.

Also includes the simplified descriptor from BiCETest-master (blur ->
subsample 18x18 -> fixed threshold) for comparison.
"""

import numpy as np


def _blur_separable(img, sigma):
    radius = max(1, int(3.0 * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    p = np.pad(img, ((0, 0), (radius, radius)), mode="edge")
    img = np.stack([np.convolve(r, kernel, mode="valid") for r in p])
    p = np.pad(img, ((radius, radius), (0, 0)), mode="edge")
    img = np.stack([np.convolve(c, kernel, mode="valid") for c in p.T]).T
    return img


def orientation_maps(img, sigma=1.5):
    """Edge tangent orientation in [0, pi) and gradient magnitude, per pixel.

    img: float32 (H, W) line drawing, ink ~1 on background ~0.
    """
    img = _blur_separable(img.astype(np.float32), sigma)
    gy, gx = np.gradient(img)
    mag = np.hypot(gx, gy)
    # gradient is normal to the stroke; tangent = normal rotated by 90 deg
    theta = (np.arctan2(gy, gx) + 0.5 * np.pi) % np.pi
    return theta, mag


def bice_descriptor(theta, mag, n_theta=4, n_perp=18, n_tan=6, top_frac=0.2):
    """Compute one BiCE descriptor from orientation/magnitude patch crops.

    theta, mag: (S, S) crops of the maps from `orientation_maps`.
    Orientation is soft-assigned: each edge pixel contributes linearly to the
    two nearest orientation channels (as in ShadowDraw's edge decomposition).
    Returns a flat bool array of n_theta * n_perp * n_tan bits.
    """
    size = theta.shape[0]
    half = size / 2.0

    ys, xs = np.nonzero(mag > 1e-4)
    hist = np.zeros((n_theta, n_perp, n_tan), dtype=np.float32)
    if xs.size == 0:
        return hist.reshape(-1) > 0  # all zeros

    w = mag[ys, xs]
    th = theta[ys, xs]
    dx = xs.astype(np.float32) - half + 0.5
    dy = ys.astype(np.float32) - half + 0.5

    # soft orientation assignment between the two adjacent channels
    pos = th / (np.pi / n_theta)
    t_lo = np.floor(pos).astype(int) % n_theta
    w_hi = (pos - np.floor(pos)).astype(np.float32)
    for t in range(n_theta):
        sel_lo = t_lo == t
        sel_hi = (t_lo + 1) % n_theta == t
        sel = sel_lo | sel_hi
        if not np.any(sel):
            continue
        ws = np.where(sel_lo, 1.0 - w_hi, 0.0) + np.where(sel_hi, w_hi, 0.0)
        ws = (ws * w)[sel]

        phi = t * np.pi / n_theta  # tangent direction of this channel
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        tan_c = dx[sel] * cos_p + dy[sel] * sin_p  # along the edge
        perp_c = -dx[sel] * sin_p + dy[sel] * cos_p  # across the edge

        pi_idx = np.floor((perp_c + half) / size * n_perp).astype(int)
        ti_idx = np.floor((tan_c + half) / size * n_tan).astype(int)
        ok = (pi_idx >= 0) & (pi_idx < n_perp) & (ti_idx >= 0) & (ti_idx < n_tan)
        np.add.at(hist[t], (pi_idx[ok], ti_idx[ok]), ws[ok])

    flat = hist.reshape(-1)
    n_on = max(1, int(top_frac * flat.size))
    order = np.argsort(flat)[::-1]
    desc = np.zeros(flat.size, dtype=bool)
    keep = order[:n_on]
    desc[keep[flat[keep] > 0]] = True  # never turn on empty buckets
    return desc


def pack_descriptors(bools):
    """(N, D) bool -> (N, ceil(D/8)) uint8 packed bits."""
    return np.packbits(np.asarray(bools, dtype=bool), axis=-1)


def jaccard_packed(db_packed, db_pop, q_packed):
    """Jaccard similarity of one packed query against packed rows.

    db_packed: (N, B) uint8, db_pop: (N,) precomputed popcounts,
    q_packed: (B,) uint8. Returns (N,) float32.
    """
    inter = np.bitwise_count(db_packed & q_packed).sum(axis=1, dtype=np.int32)
    q_pop = int(np.bitwise_count(q_packed).sum())
    union = db_pop + q_pop - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0).astype(np.float32)


def simple_descriptor(blurred_patch, grid=18, threshold=64.0 / 255.0):
    """The BiCETest-master 'descriptor': blur -> subsample -> fixed threshold.

    blurred_patch: (S, S) crop of the sigma=3 blurred image, float [0, 1].
    """
    size = blurred_patch.shape[0]
    idx = (np.arange(grid) + 0.5) / grid * size
    idx = idx.astype(int)
    sub = blurred_patch[np.ix_(idx, idx)]
    return (sub > threshold).reshape(-1)


def jaccard_scores(db, query):
    """db: (N, D) bool, query: (D,) bool -> (N,) Jaccard similarities."""
    dbp = pack_descriptors(db)
    return jaccard_packed(dbp, np.bitwise_count(dbp).sum(axis=1, dtype=np.int32),
                          pack_descriptors(query))
