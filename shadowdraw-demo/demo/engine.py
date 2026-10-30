"""ShadowDraw online matching engine (offline core, no UI).

Everything runs in the canonical 300x300 DB frame; a UI drawing canvas of
480x480 maps onto it by uniform 0.625 scaling.

Pipeline per query (ShadowDraw sec 3.2, brute-force retrieval instead of
min-hash — fine at 176k descriptors):
  1. query descriptors on a 17x17 grid (60 px patches, stride 15 = half the
     DB stride, so query grid position k aligns with DB grid j at k = 2j)
  2. Jaccard voting into an (image, offset) histogram; offsets are
     (k - 2j) in half-stride units, kept when |offset| <= 4  (+-60 px)
  3. top NC images by best-offset vote -> candidate set
  4. per candidate: 1D generalized Hough refinement of (dx, dy) (eq. 2-3)
  5. per candidate: oriented-edge correlation weights (eq. 6-11)
  6. shadow S = sum_i W_i E'_i, display filtering (eq. 12)
"""

import numpy as np

from bice import bice_descriptor, jaccard_packed, orientation_maps, pack_descriptors

CANVAS = 300
PATCH = 60
DB_STRIDE = 30
Q_STRIDE = 15
Q_GRID = 17  # 17x17 query grid, exact cover of 300px
MAX_OFF = 4  # +-4 half-strides = +-60 px
N_CANDIDATES = 100
N_ORIENT = 8  # orientation channels for weighting
QR = 4  # reduction factor for weight maps (300 -> 75)
GAMMA, KAPPA = 0.5, 2.0
SIGMA_S = 1.25 * Q_STRIDE  # oriented-map blur (full-res px)
SIGMA_V = 4.0 * Q_STRIDE  # V_i blur (full-res px)


def _gauss1d(sigma):
    r = max(1, int(3 * sigma + 0.5))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur2d(img, sigma):
    k = _gauss1d(sigma)
    r = len(k) // 2
    p = np.pad(img, ((0, 0), (r, r)), mode="constant")
    img = np.stack([np.convolve(row, k, "valid") for row in p])
    p = np.pad(img, ((r, r), (0, 0)), mode="constant")
    return np.stack([np.convolve(col, k, "valid") for col in p.T]).T


def shift_image(img, dx, dy):
    """out(p) = img(p + (dx, dy)), zero-padded. dx, dy integers."""
    h, w = img.shape
    out = np.zeros_like(img)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    out[ys0 - dy : ys1 - dy, xs0 - dx : xs1 - dx] = img[ys0:ys1, xs0:xs1]
    return out


def oriented_maps(edge_map, theta=None, mag=None):
    """Decompose a binary edge map into N_ORIENT soft-assigned channels at
    quarter resolution, blurred with SIGMA_S and clipped to the single-line
    response (paper sec 3.2, image weighting)."""
    if theta is None:
        theta, mag = orientation_maps(edge_map.astype(np.float32))
    ys, xs = np.nonzero(edge_map)
    q = CANVAS // QR
    chans = np.zeros((N_ORIENT, q, q), dtype=np.float32)
    if xs.size:
        th = theta[ys, xs]
        pos = th / (np.pi / N_ORIENT)
        t_lo = np.floor(pos).astype(int) % N_ORIENT
        w_hi = (pos - np.floor(pos)).astype(np.float32)
        cy, cx = ys // QR, xs // QR
        for t in range(N_ORIENT):
            m_lo = t_lo == t
            m_hi = (t_lo + 1) % N_ORIENT == t
            w = np.where(m_lo, 1 - w_hi, 0) + np.where(m_hi, w_hi, 0)
            sel = w > 0
            np.add.at(chans[t], (cy[sel], cx[sel]), w[sel])
    sig = SIGMA_S / QR
    for t in range(N_ORIENT):
        chans[t] = blur2d(chans[t], sig)
    return np.clip(chans, 0, _single_line_peak(sig), out=chans)


_PEAK_CACHE = {}


def _single_line_peak(sig):
    """Max blurred response of an ideal 1-px line (QR ink per reduced cell)."""
    if sig not in _PEAK_CACHE:
        q = CANVAS // QR
        line = np.zeros((q, q), dtype=np.float32)
        line[:, q // 2] = QR
        _PEAK_CACHE[sig] = float(blur2d(line, sig).max())
    return _PEAK_CACHE[sig]


class _LRU(dict):
    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        self.order = []

    def get_or(self, key, fn):
        if key in self:
            return self[key]
        val = fn()
        self[key] = val
        self.order.append(key)
        if len(self.order) > self.cap:
            del self[self.order.pop(0)]
        return val


class ShadowEngine:
    def __init__(self, db_path):
        db = np.load(db_path, allow_pickle=False)
        self.names = db["names"]
        self.bitmaps = db["bitmaps"]
        self.desc = db["desc"]
        self.desc_pop = db["desc_pop"]
        self.d_img = db["desc_img"].astype(np.int64)
        self.d_gx = db["desc_gx"].astype(np.int64)
        self.d_gy = db["desc_gy"].astype(np.int64)
        self.n_images = len(self.names)
        # per-image caches (repeat candidates across interactive updates)
        self._sincos_cache = _LRU(200)  # i -> (sin*mag, cos*mag) float16
        self._chans_cache = _LRU(400)  # i -> unshifted oriented maps float16
        self.last_debug = None
        self.last_changed = []
        self.reset()

    def reset(self):
        """Clear the incremental-voting session state."""
        self._hist = np.zeros((self.n_images, 2 * MAX_OFF + 1, 2 * MAX_OFF + 1),
                              dtype=np.float32)
        self._cell_key = {}  # (kx, ky) -> patch bytes
        self._cell_votes = {}  # (kx, ky) -> (img, oy, ox, score) arrays
        self._n_inky = 0

    def edge_map(self, i):
        return np.unpackbits(self.bitmaps[i]).reshape(CANVAS, CANVAS).astype(bool)

    def _cand_fft(self, i):
        """Per-row FFTs of candidate i's sin/cos oriented maps (fb_sin rows
        for Tx, fb_cos columns for Ty). The float16 maps are cached per image;
        the two rffts are recomputed (cheap, keeps the cache small)."""
        def make():
            em = self.edge_map(i)
            th, mg = orientation_maps(em.astype(np.float32))
            return ((np.sin(th) * mg).astype(np.float16),
                    (np.cos(th) * mg).astype(np.float16))
        b_sin, b_cos = self._sincos_cache.get_or(i, make)
        return (self._rows_fft(b_sin.astype(np.float32), axis=0),
                self._rows_fft(b_cos.astype(np.float32), axis=1))

    def _cand_chans(self, i):
        return self._chans_cache.get_or(
            i, lambda: oriented_maps(self.edge_map(i)).astype(np.float16))

    # ---- stage 1-3: retrieval (incremental voting) -----------------------

    def vote(self, drawing, vote_top=80):
        """Update the running vote histogram for the new drawing.

        Only grid cells whose pixels changed since the last call are
        recomputed (the paper's incremental scheme). Call reset() to start a
        new drawing session. Returns (hist, inky patch count)."""
        changed = []
        for ky in range(Q_GRID):
            for kx in range(Q_GRID):
                y, x = ky * Q_STRIDE, kx * Q_STRIDE
                key = np.packbits(drawing[y : y + PATCH, x : x + PATCH]).tobytes()
                if self._cell_key.get((kx, ky)) != key:
                    self._cell_key[(kx, ky)] = key
                    changed.append((kx, ky))
        self.last_changed = list(changed)
        if changed:
            theta, mag = orientation_maps(drawing.astype(np.float32))
        for kx, ky in changed:
            old = self._cell_votes.pop((kx, ky), None)
            if old is not None:
                np.subtract.at(self._hist, (old[0], old[1], old[2]), old[3])
                self._n_inky -= 1
            y, x = ky * Q_STRIDE, kx * Q_STRIDE
            if drawing[y : y + PATCH, x : x + PATCH].sum() < 10:
                continue
            d = bice_descriptor(theta[y : y + PATCH, x : x + PATCH],
                                mag[y : y + PATCH, x : x + PATCH])
            if not d.any():
                continue
            scores = jaccard_packed(self.desc, self.desc_pop, pack_descriptors(d))
            top = np.argpartition(scores, -vote_top)[-vote_top:]
            ox = kx - 2 * self.d_gx[top]
            oy = ky - 2 * self.d_gy[top]
            ok = (np.abs(ox) <= MAX_OFF) & (np.abs(oy) <= MAX_OFF)
            entry = (self.d_img[top[ok]], oy[ok] + MAX_OFF, ox[ok] + MAX_OFF,
                     scores[top[ok]])
            np.add.at(self._hist, (entry[0], entry[1], entry[2]), entry[3])
            self._cell_votes[(kx, ky)] = entry
            self._n_inky += 1
        return self._hist, self._n_inky

    def candidates(self, hist, n=N_CANDIDATES):
        best = hist.reshape(self.n_images, -1).max(axis=1)
        order = np.argsort(best)[::-1][:n]
        order = order[best[order] > 0]
        cands = []
        for i in order:
            flat = int(np.argmax(hist[i]))
            oy, ox = divmod(flat, 2 * MAX_OFF + 1)
            # initial displacement d: drawing coords + d = DB coords
            dx = -(ox - MAX_OFF) * Q_STRIDE
            dy = -(oy - MAX_OFF) * Q_STRIDE
            cands.append((int(i), dx, dy, float(best[i])))
        return cands

    # ---- stage 4: alignment refinement (eq. 2-3) ------------------------
    #
    # T(x; d) = sum_p a(p) b(p + d + x e_axis)
    #         = sum_y corr_row(a[y], b[y + d_perp])[d_along + x]
    # so with per-row FFTs FA (query, conj) and FB (candidate):
    #   - the perpendicular component of d is a row re-pairing (roll of FB rows)
    #   - the along-axis component of d is an index offset into the
    #     inverse-FFT'd correlation
    # FA is computed once per query, FB is cached per candidate image.

    # 512 suffices: circular correlation of 300-px rows is alias-free for
    # |shift| <= 512 - 300 = 212, and our shifts stay under ~90 px.
    FFT_N = 512

    @classmethod
    def _rows_fft(cls, m, axis):
        """Per-row rfft of m (axis=0) or m.T (axis=1)."""
        a = m if axis == 0 else m.T
        return np.fft.rfft(a, cls.FFT_N, axis=1).astype(np.complex64)

    @classmethod
    def _corr_scores(cls, fa_conj, fb, d_along, d_perp, rng=2 * Q_STRIDE,
                     sigma_h=2.0):
        """Blurred T(x) samples for x in [-rng, rng] (the 1D Hough curve)."""
        h = fb.shape[0]
        # row re-pairing: shifted_fb[y] = fb[y + d_perp], zero outside
        g = np.zeros(fb.shape[1], dtype=np.complex64)
        y0, y1 = max(0, -d_perp), min(h, h - d_perp)
        if y1 > y0:
            g = (fa_conj[y0:y1] * fb[y0 + d_perp : y1 + d_perp]).sum(axis=0)
        corr = np.fft.irfft(g, cls.FFT_N)
        idx = (d_along + np.arange(-rng, rng + 1)) % cls.FFT_N
        scores = corr[idx].astype(np.float32)
        return np.convolve(scores, _gauss1d(sigma_h), "same")

    @classmethod
    def _refine_1d_fft(cls, fa_conj, fb, d_along, d_perp, rng=2 * Q_STRIDE,
                       sigma_h=2.0):
        """Peak location of T(x) for x in [-rng, rng] (sub-sample)."""
        scores = cls._corr_scores(fa_conj, fb, d_along, d_perp, rng, sigma_h)
        p = int(np.argmax(scores))
        if 0 < p < len(scores) - 1:
            y0_, y1_, y2_ = scores[p - 1 : p + 2]
            denom = y0_ - 2 * y1_ + y2_
            if abs(denom) > 1e-9:
                p += 0.5 * (y0_ - y2_) / denom
        return p - rng

    @staticmethod
    def _refine_1d(a_map, b_map, d, axis, rng=2 * Q_STRIDE, sigma_h=2.0):
        """Reference implementation used by tests (same math, no caching)."""
        fa_conj = np.conj(ShadowEngine._rows_fft(a_map, axis))
        fb = ShadowEngine._rows_fft(b_map, axis)
        d_along, d_perp = (d[0], d[1]) if axis == 0 else (d[1], d[0])
        return ShadowEngine._refine_1d_fft(fa_conj, fb, d_along, d_perp,
                                           rng, sigma_h)

    def _refine_fft(self, fa_sin_conj, fa_cos_conj, fb_sin, fb_cos, d0):
        """Two iterations of 1D Hough refinement on cached FFTs."""
        dx, dy = float(d0[0]), float(d0[1])
        for _ in range(2):
            dx += self._refine_1d_fft(fa_sin_conj, fb_sin, round(dx), round(dy))
            dy += self._refine_1d_fft(fa_cos_conj, fb_cos, round(dy), round(dx))
        return round(dx), round(dy)

    def refine(self, drawing_maps, cand_edge, cand_maps, d0):
        """Convenience wrapper used by tests."""
        theta_q, mag_q = drawing_maps
        theta_c, mag_c = cand_maps
        return self._refine_fft(
            np.conj(self._rows_fft(np.sin(theta_q) * mag_q, axis=0)),
            np.conj(self._rows_fft(np.cos(theta_q) * mag_q, axis=1)),
            self._rows_fft(np.sin(theta_c) * mag_c, axis=0),
            self._rows_fft(np.cos(theta_c) * mag_c, axis=1), d0)

    # ---- stage 5: weighting (eq. 6-11) ----------------------------------

    @staticmethod
    def correlations(q_chans, c_chans):
        pos = (q_chans * c_chans).sum(axis=0)
        neg = (q_chans * np.roll(c_chans, N_ORIENT // 2, axis=0)).sum(axis=0)
        return pos, neg

    # ---- full pipeline ---------------------------------------------------

    def shadow(self, drawing, n_candidates=N_CANDIDATES, debug=False,
               collect=False, collect_k=6):
        """Compute the shadow. With collect=True, a full record of the
        pipeline's intermediates (top collect_k candidates) is stored on
        self.last_debug for visualization; otherwise last_debug is None."""
        import time as _time

        self.last_debug = None
        drawing = drawing.astype(bool)
        t0 = _time.time()
        hist, n_q = self.vote(drawing)
        cands = self.candidates(hist, n_candidates)
        t_vote = _time.time()
        if not cands:
            return np.zeros((CANVAS, CANVAS), np.float32), []

        q_theta, q_mag = orientation_maps(drawing.astype(np.float32))
        fa_sin_conj = np.conj(self._rows_fft(np.sin(q_theta) * q_mag, axis=0))
        fa_cos_conj = np.conj(self._rows_fft(np.cos(q_theta) * q_mag, axis=1))
        q_chans = oriented_maps(drawing, q_theta, q_mag)

        aligned, h_scores, v_maps, refined = [], [], [], []
        for i, dx0, dy0, _v in cands:
            fb_sin, fb_cos = self._cand_fft(i)
            dx, dy = self._refine_fft(fa_sin_conj, fa_cos_conj,
                                      fb_sin, fb_cos, (dx0, dy0))
            em_al = shift_image(self.edge_map(i).astype(np.float32),
                                dx, dy).astype(bool)
            # oriented channels: shift the cached unshifted maps at quarter res
            c_chans = np.empty((N_ORIENT, CANVAS // QR, CANVAS // QR), np.float32)
            raw = self._cand_chans(i).astype(np.float32)
            qdx, qdy = round(dx / QR), round(dy / QR)
            for t in range(N_ORIENT):
                c_chans[t] = shift_image(raw[t], qdx, qdy)
            pos, neg = self.correlations(q_chans, c_chans)
            v_map = blur2d(pos, SIGMA_V / QR) + 1e-6
            aligned.append(em_al)
            h_scores.append(float((pos - neg).sum()))
            v_maps.append(v_map)
            refined.append((dx, dy))
        t_align = _time.time()

        h = np.array(h_scores, dtype=np.float32)
        h_star = float(np.sort(h)[-5:].mean()) if len(h) >= 5 else float(h.max())
        if h_star <= 0:
            return np.zeros((CANVAS, CANVAS), np.float32), []
        v = np.maximum(0.0, (h - GAMMA * h_star) / (h_star - GAMMA * h_star)) ** KAPPA

        # eq. 11: eps calibrated as sum(v) of a single ~250 px stroke, so a
        # bare first stroke renders its shadow at roughly half strength
        eps = 15.0
        alpha = v.sum() / (eps + v.sum())

        denom = np.zeros_like(v_maps[0])
        for vi, vm in zip(v, v_maps):
            denom += vi * vm
        denom = np.maximum(denom, 1e-9)

        # composite in descending-v order so debug partial sums are meaningful
        order = np.argsort(-v)
        S = np.zeros((CANVAS, CANVAS), dtype=np.float32)
        partials, contribs = [], {}
        for rank, ci in enumerate(order):
            vi, vm, em_al = v[ci], v_maps[ci], aligned[ci]
            if vi <= 0:
                break
            w_small = alpha * vi * vm / denom
            w_full = np.repeat(np.repeat(w_small, QR, axis=0), QR, axis=1)
            contrib = w_full * em_al
            S += contrib
            if collect and rank < collect_k:
                contribs[int(ci)] = contrib
                partials.append(S.copy())

        info = [
            {"name": str(self.names[cands[ci][0]]), "img": int(cands[ci][0]),
             "d0": (cands[ci][1], cands[ci][2]), "d": refined[ci],
             "h": h_scores[ci], "v": float(v[ci])}
            for ci in order
        ]
        if debug:
            print(f"query patches: {n_q}, candidates: {len(cands)}, "
                  f"h*: {h_star:.1f}, alpha: {alpha:.2f}")
            for r in info[:5]:
                print(f"  {r['name']} v={r['v']:.2f} h={r['h']:.1f} d={r['d']}")

        if collect:
            k = min(collect_k, len(order))
            dbg_cands = []
            for rank in range(k):
                ci = int(order[rank])
                i = cands[ci][0]
                dx, dy = refined[ci]
                fb_sin, fb_cos = self._cand_fft(i)
                tx = self._corr_scores(fa_sin_conj, fb_sin, dx, dy)
                ty = self._corr_scores(fa_cos_conj, fb_cos, dy, dx)
                dbg_cands.append({
                    "rank": rank, "img": int(i), "name": str(self.names[i]),
                    "d0": (cands[ci][1], cands[ci][2]), "d": (dx, dy),
                    "h": float(h_scores[ci]), "v": float(v[ci]),
                    "em_al": aligned[ci], "V": v_maps[ci],
                    "contrib": contribs.get(ci),
                    "tx": tx.tolist(), "ty": ty.tolist(),
                    "offgrid": hist[i].tolist(),
                })
            vote_best = hist.reshape(self.n_images, -1).max(axis=1)
            vote_order = np.argsort(vote_best)[::-1][:10]
            self.last_debug = {
                "drawing": drawing, "theta": q_theta, "mag": q_mag,
                "cells_inky": sorted(self._cell_votes.keys()),
                "cells_changed": list(self.last_changed),
                "n_q": int(n_q),
                "vote_top": [
                    {"img": int(i), "name": str(self.names[i]),
                     "score": float(vote_best[i]),
                     "offgrid": hist[i].tolist()} for i in vote_order],
                "cands": dbg_cands,
                "n_active": int((v > 0).sum()),
                "h_star": h_star, "alpha": float(alpha),
                "partials": partials, "S": S,
                "timing": {"vote_ms": int((t_vote - t0) * 1000),
                           "align_weight_ms": int((t_align - t_vote) * 1000),
                           "composite_ms": int((_time.time() - t_align) * 1000)},
            }
        return S, info

    @staticmethod
    def render(S, sigma_r=16.0):
        """Display filtering (eq. 12 spirit): reinforce agreeing edges."""
        if S.max() <= 0:
            return np.zeros_like(S)
        g = blur2d(S, sigma_r)
        S2 = S * np.clip(g / max(1e-9, np.percentile(g, 99)), 0, 1)
        S2 = blur2d(S2, 1.25)
        return S2 / max(1e-9, S2.max())
