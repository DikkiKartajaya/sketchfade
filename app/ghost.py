"""B · Construction Ghost: turn a hidden target face into construction layers.

Pipeline (final proposal): target -> face landmarks + confidence check
(reject low confidence) -> guide layers (part masks, block-in, axes, contours,
shading, shadow) -> expert review and frozen guide package (package.py).

The target is a generated face image (generate.py: SD + ControlNet + LCM),
whose layers are placed on its own 478 MediaPipe face landmarks
(generated_layers), or, when no generator is installed, the top aligned
ShadowDraw database face (build_layers: edges only, canonical layout, no
shading or shadow layer).
"""

import os
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 300  # ShadowDraw database frame
PARTS = ["head", "eyes", "nose", "mouth", "neck"]  # head with hair; neck with shoulders
HEAD, EYES, NOSE, MOUTH, NECK = range(5)
LAYERS = ["block-in", "skeleton", "contours", "shading", "shadow"]
FILLED = ("shading", "shadow")  # tone layers: filled regions, judged by their outline

# Canonical face layout in the 300x300 database frame, measured from the mean
# database edge image (the faces are pre-aligned crops).
CX, CY, RX, RY = 150, 160, 78, 108
EYE_Y, NOSE_Y, MOUTH_Y = 128, 185, 222


# Targets whose weakest landmark is below CONF_MIN are rejected. 0.20 rejects
# ~12 % of ShadowDraw database faces; recalibrate on generated targets in the pilot.
CONF_MIN = 0.20


def _row_landmark(target, y, x, half_h, half_w):
    """Row with the most edge pixels near y, and how distinct that peak is.

    Confidence in [0, 1]: peak contrast against the window's median row, scaled
    down when the peak holds only a few edge pixels (nothing there to snap to)."""
    y0, x0 = max(0, y - half_h), max(0, x - half_w)
    rows = target[y0 : y + half_h, x0 : x + half_w].sum(axis=1).astype(np.float32)
    if rows.size == 0 or rows.max() == 0:
        return y, 0.0
    rows = np.convolve(rows, np.ones(3) / 3, mode="same")
    peak = float(rows.max())
    conf = (peak - float(np.median(rows))) / peak * min(1.0, peak / 6.0)
    return y0 + int(np.argmax(rows)), float(np.clip(conf, 0, 1))


def find_landmarks(target, dx, dy):
    """Face landmarks of the target (in the drawing frame) with a confidence each.

    target: (300, 300) bool edges, aligned to the drawing; (dx, dy): ShadowDraw
    displacement, so the canonical layout point p sits at p - d.
    Returns (landmarks dict, confidence dict). The proposal's validation step
    rejects targets whose lowest confidence is below CONF_MIN."""
    cx, cy = CX - dx, CY - dy
    eye, c_eye = _row_landmark(target, EYE_Y - dy, cx, 12, 60)
    nose, c_nose = _row_landmark(target, NOSE_Y - dy, cx, 12, 20)
    mouth, c_mouth = _row_landmark(target, MOUTH_Y - dy, cx, 12, 35)

    # eye centers: densest column in each half of the eye band
    band = target[max(0, eye - 8) : eye + 8].sum(axis=0).astype(np.float32)
    band = np.convolve(band, np.ones(9) / 9, mode="same")
    eyes_x, mass = [], []
    for lo, hi in ((cx - 55, cx - 12), (cx + 12, cx + 55)):
        lo, hi = max(0, lo), min(SIZE, hi)
        seg = band[lo:hi]
        eyes_x.append(lo + int(np.argmax(seg)) if seg.size and seg.max() > 0
                      else (cx - 32 if lo < cx else cx + 32))
        mass.append(float(seg.sum()))
    sym = min(mass) / max(mass) if max(mass) > 0 else 0.0  # both eyes present

    # head: share of the layout ellipse with a target edge within 5 px
    t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    px = np.clip((cx + RX * np.cos(t)).astype(int), 0, SIZE - 1)
    py = np.clip((cy + RY * np.sin(t)).astype(int), 0, SIZE - 1)
    ys, xs = np.nonzero(target)
    if xs.size:
        near = [((xs - x) ** 2 + (ys - y) ** 2).min() <= 25 for x, y in zip(px, py)]
        c_head = float(np.mean(near))
    else:
        c_head = 0.0

    lm = {"cx": cx, "cy": cy, "eye": eye, "eye_l": eyes_x[0], "eye_r": eyes_x[1],
          "nose": nose, "mouth": mouth}
    conf = {"head": round(c_head, 3), "eyes": round(min(c_eye, sym), 3),
            "nose": round(c_nose, 3), "mouth": round(c_mouth, 3)}
    return lm, conf


def _blur(gray, sigma):
    im = Image.fromarray((gray * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sigma))
    return np.asarray(im, np.float32) / 255.0


def _edge_peaks(gray, sigma):
    """Thin edge candidates of a grayscale image (Canny without hysteresis):
    (gradient peaks bool, gradient magnitude)."""
    g = _blur(gray, sigma)
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    # non-maximum suppression along the dominant gradient axis
    m = np.pad(mag, 1)
    peak = np.where(np.abs(gx) >= np.abs(gy),
                    (mag >= m[1:-1, :-2]) & (mag >= m[1:-1, 2:]),
                    (mag >= m[:-2, 1:-1]) & (mag >= m[2:, 1:-1]))
    return peak, mag


def build_layers(target, dx, dy, lm=None):
    """Retrieval targets: layers on the canonical layout (no tone, so the
    shading and shadow layers stay empty).
    target: (300, 300) bool edges, already aligned to the drawing.
    (dx, dy): ShadowDraw displacement of the sketch; canonical point p sits at p - d.
    lm: landmarks from find_landmarks (computed here when not given).

    Returns (5, 300, 300) int8, one label image per layer:
    0 = no guide, k + 1 = guide pixel belonging to part k."""
    if lm is None:
        lm, _ = find_landmarks(target, dx, dy)
    cx, cy, eye, nose, mouth = lm["cx"], lm["cy"], lm["eye"], lm["nose"], lm["mouth"]
    # keep detected eye centers plausible (symmetric spacing around the axis)
    half = int(np.clip((lm["eye_r"] - lm["eye_l"]) / 2, 22, 42))

    # 1 block-in: big simple shapes
    block = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(block)
    d.ellipse([cx - RX, cy - RY, cx + RX, cy + RY], outline=HEAD + 1, width=2)
    for ex in (cx - half, cx + half):
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

    empty = np.zeros((SIZE, SIZE), np.int8)
    return np.stack([np.asarray(block), np.asarray(skel), contours,
                     empty, empty]).astype(np.int8)


# --- generated targets: layers on the target's own face landmarks ---------------

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MESH_MODEL = os.path.join(MODELS, "face_landmarker.task")
MESH_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task")
# hair / face / neck / shoulder regions; it can miss strongly stylized drawings,
# the layers then cover the face only
SEG_MODEL = os.path.join(MODELS, "selfie_multiclass_256x256.tflite")
SEG_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
           "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")
HAIR, BODY_SKIN, FACE_SKIN, CLOTHES = 1, 2, 3, 4  # segmenter categories (0 background)
SHOULDER_K = 1.6  # shoulder points: this many half face widths beside the axis
TOP, CHIN, NOSE_TIP, NOSE_BASE = 10, 152, 1, 2  # MediaPipe face mesh point ids
NOSE_ROOT, ALA_R, ALA_L = 168, 98, 327
MOUTH_R, MOUTH_L, CHEEK_R, CHEEK_L = 61, 291, 234, 454
IRIS_R, IRIS_L = 468, 473
EDGE_FRAC = 0.35  # contours: edges at least as strong as the strongest 35 % in the face
SUPPORT_PX = 3.0  # a landmark outline point is supported by an edge this close
SHADOW_T, SHADING_T = 0.30, 0.55  # tone thresholds, 0 = darkest, 1 = lit skin
MIN_BLOB = 25  # px: smaller tone regions are dropped (edges: 8 px)

_mesher = None
_segmenter = None


def _model(path, url):
    """Model file contents, downloaded once (face mesh 3.7 MB, segmenter 16 MB)."""
    if not os.path.exists(path):
        os.makedirs(MODELS, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    with open(path, "rb") as f:
        return f.read()


def _mp_image(gray):
    import mediapipe as mp
    rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _mesh_points(gray):
    """(478, 2) MediaPipe face landmarks of the target in its own frame, or None."""
    global _mesher
    from mediapipe.tasks.python import BaseOptions, vision

    if _mesher is None:
        _mesher = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=_model(MESH_MODEL, MESH_URL)),
            num_faces=1))
    res = _mesher.detect(_mp_image(gray))
    if not res.face_landmarks:
        return None
    h, w = gray.shape
    return np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], np.float32)


def _segment(gray):
    """Segmenter category per pixel: 0 background, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, 5 other."""
    global _segmenter
    from mediapipe.tasks.python import BaseOptions, vision

    if _segmenter is None:
        _segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_buffer=_model(SEG_MODEL, SEG_URL)),
            output_category_mask=True))
    res = _segmenter.segment(_mp_image(gray))
    return res.category_mask.numpy_view().reshape(gray.shape)


def _segments(*regions):
    """Mesh outline segments of face regions (e.g. "LIPS"), as (i, j) point ids."""
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as C
    return [(c.start, c.end) for r in regions for c in getattr(C, f"FACE_LANDMARKS_{r}")]


def _ids(*regions):
    return sorted({i for s in _segments(*regions) for i in s})


# eyebrows belong to the eyes
REGIONS = {HEAD: ["FACE_OVAL"], EYES: ["LEFT_EYE", "RIGHT_EYE", "LEFT_EYEBROW",
                                       "RIGHT_EYEBROW"],
           NOSE: ["NOSE"], MOUTH: ["LIPS"]}


def _hull(pts):
    import cv2
    return [tuple(p) for p in cv2.convexHull(pts.astype(np.float32))[:, 0]]


def _ellipse(pts, n=64):
    """Closed polyline of the ellipse fitted to pts (follows the face's tilt)."""
    import cv2
    (cx, cy), (w, h), a = cv2.fitEllipse(pts.astype(np.float32))
    t, a = np.linspace(0, 2 * np.pi, n + 1), np.deg2rad(a)
    x, y = w / 2 * np.cos(t), h / 2 * np.sin(t)
    return list(zip(cx + x * np.cos(a) - y * np.sin(a), cy + x * np.sin(a) + y * np.cos(a)))


def _circle(pts, n=64):
    """Closed polyline of the circle fitted to pts (least squares, stable on an arc)."""
    x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
    c = np.linalg.lstsq(np.stack([x, y, np.ones_like(x)], 1), x * x + y * y, rcond=None)[0]
    cx, cy = c[0] / 2, c[1] / 2
    r = np.sqrt(c[2] + cx * cx + cy * cy)
    t = np.linspace(0, 2 * np.pi, n + 1)
    return list(zip(cx + r * np.cos(t), cy + r * np.sin(t)))


def _drop_small(mask, n=MIN_BLOB):
    import cv2
    k, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(k, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= n
    return keep[lab]


def _touching(mask, seed):
    """The connected components of mask that overlap seed (drops stray blobs)."""
    import cv2
    _, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    keep = np.isin(lab, np.unique(lab[seed & mask]))
    return keep & mask


def _grow(mask, px):
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(im.filter(ImageFilter.MaxFilter(2 * px + 1))) > 0


def _outline(mask, eps=0.0):
    """Outer outline of the largest region of mask as (N, 2) points, simplified
    to straight segments eps px from the outline when eps > 0."""
    import cv2
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return np.zeros((0, 2), np.float32)
    c = max(cs, key=cv2.contourArea)
    if eps:
        c = cv2.approxPolyDP(c, eps, True)
    return c[:, 0].astype(np.float32)


def _part_map(pts, region, body):
    """Part index per pixel inside region: hulls of each eye (with its brow), the
    nose and the lips, grown a little; neck on body; head for the rest; -1
    outside region."""
    im = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(im)
    for part, groups in ((NOSE, [["NOSE"]]), (MOUTH, [["LIPS"]]),  # later wins
                         (EYES, [["LEFT_EYE", "LEFT_EYEBROW"],
                                 ["RIGHT_EYE", "RIGHT_EYEBROW"]])):
        for g in groups:
            d.polygon(_hull(pts[_ids(*g)]), fill=part + 1)
    grown = np.asarray(im.filter(ImageFilter.MaxFilter(9))).astype(np.int8) - 1
    rest = np.where(body, NECK, HEAD)
    return np.where(region, np.where(grown >= 0, grown, rest), -1)


def generated_layers(gray):
    """Layers of a generated target: the face parts on its own face landmarks,
    the head with hair and the neck with shoulders from the segmenter.

    gray: (300, 300) float target in [0, 1], aligned to the drawing.
    Returns (layers (5, 300, 300) int8 or None, confidence per face part). A
    part's confidence is the share of its landmark outline that lies on an edge
    of the image (0 for every part when no face is found), so a part the image
    does not clearly show fails the validation step. The neck is not validated:
    a target may be cropped at the chin."""
    pts = _mesh_points(gray)
    if pts is None:
        return None, {PARTS[p]: 0.0 for p in REGIONS}

    oval = pts[_ids("FACE_OVAL")]
    face_im = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(face_im).polygon(_hull(oval), fill=1)
    face = np.asarray(face_im) > 0

    u = pts[IRIS_L] - pts[IRIS_R]  # across the face, along its tilt
    u /= np.linalg.norm(u)
    v = np.array([-u[1], u[0]])  # down the face
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    below_chin = (xx - pts[CHIN][0]) * v[0] + (yy - pts[CHIN][1]) * v[1] > 0

    # figure: face + hair (head) and neck + shoulders (body), attached to the face
    cat = _segment(gray)
    figure = _touching(face | np.isin(cat, (HAIR, FACE_SKIN, BODY_SKIN, CLOTHES)), face)
    head = figure & (face | np.isin(cat, (HAIR, FACE_SKIN)))
    if (figure & ~face).sum() < 0.2 * face.sum():
        # the segmenter missed hair and body (strongly stylized drawing): take what
        # differs from the background (the image border), split at the chin
        border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
        fg = np.abs(_blur(gray, 3.0) - np.median(border)) > 0.15
        fg = ~_grow(~_grow(fg, 3), 3)  # close small gaps between strokes
        figure = _touching(face | fg, face)
        head = figure & (face | ~below_chin)
    body = figure & ~head
    near = _grow(figure, 4)
    part = _part_map(pts, near, _grow(body, 4) & ~head)

    # edges: thin gradient peaks around the figure, as strong as the strongest
    # EDGE_FRAC of those in the face (hair texture does not set the bar)
    peak, mag = _edge_peaks(gray, 1.0)
    bar = np.percentile(mag[peak & _grow(face, 4)], 100 * (1 - EDGE_FRAC))
    edges = _drop_small(peak & near & (mag > bar), 8)

    # confidence: landmark outline points (4 per segment) with an edge nearby
    ey, ex = np.nonzero(edges)
    e = np.stack([ex, ey], 1).astype(np.float32)
    t = np.linspace(0, 1, 4, endpoint=False)[:, None, None]
    conf = {}
    for p, name in enumerate(PARTS):
        if p not in REGIONS:
            continue
        a, b = np.array(_segments(*REGIONS[p])).T
        q = (pts[a] * (1 - t) + pts[b] * t).reshape(-1, 2)
        d2 = ((q[:, None, :] - e[None]) ** 2).sum(-1).min(1) if len(e) else np.inf
        conf[name] = round(float(np.mean(d2 <= SUPPORT_PX ** 2)), 3)

    P = lambda i: tuple(pts[i])  # noqa: E731
    eye_mid = (pts[IRIS_R] + pts[IRIS_L]) / 2

    def inside(mask, p):
        x, y = int(round(p[0])), int(round(p[1]))
        return 0 <= x < SIZE and 0 <= y < SIZE and mask[y, x]

    # top of the skull (with hair): up the axis from the hairline while in the head
    top = pts[TOP].copy()
    while inside(head, top - v):
        top -= v
    # shoulders: first body pixel below the chin, SHOULDER_K half face widths out
    half = np.linalg.norm(pts[CHEEK_L] - pts[CHEEK_R]) / 2
    shoulders = []
    for s in (-1, 1):
        q = pts[CHIN] + s * SHOULDER_K * half * u
        for _ in range(SIZE):
            if inside(body, q):
                shoulders.append(q)
                break
            q = q + v

    # 1 block-in: cranium circle (skull with hair) and face ellipse, eyes and
    # lips, nose wedge, neck and shoulders as a few straight lines
    block = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(block)
    cranium = _outline(head)
    cranium = cranium[(cranium - eye_mid) @ v < 0]  # head outline above the eyes
    if len(cranium) >= 5:
        d.line(_circle(cranium), fill=HEAD + 1, width=2)
    d.line(_ellipse(oval), fill=HEAD + 1, width=2)
    for r in ("LEFT_EYE", "RIGHT_EYE"):
        d.line(_ellipse(pts[_ids(r)]), fill=EYES + 1, width=2)
    d.line([P(NOSE_ROOT), P(ALA_R), P(NOSE_BASE), P(ALA_L), P(NOSE_ROOT)],
           fill=NOSE + 1, width=2)
    d.line(_ellipse(pts[_ids("LIPS")]), fill=MOUTH + 1, width=2)
    body_line, near_head = _outline(body, eps=5), _grow(head, 2)
    for a, b in zip(body_line, np.roll(body_line, -1, axis=0)):
        # skip the image border (cropped shoulders) and the seam under the jaw
        on_border = [min(p.min(), SIZE - 1 - p.max()) < 2 for p in (a, b)]
        if not all(on_border) and not inside(near_head, (a + b) / 2):
            d.line([tuple(a), tuple(b)], fill=NECK + 1, width=2)

    # 2 skeleton: center axis from the top of the skull to the base of the neck,
    # proportion lines across the face and the shoulder line, along its tilt
    skel = Image.new("L", (SIZE, SIZE), 0)
    d = ImageDraw.Draw(skel)

    def across(c, i, j, extra=6):  # line through c, as wide as points i..j (+extra)
        h = np.linalg.norm(pts[j] - pts[i]) / 2 + extra
        return [tuple(c - u * h), tuple(c + u * h)]

    d.line([tuple(top), P(TOP), P(NOSE_TIP), P(CHIN)], fill=HEAD + 1, width=2)
    d.line(across(eye_mid, CHEEK_R, CHEEK_L, 0), fill=EYES + 1, width=2)
    d.line(across(pts[NOSE_BASE], ALA_R, ALA_L), fill=NOSE + 1, width=2)
    d.line(across((pts[MOUTH_R] + pts[MOUTH_L]) / 2, MOUTH_R, MOUTH_L),
           fill=MOUTH + 1, width=2)
    if len(shoulders) == 2:
        base = pts[CHIN] + v * ((shoulders[0] + shoulders[1]) / 2 - pts[CHIN]) @ v
        d.line([P(CHIN), tuple(base)], fill=NECK + 1, width=2)
        d.line([tuple(shoulders[0]), tuple(shoulders[1])], fill=NECK + 1, width=2)

    # 3 contours: the target's own edges on and around the figure, labeled by part
    contours = np.where(edges, part + 1, 0)

    # 4-5 tone: blurred value in the figure, scaled by the face: 0 = darkest,
    # 1 = lit skin (dark hair becomes one shadow mass)
    tone = _blur(gray, 2.5)
    lo, hi = np.percentile(tone[face], [2, 90])
    val = (tone - lo) / max(hi - lo, 1e-3)
    shadow = _drop_small(figure & (val < SHADOW_T))
    shading = _drop_small(figure & (val < SHADING_T) & ~shadow)

    layers = np.stack([np.asarray(block), np.asarray(skel), contours,
                       np.where(shading, part + 1, 0), np.where(shadow, part + 1, 0)])
    return layers.astype(np.int8), conf
