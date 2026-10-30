"""ShadowDraw UI server (stdlib only — no FastAPI/websockets needed).

  python3 server.py [port]     # default 9000

Endpoints:
  GET  /                static/index.html
  POST /api/shadow      {"strokes":[{"pts":[[x,y],...],"erase":bool,"w":px}]}
                        coords in 480x480 canvas space
                        -> {"shadow": base64 PNG (300x300 grayscale),
                            "top": [...], "ms": int}
  POST /api/save        persist the stroke log -> {"file": name}

The engine votes incrementally, so consecutive /api/shadow calls during one
drawing session are fast; strokes are re-rasterized fully each call (cheap).
"""

import base64
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw

from bice import bice_descriptor, jaccard_packed, orientation_maps, pack_descriptors
from engine import CANVAS, PATCH, Q_STRIDE, ShadowEngine

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
SESSIONS = os.path.join(HERE, "sessions")
DB_PATH = os.path.join(HERE, "data", "face_db.npz")
UI_CANVAS = 480
SCALE = CANVAS / UI_CANVAS  # 0.625

engine_lock = threading.Lock()
engine = None


def rasterize(strokes):
    """Vector strokes (480-space) -> (300,300) bool ink raster."""
    im = Image.new("L", (CANVAS, CANVAS), 0)
    dr = ImageDraw.Draw(im)
    for s in strokes:
        pts = [(p[0] * SCALE, p[1] * SCALE) for p in s["pts"]]
        if len(pts) == 1:
            pts = pts * 2
        w = max(1, round(s.get("w", 3) * SCALE))
        fill = 0 if s.get("erase") else 255
        if s.get("erase"):
            w = max(w, 12)
        dr.line(pts, fill=fill, width=w, joint="curve")
    return np.asarray(im) > 128


# ---- image encoding helpers for the pipeline visualization ---------------

_VIRIDIS = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140],
                     [94, 201, 98], [253, 231, 37]], dtype=np.float32)


def b64png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def gray_png(arr, size=None, invert=True):
    """float [0,1] or bool array -> grayscale PNG (dark ink on light)."""
    a = np.clip(np.asarray(arr, dtype=np.float32), 0, 1)
    if invert:
        a = 1.0 - a
    img = Image.fromarray((a * 255).astype(np.uint8))
    if size:
        img = img.resize((size, size), Image.BILINEAR)
    return b64png(img)


def heat_png(arr, size=None):
    """float array -> viridis-style heatmap PNG."""
    a = np.asarray(arr, dtype=np.float32)
    a = (a - a.min()) / max(1e-9, float(a.max() - a.min()))
    x = a * (len(_VIRIDIS) - 1)
    i = np.clip(x.astype(int), 0, len(_VIRIDIS) - 2)
    f = (x - i)[..., None]
    rgb = (_VIRIDIS[i] * (1 - f) + _VIRIDIS[i + 1] * f).astype(np.uint8)
    img = Image.fromarray(rgb)
    if size:
        img = img.resize((size, size), Image.NEAREST if arr.shape[0] < 40
                         else Image.BILINEAR)
    return b64png(img)


def orient_png(theta, mag, size=None):
    """orientation field -> hue = angle, brightness = magnitude."""
    h = (np.mod(theta, np.pi) / np.pi * 255).astype(np.uint8)
    s = np.full_like(h, 230)
    m = np.asarray(mag, np.float32)
    v = (np.clip(m / max(1e-9, np.percentile(m, 99)), 0, 1) ** 0.5 * 255
         ).astype(np.uint8)
    img = Image.merge("HSV", [Image.fromarray(c) for c in (h, s, v)]).convert("RGB")
    if size:
        img = img.resize((size, size), Image.BILINEAR)
    return b64png(img)


def overlay_png(drawing, em_al, size=None):
    """query strokes (blue) over an aligned candidate edge map (sepia)."""
    h, w = drawing.shape
    out = np.full((h, w, 3), 247, dtype=np.uint8)
    out[em_al] = [150, 128, 96]
    out[drawing] = [36, 52, 110]
    img = Image.fromarray(out)
    if size:
        img = img.resize((size, size), Image.NEAREST)
    return b64png(img)


def green_edge_png(em_al):
    """aligned edges as a green RGBA overlay for the drawing canvas."""
    h, w = em_al.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[em_al] = [30, 150, 60, 210]
    return b64png(Image.fromarray(rgba))


def debug_snapshot(dbg):
    """engine.last_debug -> JSON-friendly snapshot for the pipeline drawer."""
    T = 160  # thumbnail size
    cands = [{
        "rank": c["rank"], "name": c["name"], "img": c["img"],
        "d0": list(c["d0"]), "d": list(c["d"]),
        "h": round(c["h"], 1), "v": round(c["v"], 3),
        "thumb": gray_png(c["em_al"], T),
        "vmap": heat_png(c["V"], T),
        "tx": [round(float(x), 1) for x in c["tx"]],
        "ty": [round(float(x), 1) for x in c["ty"]],
        "offgrid": c["offgrid"],
    } for c in dbg["cands"]]
    return {
        "raster": gray_png(dbg["drawing"], T),
        "orient": orient_png(dbg["theta"], dbg["mag"], T),
        "cells_inky": [list(c) for c in dbg["cells_inky"]],
        "cells_changed": [list(c) for c in dbg["cells_changed"]],
        "n_q": dbg["n_q"],
        "vote_top": [{k: v for k, v in r.items()} for r in dbg["vote_top"]],
        "cands": cands,
        "n_active": dbg["n_active"],
        "h_star": round(dbg["h_star"], 1), "alpha": round(dbg["alpha"], 3),
        "partials": [gray_png(np.clip(p * 3, 0, 1), T) for p in dbg["partials"]],
        "S": gray_png(np.clip(dbg["S"] * 3, 0, 1), T),
        "S_filt": gray_png(engine.render(dbg["S"]), T),
        "overlay_edge": green_edge_png(dbg["cands"][0]["em_al"]) if cands else None,
        "timing": dbg["timing"],
    }


LAST = {"drawing": None, "dbg": None}


def shadow_response(strokes, debug=False):
    t0 = time.time()
    drawing = rasterize(strokes)
    with engine_lock:
        S, info = engine.shadow(drawing, collect=debug)
        Sr = engine.render(S)
        dbg = engine.last_debug
    LAST["drawing"] = drawing
    if dbg is not None:
        LAST["dbg"] = dbg
    out = {
        "shadow": gray_png(Sr, invert=False),
        "top": [{"name": r["name"], "v": round(r["v"], 2)} for r in info[:5]],
        "ms": int((time.time() - t0) * 1000),
    }
    if debug and dbg is not None:
        out["dbg"] = debug_snapshot(dbg)
    return out


def inspect_patch(kx, ky, top_n=8):
    """Recompute one query patch's BiCE descriptor + its top DB matches."""
    drawing = LAST["drawing"]
    if drawing is None:
        return {"error": "no query yet"}
    y, x = ky * Q_STRIDE, kx * Q_STRIDE
    with engine_lock:
        theta, mag = orientation_maps(drawing.astype(np.float32))
        d = bice_descriptor(theta[y : y + PATCH, x : x + PATCH],
                            mag[y : y + PATCH, x : x + PATCH])
        # descriptor as 4 orientation channels of 18(perp) x 6(tan)
        bits = d.reshape(4, 18, 6)
        chan_imgs = [gray_png(np.kron(b.astype(float), np.ones((8, 8))),
                              invert=True) for b in bits]
        scores = jaccard_packed(engine.desc, engine.desc_pop,
                                pack_descriptors(d))
        order = np.argsort(scores)[::-1][:top_n]
        matches = []
        for di in order:
            i = int(engine.d_img[di])
            gx, gy = int(engine.d_gx[di]), int(engine.d_gy[di])
            em = engine.edge_map(i)
            patch = em[gy * 30 : gy * 30 + PATCH, gx * 30 : gx * 30 + PATCH]
            matches.append({"name": str(engine.names[i]),
                            "score": round(float(scores[di]), 3),
                            "img": gray_png(patch, 120)})
    return {"kx": kx, "ky": ky,
            "patch": gray_png(drawing[y : y + PATCH, x : x + PATCH], 120),
            "channels": chan_imgs, "matches": matches}


def inspect_cand(rank):
    """Large views for one candidate from the cached debug record."""
    dbg = LAST["dbg"]
    if dbg is None or rank >= len(dbg["cands"]):
        return {"error": "no debug record"}
    c = dbg["cands"][rank]
    return {"rank": rank, "name": c["name"],
            "overlay": overlay_png(dbg["drawing"], c["em_al"], 480),
            "vmap": heat_png(c["V"], 300),
            "contrib": gray_png(np.clip(c["contrib"] * 3, 0, 1), 300)
                       if c["contrib"] is not None else None}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/inspect/"):
            q = parse_qs(urlparse(self.path).query)
            if self.path.startswith("/api/inspect/patch"):
                self._send(200, inspect_patch(int(q["kx"][0]), int(q["ky"][0])))
            elif self.path.startswith("/api/inspect/cand"):
                self._send(200, inspect_cand(int(q["rank"][0])))
            else:
                self._send(404, {"error": "not found"})
            return
        path = "index.html" if self.path in ("/", "") else self.path.lstrip("/")
        path = os.path.normpath(path)
        full = os.path.join(STATIC, path)
        if path.startswith(("..", "/")) or not os.path.isfile(full):
            self._send(404, {"error": "not found"})
            return
        ctype = {"html": "text/html", "js": "text/javascript",
                 "css": "text/css", "png": "image/png"}.get(
                     path.rsplit(".", 1)[-1], "application/octet-stream")
        with open(full, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "bad json"})
            return
        if self.path == "/api/shadow":
            self._send(200, shadow_response(payload.get("strokes", []),
                                            debug=bool(payload.get("debug"))))
        elif self.path == "/api/save":
            os.makedirs(SESSIONS, exist_ok=True)
            name = time.strftime("session_%Y%m%d_%H%M%S.json")
            with open(os.path.join(SESSIONS, name), "w") as f:
                json.dump(payload, f)
            self._send(200, {"file": name})
        else:
            self._send(404, {"error": "not found"})


def main():
    global engine
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    print("loading engine ...")
    engine = ShadowEngine(DB_PATH)
    print(f"serving on http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
