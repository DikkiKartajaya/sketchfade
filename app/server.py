"""SketchFade server: A (Fading Shadow) + B (Construction Ghost) on top of the
ShadowDraw demo engine. Standard library HTTP server + numpy/Pillow.

  python server.py [port]      # default 9100

Flow: rough sketch (live ShadowDraw shadow) + prompt -> Generate target
(hidden; SD + ControlNet + LCM, see generate.py) -> draw layer by layer.
Every stroke is scored against the hidden target's current layer; the score
updates the per-part skill, which decides how much help each part gets.

Endpoints (all POST, JSON, points in 480x480 canvas space):
  /api/shadow      {strokes}         -> {shadow}        rough-sketch phase
  /api/pin         {strokes, prompt} -> state           generate the target
  /api/regenerate  {}                -> state           new seed / next face
  /api/layer       {layer, test}    -> state
  /api/stroke      {pts, test}      -> state + {pts, part, err}
  /api/check       {test}           -> state with check-on-request parts shown
  /api/save        {}               -> {file}           stroke log for the study
"""

import base64
import io
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "..", "shadowdraw-demo", "demo")
sys.path.insert(0, DEMO)

from engine import CANVAS, ShadowEngine, shift_image  # noqa: E402

import fading  # noqa: E402
import generate  # noqa: E402
import ghost  # noqa: E402

UI = 480
SCALE = CANVAS / UI
N_TARGETS = 10  # retrieval fallback: how many faces Regenerate cycles through
GEN = generate.available()  # False -> retrieval target (no diffusers installed)

lock = threading.Lock()
engine = None
skill = fading.Skill(os.path.join(HERE, "skill.json"))
S = {"cands": [], "k": 0, "sketch": None, "prompt": "", "layers": None,
     "layer": 0, "log": []}

# run log for tracing back what happened: console + logs/sketchfade.log
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(os.path.join(HERE, "logs", "sketchfade.log"),
                                  encoding="utf-8")])
log = logging.getLogger("sketchfade")
logging.getLogger("httpx").setLevel(logging.WARNING)  # HF download chatter


def fmt_skill():
    return " ".join(f"{p}={s:.2f}" for p, s in skill.s.items())


def rasterize(strokes):
    im = Image.new("L", (CANVAS, CANVAS), 0)
    dr = ImageDraw.Draw(im)
    for s in strokes:
        pts = [(x * SCALE, y * SCALE) for x, y in s["pts"]]
        dr.line(pts * 2 if len(pts) == 1 else pts, fill=255, width=2, joint="curve")
    return np.asarray(im) > 128


def rgba_png(alpha, rgb):
    img = np.zeros(alpha.shape + (4,), np.uint8)
    img[..., :3] = rgb
    img[..., 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_target():
    """B: the hidden target. Generated from the sketch + prompt (seed k); the
    ShadowDraw match of the sketch only gives the face layout (d). Without a
    generator, the k-th retrieved database face is used instead."""
    i, (dx, dy) = S["cands"][0 if GEN else S["k"]]
    if GEN:
        log.info("generating target: prompt=%r seed=%d ...", S["prompt"], S["k"])
        t0 = time.time()
        gray = generate.generate(S["sketch"], S["prompt"], seed=S["k"])
        target = ghost.image_edges(gray)
        os.makedirs(os.path.join(HERE, "targets"), exist_ok=True)
        name = time.strftime(f"target_%Y%m%d_%H%M%S_seed{S['k']}.png")
        Image.fromarray((gray * 255).astype(np.uint8)).save(
            os.path.join(HERE, "targets", name))
        log.info("target seed %d: %.1fs, saved targets/%s, d=(%d, %d), %d edge px",
                 S["k"], time.time() - t0, name, dx, dy, target.sum())
    else:
        gray = None
        target = shift_image(engine.edge_map(i).astype(np.float32), dx, dy) > 0.5
        log.info("target %d/%d (retrieval): db image %s, d=(%d, %d), %d edge px",
                 S["k"] + 1, len(S["cands"]), engine.names[i], dx, dy, target.sum())
    S["layers"] = ghost.build_layers(target, dx, dy, gray)
    S["layer"] = 0


def state(test=False, check=False):
    levels = skill.levels()
    g = (np.zeros((CANVAS, CANVAS)) if test
         else fading.ghost(S["layers"], S["layer"], levels, check))
    return {
        "layer": S["layer"], "layers": ghost.LAYERS,
        "target": f"seed {S['k']}" if GEN else f"{S['k'] + 1}/{len(S['cands'])}",
        "skill": [{"part": p, "s": round(skill.s[p], 2),
                   "level": fading.LEVEL_NAMES[levels[j]]}
                  for j, p in enumerate(ghost.PARTS)],
        "ghost": rgba_png(g * 0.7, (70, 95, 170)),
    }


def on_stroke(p, test):
    pts = p["pts"]
    w = p.get("w", 3)
    strength = float(np.clip(p.get("help", 1.0), 0, 1))  # AI help slider
    raw = np.array(pts, np.float32) * SCALE
    part, err, nearest = fading.match_stroke(raw, S["layers"][S["layer"]])
    before = skill.levels()
    level = before[part]
    out = raw
    if not test:
        if level == fading.SHADOW and strength > 0:
            out = fading.correct(raw, nearest, fading.PULL * strength)
        skill.update(part, err)  # measured on the raw stroke, not the corrected one
    log.info("stroke layer=%s part=%s err=%.1fpx n=%d w=%s aihelp=%.0f%% "
             "help='%s'%s%s | %s",
             ghost.LAYERS[S["layer"]], ghost.PARTS[part], err, len(pts), w,
             strength * 100, fading.LEVEL_NAMES[level],
             " corrected" if out is not raw else "", " TEST" if test else "",
             fmt_skill())
    for p, (a, b) in enumerate(zip(before, skill.levels())):
        if a != b:
            log.info("help  %s: '%s' -> '%s'", ghost.PARTS[p],
                     fading.LEVEL_NAMES[a], fading.LEVEL_NAMES[b])
    S["log"].append({"t": time.time(), "target": S["k"], "layer": S["layer"],
                     "part": ghost.PARTS[part], "err": round(err, 2),
                     "level": level, "test": test, "w": w, "aihelp": strength,
                     "pts": pts})
    return {**state(test), "pts": (out / SCALE).round(1).tolist(),
            "part": ghost.PARTS[part], "err": round(err, 1)}


def handle(path, p):
    test = bool(p.get("test"))
    if path == "/api/shadow":
        S_raw, info = engine.shadow(rasterize(p.get("strokes", [])))
        log.info("shadow strokes=%d top=%s", len(p.get("strokes", [])),
                 ", ".join(f"{r['name']}({r['v']:.2f})" for r in info[:3]) or "-")
        return {"shadow": rgba_png(engine.render(S_raw) * 0.8, (90, 80, 70))}
    if path == "/api/pin":
        S["sketch"] = rasterize(p.get("strokes", []))
        _, info = engine.shadow(S["sketch"])
        if not info:
            return {"error": "draw a rough face first"}
        S["cands"] = [(r["img"], r["d"]) for r in info[:N_TARGETS]]
        S["k"] = 0
        S["prompt"] = p.get("prompt", "").strip()
        log.info("pin from %d strokes | %s", len(p.get("strokes", [])), fmt_skill())
        build_target()
        return state(test)
    if S["layers"] is None:
        return {"error": "generate a target first"}
    if path == "/api/regenerate":
        S["k"] = S["k"] + 1 if GEN else (S["k"] + 1) % len(S["cands"])
        build_target()
        return state(test)
    if path == "/api/layer":
        S["layer"] = int(np.clip(p.get("layer", 0), 0, len(ghost.LAYERS) - 1))
        log.info("layer %d. %s%s", S["layer"] + 1, ghost.LAYERS[S["layer"]],
                 " TEST" if test else "")
        return state(test)
    if path == "/api/stroke":
        return on_stroke(p, test)
    if path == "/api/check":
        shown = [ghost.PARTS[j] for j, l in enumerate(skill.levels())
                 if l == fading.CHECK]
        log.info("check layer=%s reveals=%s", ghost.LAYERS[S["layer"]],
                 ",".join(shown) or "nothing")
        return state(test, check=True)
    if path == "/api/save":
        os.makedirs(os.path.join(HERE, "sessions"), exist_ok=True)
        name = time.strftime("session_%Y%m%d_%H%M%S.json")
        with open(os.path.join(HERE, "sessions", name), "w") as f:
            json.dump({"skill": skill.s, "log": S["log"]}, f)
        log.info("saved sessions/%s (%d strokes)", name, len(S["log"]))
        return {"file": name}
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            log.warning("GET %s 404", self.path)
            return self._send(404, {"error": "not found"})
        log.info("GET %s (page loaded)", self.path)
        with open(os.path.join(HERE, "static", "index.html"), "rb") as f:
            self._send(200, f.read(), "text/html; charset=utf-8")

    def do_POST(self):
        t0 = time.time()
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
            with lock:
                out = handle(self.path, payload)
            code = 404 if out is None else 200
        except Exception:
            log.exception("POST %s failed", self.path)  # full traceback in the log
            out, code = {"error": "server error, see logs/sketchfade.log"}, 500
        out = out or {"error": "not found"}
        ms = (time.time() - t0) * 1000
        if "error" in out:
            log.warning("POST %s %d %.0fms error=%s", self.path, code, ms, out["error"])
        else:
            log.info("POST %s %d %.0fms", self.path, code, ms)
        self._send(code, out)


def main():
    global engine
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9100
    log.info("=== SketchFade start, port %d ===", port)
    log.info("skill loaded: %s", fmt_skill())
    t0 = time.time()
    engine = ShadowEngine(os.path.join(DEMO, "data", "face_db.npz"))
    log.info("ShadowDraw engine ready: %d images (%.1fs)", engine.n_images,
             time.time() - t0)
    log.info("target source: %s", f"SD + ControlNet + LCM on {generate.device_name()}"
             if GEN else "ShadowDraw retrieval (pip install diffusers peft for generation)")
    log.info("serving on http://127.0.0.1:%d", port)
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        log.info("=== SketchFade stop ===")


if __name__ == "__main__":
    main()
