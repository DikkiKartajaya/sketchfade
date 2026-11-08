"""SketchFade server: A (Fading Shadow) + B (Construction Ghost) on top of the
ShadowDraw demo engine. Standard library HTTP server + numpy/Pillow.

  python server.py [port]      # default 9100

Two modes, following the final proposal (Lecture 4):

  Prepare (before recruitment): rough sketch + prompt -> Generate target
    (hidden; SD + ControlNet + LCM) -> landmark confidence check (regenerate
    when low) -> construction layers -> Save package -> expert review
    (python package.py review ...). Packages are frozen from then on.

  Study (per participant): Start {participant, group, package} -> tasks
    practice   help per part from the participant's level (L3..L0); fixed
               group always L3. Each layer ends with an unassisted probe in
               a blank area (both groups); the probe decides the level.
    baseline / post / retention / transfer
               formal tests: no overlay, correction, Check or feedback;
               the drawing is exported under a blinded code for the raters.

Endpoints (all POST, JSON, points in 480x480 canvas space):
  /api/shadow        {strokes}                    -> {shadow}
  /api/pin           {strokes, prompt}            -> state     generate + validate
  /api/regenerate    {}                           -> state     next seed
  /api/package/save  {}                           -> {package}
  /api/packages      {}                           -> {packages}
  /api/start         {participant, group, package}-> state
  /api/task          {task}                       -> state
  /api/layer         {layer}                      -> state
  /api/stroke        {pts, w}                     -> state + {pts, part, err}
  /api/check         {}                           -> state     L1 parts shown
  /api/probe/start   {}                           -> state
  /api/probe/submit  {strokes}                    -> state + {results, changes}
  /api/export        {strokes}                    -> {code}

Every event goes to sessions/<participant>.jsonl (timestamped, with participant,
group, task, target, layer, part and level) and to logs/sketchfade.log.
"""

import base64
import csv
import io
import json
import logging
import os
import re
import secrets
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
import package  # noqa: E402

UI = 480
SCALE = CANVAS / UI
N_TARGETS = 10  # retrieval fallback: how many faces Regenerate cycles through
MAX_TRIES = 3  # seeds tried per Generate / Regenerate before giving up
GEN = generate.available()  # False -> retrieval target (no diffusers installed)
TASKS = ["practice", "baseline", "post", "retention", "transfer"]
TEST_REF_LAYER = ghost.LAYERS.index("contours")  # side reference in formal tests

for d in ("logs", "sessions", "participants", "exports"):
    os.makedirs(os.path.join(HERE, d), exist_ok=True)

lock = threading.Lock()
engine = None
S = {"cands": [], "k": 0, "sketch": None, "prompt": "", "layers": None,
     "gray": None, "conf": None, "valid": False, "source": None, "layer": 0,
     "package": None, "task": "practice", "probe": None,
     "ctrl": fading.Controller(None, "preview", "fixed")}

# run log for tracing back what happened: console + logs/sketchfade.log
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(os.path.join(HERE, "logs", "sketchfade.log"),
                                  encoding="utf-8")])
log = logging.getLogger("sketchfade")
logging.getLogger("httpx").setLevel(logging.WARNING)  # HF download chatter


# --- helpers -----------------------------------------------------------------

def fmt_levels():
    return " ".join(f"{p}=L{l}" for p, l in S["ctrl"].level.items())


def target_label():
    if S["package"]:
        return S["package"]
    if S["layers"] is None:
        return None
    return f"seed {S['k']}" if GEN else f"db {S['k'] + 1}/{len(S['cands'])}"


def event(kind, **fields):
    """Study logger: one timestamped JSON line per event."""
    c = S["ctrl"]
    rec = {"t": round(time.time(), 3), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "event": kind, "participant": c.participant, "group": c.group,
           "task": S["task"], "target": target_label(),
           "layer": ghost.LAYERS[S["layer"]] if S["layers"] is not None else None,
           **fields}
    name = c.participant if c.path else "_prepare"
    with open(os.path.join(HERE, "sessions", f"{name}.jsonl"), "a",
              encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def rasterize(strokes, size=CANVAS, scale=SCALE, width=2):
    im = Image.new("L", (size, size), 0)
    dr = ImageDraw.Draw(im)
    for s in strokes:
        pts = [(x * scale, y * scale) for x, y in s["pts"]]
        w = width if width else max(1, int(s.get("w", 3)))
        dr.line(pts * 2 if len(pts) == 1 else pts, fill=255, width=w, joint="curve")
    return np.asarray(im) > 128


def rgba_png(alpha, rgb):
    img = np.zeros(alpha.shape + (4,), np.uint8)
    img[..., :3] = rgb
    img[..., 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def helped():
    """Help (overlay, correction, Check, feedback) only in practice, not in probes."""
    return S["task"] == "practice" and S["probe"] is None


# --- B: prepare targets --------------------------------------------------------

def try_target(k):
    """One candidate target for seed / retrieval index k: (gray, edges, d, source)."""
    i, (dx, dy) = S["cands"][0 if GEN else k % len(S["cands"])]
    if GEN:
        log.info("generating target: prompt=%r seed=%d ...", S["prompt"], k)
        t0 = time.time()
        gray = generate.generate(S["sketch"], S["prompt"], seed=k)
        log.info("target seed %d generated in %.1fs", k, time.time() - t0)
        return gray, ghost.image_edges(gray), (dx, dy), "generated"
    target = shift_image(engine.edge_map(i).astype(np.float32), dx, dy) > 0.5
    return None, target, (dx, dy), f"retrieval:{engine.names[i]}"


def build_target():
    """B: generate (or retrieve) a hidden target and validate its landmarks.
    Low confidence -> try the next seed, up to MAX_TRIES; the best attempt is
    kept but marked invalid, so it cannot be saved as a package."""
    best = None
    for attempt in range(MAX_TRIES):
        gray, target, (dx, dy), source = try_target(S["k"])
        lm, conf = ghost.find_landmarks(target, dx, dy)
        ok = min(conf.values()) >= ghost.CONF_MIN
        log.info("target %s: landmark confidence %s -> %s", source,
                 " ".join(f"{p}={c:.2f}" for p, c in conf.items()),
                 "ok" if ok else "LOW")
        event("target", source=source, seed=S["k"], prompt=S["prompt"],
              confidence=conf, valid=ok)
        cand = (min(conf.values()), S["k"], gray, target, dx, dy, lm, conf, source)
        if best is None or cand[0] > best[0]:
            best = cand
        if ok:
            break
        if attempt < MAX_TRIES - 1:
            S["k"] += 1
    _, k, gray, target, dx, dy, lm, conf, source = best
    if GEN and gray is not None:
        os.makedirs(os.path.join(HERE, "targets"), exist_ok=True)
        Image.fromarray((gray * 255).astype(np.uint8)).save(os.path.join(
            HERE, "targets", time.strftime(f"target_%Y%m%d_%H%M%S_seed{k}.png")))
    S.update(k=k, gray=gray, conf=conf, source=source, package=None, layer=0,
             probe=None,  # a new target ends any running probe
             valid=min(conf.values()) >= ghost.CONF_MIN,
             layers=ghost.build_layers(target, dx, dy, gray, lm))


# --- state -------------------------------------------------------------------

def state(check=False):
    c = S["ctrl"]
    levels = c.levels()
    on = helped()
    g = (fading.ghost(S["layers"], S["layer"], levels, check and on) if on
         else np.zeros((CANVAS, CANVAS)))
    ref_layer = (S["probe"]["layer"] if S["probe"] else
                 S["layer"] if S["task"] == "practice" else TEST_REF_LAYER)
    ref = fading.guide_mask(S["layers"][ref_layer]) > 0
    return {
        "layer": S["layer"], "layers": ghost.LAYERS, "target": target_label(),
        "package": S["package"], "participant": c.participant if c.path else None,
        "group": c.group, "task": S["task"], "helped": on,
        "probe": S["probe"]["layer"] if S["probe"] else None,
        "confidence": S["conf"], "valid": S["valid"], "conf_min": ghost.CONF_MIN,
        "levels": [{"part": p, "level": levels[j], "name": fading.LEVEL_NAMES[levels[j]],
                    "help": fading.LEVEL_HELP[levels[j]], "succ": c.succ[p],
                    "fail": c.fail[p]} for j, p in enumerate(ghost.PARTS)],
        "ghost": rgba_png(g, (70, 95, 170)),
        "ref": rgba_png(ref * 0.9, (60, 60, 70)),
    }


# --- A: practice, probes, tests -------------------------------------------------

def on_stroke(p):
    pts, w = p["pts"], p.get("w", 3)
    raw = np.array(pts, np.float32) * SCALE
    part, err, nearest = fading.match_stroke(raw, S["layers"][S["layer"]])
    level = S["ctrl"].levels()[part]
    out = raw
    if helped() and level == fading.L3:
        out = fading.correct(raw, nearest)
    corrected = out is not raw
    kind = "probe_stroke" if S["probe"] else "stroke"
    log.info("%s layer=%s part=%s err=%.1fpx n=%d level=L%d%s task=%s%s",
             kind, ghost.LAYERS[S["layer"]], ghost.PARTS[part], err, len(pts),
             level, " corrected" if corrected else "", S["task"],
             "" if helped() else " (no help)")
    # raw strokes are kept apart from corrected ones
    event(kind, part=ghost.PARTS[part], level=level, err_px=round(err, 2),
          helped=helped(), corrected=corrected, w=w, raw=pts,
          shown=(out / SCALE).round(1).tolist() if corrected else None)
    res = {**state(), "pts": (out / SCALE).round(1).tolist()}
    if helped():  # formal tests and probes give no feedback
        res.update(part=ghost.PARTS[part], err=round(err, 1))
    return res


def probe_submit(p):
    layer = S["probe"]["layer"]
    strokes = p.get("strokes", [])
    drawn = rasterize(strokes)
    results = fading.score_probe(drawn, S["layers"][layer])
    before = S["ctrl"].levels()
    changes = S["ctrl"].update(results)
    for part, a, b in changes:
        log.info("level %s: L%d -> L%d", part, a, b)
    log.info("probe layer=%s %s | %s", ghost.LAYERS[layer],
             " ".join(f"{k}:{r['outcome'][0]}(e={r['err']:.3f},c={r['cov']:.2f})"
                      for k, r in results.items()), fmt_levels())
    event("probe", probe_layer=ghost.LAYERS[layer], results=results,
          levels_before=before, levels_after=S["ctrl"].levels(),
          changes=[{"part": a, "from": b, "to": c} for a, b, c in changes],
          strokes=[s["pts"] for s in strokes])
    S["probe"] = None
    return {**state(), "results": results,
            "changes": [{"part": a, "from": b, "to": c} for a, b, c in changes]}


def export(p):
    """Formal test drawing -> exports/<code>.png; the key stays in exports/key.csv."""
    code = secrets.token_hex(4)
    img = ~rasterize(p.get("strokes", []), UI, 1.0, width=None)
    Image.fromarray(img.astype(np.uint8) * 255).save(
        os.path.join(HERE, "exports", f"{code}.png"))
    key = os.path.join(HERE, "exports", "key.csv")
    new = not os.path.exists(key)
    with open(key, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["code", "participant", "group", "task", "target", "time"])
        wr.writerow([code, S["ctrl"].participant, S["ctrl"].group, S["task"],
                     target_label(), time.strftime("%Y-%m-%d %H:%M:%S")])
    event("export", code=code, n_strokes=len(p.get("strokes", [])))
    log.info("export %s.png task=%s participant=%s", code, S["task"],
             S["ctrl"].participant)
    return {"code": code}


def start(p):
    pid = str(p.get("participant", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", pid):
        return {"error": "participant id: letters, digits, - or _"}
    pkg = p.get("package")
    if not pkg:
        return {"error": "no approved package yet: Save package under Prepare "
                         "targets, then run: python package.py review <id> --approve "
                         "--reviewer NAME, and reload the page"}
    try:
        layers, meta = package.load(pkg)
    except (OSError, ValueError):
        return {"error": f"package {pkg!r} not found"}
    if meta["review"]["status"] != "approved":
        return {"error": f"package {pkg} is not approved (python package.py review)"}
    path = os.path.join(HERE, "participants", f"{pid}.json")
    existed = os.path.exists(path)
    try:
        ctrl = fading.Controller(path, pid, p.get("group", "adaptive"))
    except ValueError as e:
        return {"error": str(e)}
    S.update(ctrl=ctrl, layers=layers, package=pkg, layer=0, probe=None,
             task="practice", gray=None, conf=meta.get("confidence"), valid=True)
    log.info("session %s participant=%s group=%s package=%s | %s",
             "resumed" if existed else "started", pid, ctrl.group, pkg, fmt_levels())
    event("session_start", resumed=existed, levels=ctrl.levels())
    return state()


def handle(path, p):
    if path == "/api/shadow":
        S_raw, info = engine.shadow(rasterize(p.get("strokes", [])))
        return {"shadow": rgba_png(engine.render(S_raw) * 0.8, (90, 80, 70))}
    if path == "/api/packages":
        return {"packages": package.list_all()}
    if path == "/api/start":
        return start(p)
    if path == "/api/pin":
        S["sketch"] = rasterize(p.get("strokes", []))
        _, info = engine.shadow(S["sketch"])
        if not info:
            return {"error": "draw a rough face first"}
        S["cands"] = [(r["img"], r["d"]) for r in info[:N_TARGETS]]
        S["k"] = 0
        S["prompt"] = p.get("prompt", "").strip()
        S["ctrl"] = fading.Controller(None, "preview", "fixed")
        S["task"], S["probe"] = "practice", None
        log.info("pin from %d strokes, prompt=%r", len(p.get("strokes", [])), S["prompt"])
        build_target()
        return state()
    if S["layers"] is None:
        return {"error": "generate a target or start a session first"}
    if path == "/api/regenerate":
        if S["package"]:
            return {"error": "packages are frozen; generate a new target instead"}
        S["k"] += 1
        build_target()
        return state()
    if path == "/api/package/save":
        if S["package"] or not S["valid"]:
            return {"error": "only a new target that passed the landmark check"}
        pid = package.save(S["layers"], S["gray"], {
            "prompt": S["prompt"], "seed": S["k"], "source": S["source"],
            "confidence": S["conf"], "generator": "SD1.5+ControlNet+LCM" if GEN
            else "ShadowDraw retrieval"})
        log.info("package saved: packages/%s (pending review)", pid)
        event("package_saved", package=pid)
        return {"package": pid}
    if path == "/api/task":
        task = p.get("task")
        if task not in TASKS:
            return {"error": f"task must be one of {TASKS}"}
        S["task"], S["probe"], S["layer"] = task, None, 0
        log.info("task %s participant=%s", task, S["ctrl"].participant)
        event("task", levels=S["ctrl"].levels())
        return state()
    if path == "/api/layer":
        S["layer"] = int(np.clip(p.get("layer", 0), 0, len(ghost.LAYERS) - 1))
        log.info("layer %d. %s", S["layer"] + 1, ghost.LAYERS[S["layer"]])
        event("layer", levels=S["ctrl"].levels())
        return state()
    if path == "/api/stroke":
        return on_stroke(p)
    if path == "/api/check":
        if not helped():
            return {"error": "Check is off in tests and probes"}
        shown = [ghost.PARTS[j] for j, l in enumerate(S["ctrl"].levels())
                 if l == fading.L1]
        log.info("check layer=%s reveals=%s", ghost.LAYERS[S["layer"]],
                 ",".join(shown) or "nothing")
        event("check", reveals=shown)
        return state(check=True)
    if path == "/api/probe/start":
        if S["task"] != "practice":
            return {"error": "probes run during practice"}
        S["probe"] = {"layer": S["layer"], "t0": time.time()}
        log.info("probe start layer=%s", ghost.LAYERS[S["layer"]])
        event("probe_start", levels=S["ctrl"].levels())
        return state()
    if path == "/api/probe/submit":
        if S["probe"] is None:
            return {"error": "no probe running"}
        return probe_submit(p)
    if path == "/api/export":
        return export(p)
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
        elif self.path != "/api/shadow":
            log.info("POST %s %d %.0fms", self.path, code, ms)
        self._send(code, out)


def main():
    global engine
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9100
    log.info("=== SketchFade start, port %d ===", port)
    t0 = time.time()
    engine = ShadowEngine(os.path.join(DEMO, "data", "face_db.npz"))
    log.info("ShadowDraw engine ready: %d images (%.1fs)", engine.n_images,
             time.time() - t0)
    log.info("target source: %s", f"SD + ControlNet + LCM on {generate.device_name()}"
             if GEN else "ShadowDraw retrieval (pip install diffusers peft for generation)")
    log.info("packages: %d (%d approved)", len(package.list_all()),
             sum(m["review"]["status"] == "approved" for m in package.list_all()))
    log.info("serving on http://127.0.0.1:%d", port)
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        log.info("=== SketchFade stop ===")


if __name__ == "__main__":
    main()
