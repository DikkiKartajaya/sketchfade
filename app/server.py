"""SketchFade server: A (Fading Shadow) + B (Construction Ghost) on top of the
ShadowDraw demo engine. Standard library HTTP server + numpy/Pillow.

  python server.py [port]      # default 9100

Two modes, following the final proposal (Lecture 4):

  Prepare (before recruitment): rough sketch + prompt + drawing style ->
    Generate target (hidden; SD + ControlNet + LCM) -> face landmarks +
    confidence check (regenerate when low) -> construction layers -> Save
    package -> expert review (in the UI or python package.py review ...).
    Packages are frozen from then on.

  Layers panel (after Chen et al. 2025, guidance beside the canvas): every
    construction layer is shown as a thumbnail and can be shown on or hidden
    from the canvas background; visible layers stack, each in its own color.
    Moving to a layer shows it. The per-part help level still sets how strongly
    each visible layer is drawn, and probes and tests show none.

  Study (per participant): Start {participant, group, package} -> tasks
    practice   help per part from the participant's level (L3..L0); fixed
               group always L3. Each layer ends with an unassisted probe in
               a blank area (both groups); the probe decides the level.
    baseline / post / retention / transfer
               formal tests: no overlay, correction, Check or feedback;
               the drawing is exported under a blinded code for the raters.

Endpoints (all POST, JSON, points in 480x480 canvas space):
  /api/shadow        {strokes}                    -> {shadow}
  /api/pin           {strokes, prompt, style}     -> state     generate + validate
  /api/regenerate    {}                           -> state     next seed
  /api/package/save  {}                           -> {package}
  /api/packages      {}                           -> {packages}
  /api/package/review {package, approve, reviewer, note} -> {package, status}
  /api/start         {participant, group, package}-> state
  /api/task          {task}                       -> state
  /api/layer         {layer}                      -> state
  /api/visible       {visible}                    -> state     layers on the canvas
  /api/stroke        {pts, w}                     -> state + {pts, part, err}
  /api/check         {}                           -> state     L1 parts shown
  /api/probe/start   {}                           -> state
  /api/probe/submit  {strokes}                    -> state + {results, changes}
  /api/export        {strokes}                    -> {code}
  /api/summary       {}                           -> {layers, best} best probe per layer

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
SHOW_TARGET = True  # development only: show the hidden generated target beside the canvas
# one color per layer, so stacked layers stay apart on the canvas
LAYER_RGB = [(194, 112, 61), (59, 143, 106), (70, 95, 170), (138, 90, 158), (80, 62, 50)]

for d in ("logs", "sessions", "participants", "exports"):
    os.makedirs(os.path.join(HERE, d), exist_ok=True)

lock = threading.Lock()
engine = None
S = {"cands": [], "k": 0, "sketch": None, "prompt": "", "style": generate.DEFAULT_STYLE,
     "layers": None, "gray": None, "conf": None, "valid": False, "source": None,
     "layer": 0, "visible": {0}, "package": None, "task": "practice", "probe": None,
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


def gray_png(gray):
    buf = io.BytesIO()
    Image.fromarray((gray * 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def helped():
    """Help (overlay, correction, Check, feedback) only in practice, not in probes."""
    return S["task"] == "practice" and S["probe"] is None


# --- B: prepare targets --------------------------------------------------------

def try_target(k):
    """One candidate target for seed / retrieval index k: (gray, layers, conf, source).
    Generated targets get layers on their own face landmarks (None: no face)."""
    i, (dx, dy) = S["cands"][0 if GEN else k % len(S["cands"])]
    if GEN:
        log.info("generating target: prompt=%r style=%s seed=%d ...",
                 S["prompt"], S["style"], k)
        t0 = time.time()
        gray = generate.generate(S["sketch"], S["prompt"], k, S["style"])
        log.info("target seed %d generated in %.1fs", k, time.time() - t0)
        return (gray, *ghost.generated_layers(gray), "generated")
    target = shift_image(engine.edge_map(i).astype(np.float32), dx, dy) > 0.5
    lm, conf = ghost.find_landmarks(target, dx, dy)
    return (None, ghost.build_layers(target, dx, dy, lm), conf,
            f"retrieval:{engine.names[i]}")


def build_target():
    """B: generate (or retrieve) a hidden target and validate its landmarks.
    Low confidence -> try the next seed, up to MAX_TRIES; the best attempt is
    kept but marked invalid, so it cannot be saved as a package.
    Returns an error dict when no attempt shows a face at all, else None."""
    best = None
    for attempt in range(MAX_TRIES):
        gray, layers, conf, source = try_target(S["k"])
        ok = layers is not None and min(conf.values()) >= ghost.CONF_MIN
        log.info("target %s: landmark confidence %s -> %s", source,
                 " ".join(f"{p}={c:.2f}" for p, c in conf.items()),
                 "ok" if ok else "LOW" if layers is not None else "NO FACE")
        event("target", source=source, seed=S["k"], prompt=S["prompt"],
              style=S["style"], confidence=conf, valid=ok)
        cand = (min(conf.values()), S["k"], gray, layers, conf, source)
        if layers is not None and (best is None or cand[0] > best[0]):
            best = cand
        if ok:
            break
        if attempt < MAX_TRIES - 1:
            S["k"] += 1
    if best is None:
        return {"error": f"no face found in {MAX_TRIES} generated targets: "
                         "change the sketch or the prompt"}
    _, k, gray, layers, conf, source = best
    if GEN and gray is not None:
        os.makedirs(os.path.join(HERE, "targets"), exist_ok=True)
        Image.fromarray((gray * 255).astype(np.uint8)).save(os.path.join(
            HERE, "targets", time.strftime(f"target_%Y%m%d_%H%M%S_seed{k}.png")))
    S.update(k=k, gray=gray, conf=conf, source=source, package=None, layer=0,
             visible={0}, probe=None,  # a new target ends any running probe
             valid=min(conf.values()) >= ghost.CONF_MIN, layers=layers)
    return None


# --- state -------------------------------------------------------------------

def overlay(levels, check):
    """Visible layers stacked on the canvas (tone fills below lines), each in its
    own color, at the per-part help level. Check also reveals the current layer."""
    rgb = np.zeros((CANVAS, CANVAS, 3), np.float32)  # premultiplied
    alpha = np.zeros((CANVAS, CANVAS, 1), np.float32)
    shown = S["visible"] | {S["layer"]} if check else S["visible"]
    for i in sorted(shown, key=lambda i: ghost.LAYERS[i] not in ghost.FILLED):
        a = fading.ghost(S["layers"], i, levels, check)[..., None]
        rgb = np.float32(LAYER_RGB[i]) * a + rgb * (1 - a)
        alpha = a + alpha * (1 - a)
    return alpha[..., 0], rgb / np.maximum(alpha, 1e-6)


def state(check=False):
    c = S["ctrl"]
    levels = c.levels()
    on = helped()
    g, rgb = (overlay(levels, check and on) if on
              else (np.zeros((CANVAS, CANVAS)), (0, 0, 0)))
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
        "ghost": rgba_png(g, rgb),
        "ref": rgba_png(ref * 0.9, (60, 60, 70)),
        # layers panel: each layer in full, its color, and what is on the canvas
        "thumbs": [rgba_png((L > 0) * 0.9, LAYER_RGB[i]) for i, L in enumerate(S["layers"])],
        "colors": ["#%02x%02x%02x" % col for col in LAYER_RGB],
        "visible": sorted(S["visible"]),
        "target_img": (gray_png(S["gray"]) if SHOW_TARGET and S["gray"] is not None
                       else None),
    }


# --- A: practice, probes, tests -------------------------------------------------

def on_stroke(p):
    pts, w = p["pts"], p.get("w", 3)
    raw = np.array(pts, np.float32) * SCALE
    part, err, nearest = fading.match_stroke(raw, S["layers"][S["layer"]])
    level = S["ctrl"].levels()[part]
    out = raw
    if helped() and level == fading.L3 and S["layer"] in S["visible"]:
        out = fading.correct(raw, nearest)  # only toward a guide that is shown
    corrected = out is not raw
    kind = "probe_stroke" if S["probe"] else "stroke"
    log.info("%s layer=%s part=%s err=%.1fpx n=%d level=L%d%s task=%s%s",
             kind, ghost.LAYERS[S["layer"]], ghost.PARTS[part], err, len(pts),
             level, " corrected" if corrected else "", S["task"],
             "" if helped() else " (no help)")
    # raw strokes are kept apart from corrected ones
    event(kind, part=ghost.PARTS[part], level=level, err_px=round(err, 2),
          helped=helped(), corrected=corrected, w=w, raw=pts,
          visible=[ghost.LAYERS[i] for i in sorted(S["visible"])],
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


def summary():
    """Best probe per layer for this participant and package, read back from the
    session log: most parts with success, ties broken by the lower mean error."""
    c = S["ctrl"]
    if not c.path:
        return {"error": "the summary needs a study session"}
    best = [None] * len(ghost.LAYERS)
    with open(os.path.join(HERE, "sessions", f"{c.participant}.jsonl"),
              encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["event"] != "probe" or rec["target"] != target_label():
                continue
            rs = list(rec["results"].values())
            if not rs:
                continue
            cand = {"success": sum(r["outcome"] == "success" for r in rs),
                    "parts": len(rs),
                    "err": sum(r["err"] for r in rs) / len(rs),
                    "cov": sum(r["cov"] for r in rs) / len(rs),
                    "time": rec["time"], "strokes": rec["strokes"]}
            i = ghost.LAYERS.index(rec["probe_layer"])
            b = best[i]
            if b is None or (cand["success"], -cand["err"]) > (b["success"], -b["err"]):
                best[i] = cand
    return {"layers": ghost.LAYERS, "best": best}


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
    if len(layers) != len(ghost.LAYERS):
        return {"error": f"package {pkg} has {len(layers)} layers, not "
                         f"{len(ghost.LAYERS)} ({', '.join(ghost.LAYERS)}): prepare it again"}
    path = os.path.join(HERE, "participants", f"{pid}.json")
    existed = os.path.exists(path)
    try:
        ctrl = fading.Controller(path, pid, p.get("group", "adaptive"))
    except ValueError as e:
        return {"error": str(e)}
    S.update(ctrl=ctrl, layers=layers, package=pkg, layer=0, visible={0}, probe=None,
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
    if path == "/api/package/review":  # expert review from the UI (same as package.py review)
        reviewer = str(p.get("reviewer", "")).strip()
        if not reviewer:
            return {"error": "the review needs the reviewer's name"}
        try:
            m = package.review(p.get("package"), bool(p.get("approve")), reviewer,
                               str(p.get("note", "")))
        except (OSError, ValueError):
            return {"error": f"package {p.get('package')!r} not found"}
        log.info("package %s %s by %s", m["id"], m["review"]["status"], reviewer)
        event("package_review", package=m["id"], status=m["review"]["status"],
              reviewer=reviewer)
        return {"package": m["id"], "status": m["review"]["status"]}
    if path == "/api/start":
        return start(p)
    if path == "/api/pin":
        S["sketch"] = rasterize(p.get("strokes", []))
        _, info = engine.shadow(S["sketch"])
        if not info:
            return {"error": "draw a rough face first"}
        S["cands"] = [(r["img"], r["d"]) for r in info[:N_TARGETS]]
        S["k"] = 0
        style = p.get("style", generate.DEFAULT_STYLE)
        if style not in generate.STYLES:
            return {"error": f"style must be one of {list(generate.STYLES)}"}
        S["prompt"], S["style"] = p.get("prompt", "").strip(), style
        S["ctrl"] = fading.Controller(None, "preview", "fixed")
        S["task"], S["probe"] = "practice", None
        log.info("pin from %d strokes, prompt=%r style=%s", len(p.get("strokes", [])),
                 S["prompt"], style)
        # the rough sketch as drawn, so a target can be regenerated and compared later
        event("pin", prompt=S["prompt"], style=style,
              strokes=[{"pts": s["pts"], "w": s.get("w", 3)} for s in p.get("strokes", [])])
        return build_target() or state()
    if S["layers"] is None:
        return {"error": "generate a target or start a session first"}
    if path == "/api/regenerate":
        if S["package"]:
            return {"error": "packages are frozen; generate a new target instead"}
        S["k"] += 1
        return build_target() or state()
    if path == "/api/package/save":
        if S["package"] or not S["valid"]:
            return {"error": "only a new target that passed the landmark check"}
        pid = package.save(S["layers"], S["gray"], {
            "prompt": S["prompt"], "style": S["style"], "seed": S["k"],
            "source": S["source"], "confidence": S["conf"],
            "generator": "SD1.5+ControlNet+LCM" if GEN else "ShadowDraw retrieval"})
        log.info("package saved: packages/%s (pending review)", pid)
        event("package_saved", package=pid)
        return {"package": pid}
    if path == "/api/task":
        task = p.get("task")
        if task not in TASKS:
            return {"error": f"task must be one of {TASKS}"}
        S["task"], S["probe"], S["layer"], S["visible"] = task, None, 0, {0}
        log.info("task %s participant=%s", task, S["ctrl"].participant)
        event("task", levels=S["ctrl"].levels())
        return state()
    if path == "/api/layer":
        S["layer"] = int(np.clip(p.get("layer", 0), 0, len(ghost.LAYERS) - 1))
        S["visible"].add(S["layer"])  # moving to a layer shows it
        log.info("layer %d. %s", S["layer"] + 1, ghost.LAYERS[S["layer"]])
        event("layer", levels=S["ctrl"].levels())
        return state()
    if path == "/api/visible":
        S["visible"] = {int(i) for i in p.get("visible", [])
                        if 0 <= int(i) < len(ghost.LAYERS)}
        names = [ghost.LAYERS[i] for i in sorted(S["visible"])]
        log.info("visible layers: %s", ",".join(names) or "none")
        event("visible", visible=names)
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
    if path == "/api/summary":
        return summary()
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
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
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
