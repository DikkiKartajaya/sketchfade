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
    each visible layer is drawn. Tests show none; a probe starts with none, and the
    drawer may switch on guides of other steps (never the probed one), logged per probe.

  Study (per participant): Start {participant, group, package} -> tasks
    practice   help per part from the participant's level (L3..L0); fixed
               group always L3. Each line step ends with a probe: the step
               redrawn without its guide (both groups; the drawer's own earlier
               steps and, if switched on, other steps' guides as context); the
               probe decides the level.
    baseline / post / retention / transfer
               formal tests: no overlay, correction, Check or feedback;
               the drawing is saved as exports/<group>/final/
               <participant>_<group>_<task>_<key>.png (see "saved drawings").

Endpoints (all POST, JSON, points in 480x480 canvas space):
  /api/shadow        {strokes}                    -> {shadow}
  /api/pin           {strokes, prompt, style}     -> state     generate + validate
  /api/regenerate    {}                           -> state     next N_CHOICES seeds
  /api/choose        {index}                      -> state     pick a candidate target
  /api/package/save  {}                           -> {package}
  /api/packages      {}                           -> {packages}
  /api/package/review {package, approve, reviewer, note} -> {package, status}
  /api/start         {participant, group, package}-> state
  /api/task          {task}                       -> state
  /api/layer         {layer}                      -> state
  /api/visible       {visible}                    -> state     layers on the canvas; during a
                                                               probe: other steps' guides (off at start)
  /api/stroke        {pts, w}                     -> state + {pts, part, err}
  /api/check         {}                           -> state     L1 parts shown
  /api/probe/start   {}                           -> state
  /api/probe/submit  {strokes, own_shown}         -> state + {results, changes}
                     (logs the guides and own earlier steps shown during the probe)
  /api/export        {strokes}                    -> {code, file}   formal test drawing
  /api/practice/step {strokes}                    -> {pass, files}  practice step image(s)
  /api/summary       {}                           -> {layers, best} best probe per layer
  /api/state         {}                           -> state     (no change)
  /api/erase         {kind: undo|clear|eraser, strokes} -> {ok} log what was removed on the canvas
  /api/ping          {}                           -> {ok}      keep-alive (connect.bat), not logged

Pages: / is the researcher app; /participant is the simplified participant view
(static/participant.html) for the tablet, on the same session.

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
# candidates per Generate / Regenerate (seeds k .. k+3), shown side by side so the
# user picks the target, like ShadowDraw's candidates and Chen et al. 2025's four
# guidance sketches; the generated image itself stays hidden
N_CHOICES = 4
GEN = generate.available()  # False -> retrieval target (no diffusers installed)
TASKS = ["practice", "baseline", "post", "retention", "transfer"]
TEST_REF_LAYER = ghost.LAYERS.index("contours")  # side reference in formal tests
# the final practice image: construction lines (block-in, skeleton) erased, as in a finished drawing
FINAL_LAYERS = [name for name in ghost.LAYERS if name not in ("block-in", "skeleton")]
SHOW_TARGET = True  # development only: show the hidden generated target beside the canvas
# one color per layer, so stacked layers stay apart on the canvas
LAYER_RGB = [(194, 112, 61), (59, 143, 106), (70, 95, 170), (138, 90, 158), (80, 62, 50)]
# Remote runs (serve_gpu.sh + connect.bat): the laptop pings /api/ping through the
# SSH tunnel; with no request for this many seconds the server exits, so the GPU is
# released even when the connection dropped without closing the SSH session.
# 0 (default, local runs) = never.
IDLE_EXIT = float(os.environ.get("SKETCHFADE_IDLE_EXIT") or 0)
last_seen = time.time()

for d in ("logs", "sessions", "participants", "exports"):
    os.makedirs(os.path.join(HERE, d), exist_ok=True)

lock = threading.Lock()
engine = None
S = {"cands": [], "k": 0, "seed": 0, "choices": [], "choice": None, "sketch": None, "prompt": "", "style": generate.DEFAULT_STYLE,
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
    return (f"seed {S['seed']}" if GEN
            else f"db {S['seed'] % len(S['cands']) + 1}/{len(S['cands'])}")


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
    """B: generate (or retrieve) N_CHOICES hidden candidate targets (seeds k ..
    k+N_CHOICES-1) and validate the landmarks of each. Candidates without a face
    are dropped; low-confidence ones stay listed but cannot be saved. The best
    candidate (valid first, then highest weakest-landmark confidence) is selected;
    the user can pick another (/api/choose). The next batch starts after these seeds.
    Returns an error dict when no candidate shows a face at all, else None."""
    choices = []
    for seed in range(S["k"], S["k"] + N_CHOICES):
        gray, layers, conf, source = try_target(seed)
        ok = layers is not None and min(conf.values()) >= ghost.CONF_MIN
        log.info("candidate %s seed %d: landmark confidence %s -> %s", source, seed,
                 " ".join(f"{p}={c:.2f}" for p, c in conf.items()),
                 "ok" if ok else "LOW" if layers is not None else "NO FACE")
        event("target", source=source, seed=seed, prompt=S["prompt"],
              style=S["style"], confidence=conf, valid=ok)
        if layers is None:
            continue
        if GEN and gray is not None:
            os.makedirs(os.path.join(HERE, "targets"), exist_ok=True)
            Image.fromarray((gray * 255).astype(np.uint8)).save(os.path.join(
                HERE, "targets", time.strftime(f"target_%Y%m%d_%H%M%S_seed{seed}.png")))
        choices.append({"seed": seed, "gray": gray, "layers": layers, "conf": conf,
                        "source": source, "valid": ok})
    S["k"] += N_CHOICES
    if not choices:
        return {"error": f"no face found in {N_CHOICES} generated targets: "
                         "change the sketch or the prompt"}
    S["choices"] = choices
    best = max(range(len(choices)),
               key=lambda i: (choices[i]["valid"], min(choices[i]["conf"].values())))
    choose(best, auto=True)
    return None


def choose(i, auto=False):
    """Make candidate i the current (hidden) target."""
    c = S["choices"][i]
    S.update(choice=i, seed=c["seed"], gray=c["gray"], conf=c["conf"],
             source=c["source"], valid=c["valid"], layers=c["layers"], package=None,
             layer=0, visible={0}, probe=None)  # a new target ends any running probe
    log.info("candidate %d/%d chosen%s: seed %d", i + 1, len(S["choices"]),
             " (best, automatic)" if auto else " by the user", c["seed"])
    event("choose", choice=i, seed=c["seed"], auto=auto, valid=c["valid"],
          confidence=c["conf"])


def choice_thumb(layers):
    """Candidate thumbnail: the contours over a faint block-in. A line drawing, so
    the user can judge the face without seeing the generated image."""
    a_block = (layers[0] > 0) * 0.35
    a_cont = (layers[2] > 0) * 0.95
    alpha = np.maximum(a_block, a_cont)
    rgb = np.where((layers[2] > 0)[..., None], LAYER_RGB[2], LAYER_RGB[0])
    img = np.zeros(alpha.shape + (4,), np.uint8)
    img[..., :3] = rgb
    img[..., 3] = (alpha * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# --- state -------------------------------------------------------------------

def overlay(levels, check, shown=None):
    """Visible layers stacked on the canvas (tone fills below lines), each in its
    own color, at the per-part help level. Check also reveals the current layer.
    shown: the layers to stack (default: the practice selection)."""
    rgb = np.zeros((CANVAS, CANVAS, 3), np.float32)  # premultiplied
    alpha = np.zeros((CANVAS, CANVAS, 1), np.float32)
    if shown is None:
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
    probe = S["probe"]
    if on:
        g, rgb = overlay(levels, check)
    elif probe and probe["guides"]:
        # probe context: guides of other steps the drawer switched on (never the probed
        # step), at the strongest level for every part: context to place the redraw, not
        # the practised help. Every probe logs which were on.
        g, rgb = overlay([fading.L3] * len(ghost.PARTS), False, probe["guides"])
    else:
        g, rgb = np.zeros((CANVAS, CANVAS)), (0, 0, 0)
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
        # during a probe: the guides switched on for it (the practice selection comes back after)
        "visible": sorted(probe["guides"] if probe else S["visible"]),
        # prepare mode only: a study session (package loaded) never sends the target,
        # so it cannot reach the participant's tablet
        "target_img": (gray_png(S["gray"]) if SHOW_TARGET and S["gray"] is not None
                       and not S["package"] else None),
        # prepare mode: the candidate targets of the last Generate, as line drawings
        "choices": ([] if S["package"] else
                    [{"thumb": choice_thumb(c["layers"]), "seed": c["seed"],
                      "valid": c["valid"], "conf": min(c["conf"].values())}
                     for c in S["choices"]]),
        "choice": None if S["package"] else S["choice"],
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
    guides = [ghost.LAYERS[i] for i in sorted(S["probe"]["guides"])]
    own = [str(x) for x in p.get("own_shown", [])]  # the drawer's earlier steps shown faint
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
          strokes=[s["pts"] for s in strokes],
          guides=guides, own_shown=own,     # context shown during the probe
          timeout=bool(p.get("timeout")))  # True: submitted by the probe time limit
    S["probe"] = None
    return {**state(), "results": results,
            "changes": [{"part": a, "from": b, "to": c} for a, b, c in changes]}


# --- saved drawings --------------------------------------------------------------
# exports/<group>/<folder>/<participant>_<group>_<stage>[_pass<N>]_<key>.png
#   folder   block-in .. shadow: one practice step (only that step's strokes)
#            final: a practice pass, its finishing steps combined (FINAL_LAYERS, stage "final"), and the
#                   formal test drawings (stage = baseline / post / retention / transfer)
#   key      8 random hex digits, so a file name never repeats
# exports/index.csv lists every saved drawing (one row each).
INDEX_COLS = ["time", "participant", "group", "kind", "stage", "pass", "target", "key", "file"]


def strokes_png(strokes):
    """A drawing as saved: black strokes at their real pen width on white, 480 x 480."""
    return Image.fromarray((~rasterize(strokes, UI, 1.0, width=None)).astype(np.uint8) * 255)


def save_drawing(img, kind, stage, pass_no=None):
    """Save one drawing under the participant's group, add it to index.csv, return its path
    relative to exports/."""
    c = S["ctrl"]
    key = secrets.token_hex(4)
    folder = stage if stage in ghost.LAYERS else "final"
    name = "_".join([c.participant, c.group, stage]
                    + ([f"pass{pass_no}"] if pass_no else []) + [key]) + ".png"
    os.makedirs(os.path.join(HERE, "exports", c.group, folder), exist_ok=True)
    img.save(os.path.join(HERE, "exports", c.group, folder, name))
    rel = f"{c.group}/{folder}/{name}"
    index = os.path.join(HERE, "exports", "index.csv")
    new = not os.path.exists(index)
    with open(index, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(INDEX_COLS)
        wr.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), c.participant, c.group, kind, stage,
                     pass_no or "", target_label(), key, rel])
    return rel


def export(p):
    """Formal test drawing -> exports/<group>/final/<participant>_<group>_<task>_<key>.png."""
    strokes = p.get("strokes", [])
    rel = save_drawing(strokes_png(strokes), "test", S["task"])
    key = rel.rsplit("_", 1)[1][:-4]
    timeout = bool(p.get("timeout"))  # True: saved by the test time limit
    event("export", code=key, file=rel, n_strokes=len(strokes), timeout=timeout)
    log.info("export %s task=%s participant=%s%s", rel, S["task"], S["ctrl"].participant,
             " (time limit)" if timeout else "")
    return {"code": key, "file": rel}


def practice_pass(c):
    """The participant's latest practice pass number (0 before the first)."""
    pat = re.compile(rf"{re.escape(c.participant)}_{c.group}_{ghost.LAYERS[0]}_pass(\d+)_[0-9a-f]+\.png")
    folder = os.path.join(HERE, "exports", c.group, ghost.LAYERS[0])
    names = os.listdir(folder) if os.path.isdir(folder) else []
    return max((int(m.group(1)) for m in map(pat.fullmatch, names) if m), default=0)


def practice_step(p):
    """Practice step done -> exports/<group>/<layer>/<participant>_<group>_<layer>_pass<N>_<key>.png;
    after the last step also exports/<group>/final/<participant>_<group>_final_pass<N>_<key>.png
    (the finishing steps of the pass combined, FINAL_LAYERS: construction lines erased, as in
    a finished drawing; block-in and skeleton stay in their own step images). Step 1 starts a
    new pass, so a second practice block or Practise again never overwrites an earlier one."""
    c = S["ctrl"]
    if not c.path:
        return {"error": "practice images need a study session"}
    if S["task"] != "practice" or S["probe"] is not None:
        return {"error": "practice images are saved at the end of a practice step"}
    latest = practice_pass(c)
    n = latest + 1 if S["layer"] == 0 or latest == 0 else latest
    strokes = p.get("strokes", [])
    files = [save_drawing(strokes_png(strokes), "practice", ghost.LAYERS[S["layer"]], n)]
    if S["layer"] == len(ghost.LAYERS) - 1:  # last step: the finishing steps of this pass combined
        steps = []
        for layer in FINAL_LAYERS:  # the newest image of each finishing step in this pass
            folder = os.path.join(HERE, "exports", c.group, layer)
            mine = [os.path.join(folder, f) for f in os.listdir(folder)
                    if f.startswith(f"{c.participant}_{c.group}_{layer}_pass{n}_")]
            if mine:
                steps.append(np.asarray(Image.open(max(mine, key=os.path.getmtime))))
        final = np.minimum.reduce(steps) if steps else np.asarray(strokes_png([]))
        files.append(save_drawing(Image.fromarray(final), "practice", "final", n))
    event("practice_step", pass_no=n, files=files, n_strokes=len(strokes))
    log.info("practice step saved: %s", ", ".join(f"exports/{f}" for f in files))
    return {"pass": n, "files": files}


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
             task="practice", gray=None, conf=meta.get("confidence"), valid=True,
             choices=[], choice=None)
    log.info("session %s participant=%s group=%s package=%s | %s",
             "resumed" if existed else "started", pid, ctrl.group, pkg, fmt_levels())
    event("session_start", resumed=existed, levels=ctrl.levels())
    return {**state(), "resumed": existed}  # the participant view tells the researcher


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
        return build_target() or state()  # the next N_CHOICES seeds
    if path == "/api/choose":
        if S["package"] or not S["choices"]:
            return {"error": "no candidates to choose from: generate a target first"}
        i = int(p.get("index", -1))
        if not 0 <= i < len(S["choices"]):
            return {"error": f"candidate index must be 0 to {len(S['choices']) - 1}"}
        choose(i)
        return state()
    if path == "/api/package/save":
        if S["package"] or not S["valid"]:
            return {"error": "only a new target that passed the landmark check"}
        pid = package.save(S["layers"], S["gray"], {
            "prompt": S["prompt"], "style": S["style"], "seed": S["seed"],
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
        want = {int(i) for i in p.get("visible", []) if 0 <= int(i) < len(ghost.LAYERS)}
        if S["probe"]:  # probe context: other steps' guides only, apart from the practice selection
            S["probe"]["guides"] = want - {S["probe"]["layer"]}
            names = [ghost.LAYERS[i] for i in sorted(S["probe"]["guides"])]
            log.info("probe guides: %s", ",".join(names) or "none")
            event("probe_guides", guides=names)
            return state()
        S["visible"] = want
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
        S["probe"] = {"layer": S["layer"], "t0": time.time(), "guides": set()}  # guides start off
        log.info("probe start layer=%s", ghost.LAYERS[S["layer"]])
        event("probe_start", levels=S["ctrl"].levels())
        return state()
    if path == "/api/probe/submit":
        if S["probe"] is None:
            return {"error": "no probe running"}
        return probe_submit(p)
    if path == "/api/export":
        return export(p)
    if path == "/api/practice/step":
        return practice_step(p)
    if path == "/api/summary":
        return summary()
    if path == "/api/state":  # read the current state without changing it
        return state()
    if path == "/api/erase":  # Undo / Clear / eraser on the canvas: logged, nothing else changes
        kind = p.get("kind")
        if kind not in ("undo", "clear", "eraser"):
            return {"error": "kind must be undo, clear or eraser"}
        # undo / clear: whole strokes; eraser: the runs of points rubbed out in one gesture
        removed = [s.get("pts", []) for s in p.get("strokes", [])]
        log.info("%s: %d %s removed from the canvas, layer=%s%s", kind, len(removed),
                 "piece(s)" if kind == "eraser" else "stroke(s)",
                 ghost.LAYERS[S["layer"]], " (probe)" if S["probe"] else "")
        event(kind, n=len(removed), probe=S["probe"] is not None, removed=removed)
        return {"ok": True}
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
        global last_seen
        last_seen = time.time()
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        # / : researcher app (prepare, review, sessions); /participant : the
        # simplified participant view on the tablet (same server and data)
        pages = {"/": "index.html", "/index.html": "index.html",
                 "/participant": "participant.html", "/participant.html": "participant.html"}
        page = pages.get(self.path.split("?")[0])
        if page is None:
            log.warning("GET %s 404", self.path)
            return self._send(404, {"error": "not found"})
        log.info("GET %s (page loaded)", self.path)
        with open(os.path.join(HERE, "static", page), "rb") as f:
            self._send(200, f.read(), "text/html; charset=utf-8")

    def do_POST(self):
        global last_seen
        t0 = last_seen = time.time()
        n = int(self.headers.get("Content-Length", 0))
        if self.path == "/api/ping":  # keep-alive from connect.bat; not logged, no lock
            self.rfile.read(n)
            return self._send(200, {"ok": True})
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


def idle_watchdog():
    """Exit when no request arrived for IDLE_EXIT seconds (laptop gone): the
    process ends and the GPU memory is released."""
    while True:
        time.sleep(5)
        idle = time.time() - last_seen
        if idle > IDLE_EXIT:
            log.info("=== SketchFade stop: no request for %.0fs (laptop disconnected), "
                     "releasing the GPU ===", idle)
            logging.shutdown()
            os._exit(0)


def main():
    global engine, last_seen
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
    if IDLE_EXIT > 0:
        log.info("idle exit: stops after %.0fs without a request", IDLE_EXIT)
        last_seen = time.time()  # model loading above does not count as idle
        threading.Thread(target=idle_watchdog, daemon=True).start()
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        log.info("=== SketchFade stop ===")


if __name__ == "__main__":
    main()
