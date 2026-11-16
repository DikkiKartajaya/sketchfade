"""Automated tests for the SketchFade study system (Study Instruments v0.4).

Three layers, from fast to full:

  TestFading   help-level logic and probe scoring (fading.py)
  TestAPI      the study API end to end through server.handle() (server.py, package.py)
  TestBrowser  the participant view in a real Chromium against a live server
               (static/participant.html): setup, warm-up, tests, timers, practice,
               probes, eraser, pen sizes and blinding; Transfer in the researcher app (/);
               and BOT01, a full session whose images are copied to tests/output/BOT01/
               (overview.png shows the practice steps, the combined finals and the tests)

Everything runs in a temporary sandbox: fake approved portrait packages, and the
server's data folders (sessions/, participants/, exports/, packages/) pointed at a
temp folder. The real study data and the tracked logs/sketchfade.log are never
touched, and neither the ShadowDraw engine nor the image generator is loaded.

Setup (once, in the app's virtual environment):

  pip install pytest playwright
  python -m playwright install chromium

Run (from the app folder):

  python -m pytest tests/test_auto.py -v              # everything
  python -m pytest tests/test_auto.py -v -k "not Browser"   # no browser, a few seconds
  python -m pytest tests/test_auto.py -v -k Browser --headed  # watch the browser tests

The browser tests are skipped (not failed) when Playwright or Chromium is missing.
"""

import copy
import csv
import json
import logging
import re
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))

import fading  # noqa: E402
import ghost  # noqa: E402
import package  # noqa: E402
import server  # noqa: E402

# Importing server.py attaches a file handler to logs/sketchfade.log (tracked in git):
# detach it, so a test run never writes to the real log.
for _h in list(logging.getLogger().handlers):
    if isinstance(_h, logging.FileHandler):
        logging.getLogger().removeHandler(_h)
        _h.close()

S0 = copy.deepcopy(server.S)          # the server's state as freshly started
L0, L1, L2, L3 = fading.L0, fading.L1, fading.L2, fading.L3
PARTS = ghost.PARTS
PROBE_LAYERS = ["block-in", "skeleton", "contours"]   # must match participant.html
PEN_SIZES = [1, 2, 3, 5, 8, 12]                        # must match participant.html
BLINDING_WORDS = ["fade", "less help", "is back", "adaptive", "fixed", "level"]


# --- fake portrait packages ------------------------------------------------------

def make_layers(dx=0):
    """Five (300, 300) label images (0 = empty, k = part k-1), like a real package."""
    def layer(draw_fn):
        im = Image.new("L", (300, 300), 0)
        draw_fn(ImageDraw.Draw(im))
        return np.asarray(im, np.int8)

    def block_in(d):
        d.ellipse([72 + dx, 52, 228 + dx, 268], outline=1, width=2)                 # head
        d.rectangle([105 + dx, 120, 135 + dx, 136], outline=2, width=2)              # eyes
        d.rectangle([165 + dx, 120, 195 + dx, 136], outline=2, width=2)
        d.line([150 + dx, 150, 140 + dx, 190, 160 + dx, 190], fill=3, width=2)       # nose
        d.line([125 + dx, 222, 175 + dx, 222], fill=4, width=2)                      # mouth
        d.rectangle([120 + dx, 268, 180 + dx, 298], outline=5, width=2)              # neck

    def skeleton(d):
        d.line([150 + dx, 50, 150 + dx, 270], fill=1, width=2)
        d.line([95 + dx, 128, 205 + dx, 128], fill=2, width=2)
        d.line([125 + dx, 185, 175 + dx, 185], fill=3, width=2)
        d.line([115 + dx, 222, 185 + dx, 222], fill=4, width=2)
        d.line([60 + dx, 290, 240 + dx, 290], fill=5, width=2)

    def contours(d):
        d.ellipse([78 + dx, 58, 222 + dx, 262], outline=1, width=2)
        d.ellipse([108 + dx, 122, 132 + dx, 134], outline=2, width=2)
        d.ellipse([168 + dx, 122, 192 + dx, 134], outline=2, width=2)
        d.line([148 + dx, 152, 142 + dx, 188, 158 + dx, 188], fill=3, width=2)
        d.arc([128 + dx, 210, 172 + dx, 232], 20, 160, fill=4, width=2)
        d.line([125 + dx, 262, 110 + dx, 298], fill=5, width=2)
        d.line([175 + dx, 262, 190 + dx, 298], fill=5, width=2)

    def shading(d):
        d.ellipse([85 + dx, 150, 120 + dx, 230], fill=1)
        d.rectangle([125 + dx, 270, 175 + dx, 296], fill=5)

    def shadow(d):
        d.pieslice([72 + dx, 40, 228 + dx, 200], 180, 360, fill=1)   # hair only: head

    return np.stack([layer(f) for f in (block_in, skeleton, contours, shading, shadow)])


def write_package(root, pid, status="approved", n_layers=5, dx=0):
    d = Path(root) / pid
    d.mkdir(parents=True)
    np.savez_compressed(d / "layers.npz", layers=make_layers(dx)[:n_layers])
    (d / "meta.json").write_text(json.dumps({
        "prompt": f"test portrait {pid}", "id": pid, "source": "test",
        "confidence": {p: 0.9 for p in PARTS[:4]}, "review": {"status": status}}))


def perfect_strokes(layer):
    """Strokes (in the 480 canvas space) that trace every guide pixel of a layer."""
    g = fading.guide_mask(layer)
    ys, xs = np.nonzero(g)
    k = 1 / server.SCALE
    return [{"pts": [[x * k, y * k], [x * k + 0.01, y * k]], "w": 2}
            for x, y in zip(xs.tolist(), ys.tolist())]


# --- sandbox -----------------------------------------------------------------------

@pytest.fixture
def sb(tmp_path, monkeypatch):
    """A temp data folder with two approved portraits (pkgA, pkgB) and a pending one
    (pkgP); the server and the package store are pointed at it; fresh server state."""
    for sub in ("sessions", "participants", "exports", "logs", "packages"):
        (tmp_path / sub).mkdir()
    shutil.copytree(APP / "static", tmp_path / "static")
    write_package(tmp_path / "packages", "pkgA")
    write_package(tmp_path / "packages", "pkgB", dx=6)
    write_package(tmp_path / "packages", "pkgP", status="pending")
    monkeypatch.setattr(server, "HERE", str(tmp_path))
    monkeypatch.setattr(package, "ROOT", str(tmp_path / "packages"))
    server.S.clear()
    server.S.update(copy.deepcopy(S0))
    yield tmp_path
    server.S.clear()
    server.S.update(copy.deepcopy(S0))


def api(path, **body):
    """Call the API like the browser does (JSON in, JSON out)."""
    out = server.handle(path, json.loads(json.dumps(body)))
    assert out is not None, f"{path} is not an endpoint"
    return json.loads(json.dumps(out))


def events(sb, pid, kind=None):
    p = Path(sb) / "sessions" / f"{pid}.jsonl"
    if not p.exists():
        return []
    ev = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [e for e in ev if kind is None or e["event"] == kind]


def index_rows(sb):
    """exports/index.csv: one row per saved drawing."""
    p = Path(sb) / "exports" / "index.csv"
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


NAME = r"(?P<pid>[A-Za-z0-9-]+)_(?P<group>adaptive|fixed)_(?P<stage>[a-z-]+)(?:_pass(?P<pass>\d+))?_(?P<key>[0-9a-f]{8})\.png"


def wait_for(cond, timeout=10.0, what="condition"):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


# =====================================================================================
# 1. Help levels and probe scoring (fading.py)
# =====================================================================================

class TestFading:
    @staticmethod
    def run(ctrl, outcomes, part="eyes"):
        changes = []
        for o in outcomes:
            changes += ctrl.update({part: {"outcome": o}})
        return changes

    def test_outcome_thresholds(self):
        assert fading.outcome(0.03, 0.80) == "success"
        assert fading.outcome(0.01, 0.99) == "success"
        assert fading.outcome(0.04, 0.70) == "neutral"
        assert fading.outcome(0.04, 0.95) == "neutral"      # a bit too far, even with full coverage
        assert fading.outcome(0.06, 0.95) == "failure"      # too far from the guide
        assert fading.outcome(0.01, 0.49) == "failure"      # too little of the guide drawn

    def test_new_participant_starts_at_strong_help(self):
        c = fading.Controller(None, "P", "adaptive")
        assert c.levels() == [L3] * len(PARTS)

    def test_two_successes_in_a_row_lower_one_level(self):
        c = fading.Controller(None, "P", "adaptive")
        assert self.run(c, ["success"]) == []
        assert self.run(c, ["success"]) == [("eyes", L3, L2)]
        assert c.level["eyes"] == L2 and c.level["nose"] == L3   # only the probed part

    def test_neutral_or_opposite_outcome_breaks_the_streak(self):
        c = fading.Controller(None, "P", "adaptive")
        assert self.run(c, ["success", "neutral", "success", "failure", "success"]) == []
        assert c.level["eyes"] == L3

    def test_two_failures_raise_help_but_never_above_L3(self):
        c = fading.Controller(None, "P", "adaptive")
        self.run(c, ["success"] * 2)
        assert c.level["eyes"] == L2
        assert self.run(c, ["failure"] * 2) == [("eyes", L2, L3)]
        assert self.run(c, ["failure"] * 4) == []
        assert c.level["eyes"] == L3

    def test_L0_needs_six_successful_probes(self):
        """v0.4: 3 probes per part per block, 2 blocks = 6 = the minimum to reach L0."""
        c = fading.Controller(None, "P", "adaptive")
        self.run(c, ["success"] * 5)
        assert c.level["eyes"] == L1
        self.run(c, ["success"])
        assert c.level["eyes"] == L0
        assert self.run(c, ["success"] * 4) == [] and c.level["eyes"] == L0   # floor

    def test_fixed_group_never_changes_level(self):
        c = fading.Controller(None, "P", "fixed")
        assert self.run(c, ["success"] * 10 + ["failure"] * 4) == []
        assert c.levels() == [L3] * len(PARTS)

    def test_levels_and_group_survive_a_reload(self, tmp_path):
        path = str(tmp_path / "P.json")
        c = fading.Controller(path, "P", "adaptive")
        self.run(c, ["success"] * 2)
        again = fading.Controller(path, "P", "fixed")        # a different pick later
        assert again.group == "adaptive"                      # the first assignment wins
        assert again.level["eyes"] == L2

    def test_unknown_group_is_refused(self):
        with pytest.raises(ValueError):
            fading.Controller(None, "P", "control")

    def test_exact_redraw_scores_success_for_every_part(self):
        layers = make_layers()
        for i in range(3):                                    # the probed line steps
            drawn = fading.guide_mask(layers[i]) > 0
            res = fading.score_probe(drawn, layers[i])
            assert set(res) == set(PARTS)
            assert all(r["outcome"] == "success" for r in res.values()), res

    def test_empty_page_scores_failure(self):
        layers = make_layers()
        res = fading.score_probe(np.zeros((300, 300), bool), layers[0])
        assert all(r["outcome"] == "failure" for r in res.values())

    def test_parts_missing_from_a_layer_are_not_judged(self):
        layers = make_layers()
        res = fading.score_probe(fading.guide_mask(layers[4]) > 0, layers[4])
        assert set(res) == {"head"}                           # the shadow layer has hair only

    def test_guide_strength_per_level(self):
        layers = make_layers()
        levels = [L3, L2, L1, L0, L3]                         # head, eyes, nose, mouth, neck
        px = {p: tuple(np.argwhere(layers[0] == k + 1)[0]) for k, p in enumerate(PARTS)}
        a = fading.ghost(layers, 0, levels)
        assert a[px["head"]] == pytest.approx(0.60)
        assert a[px["eyes"]] == pytest.approx(0.25)
        assert a[px["nose"]] == 0                             # L1: nothing until Check
        assert a[px["mouth"]] == 0                            # L0: never
        assert fading.ghost(layers, 0, levels, check=True)[px["nose"]] == pytest.approx(0.60)
        tone = tuple(np.argwhere(layers[3] == 1)[0])          # filled layers are lighter
        assert fading.ghost(layers, 3, levels)[tone] == pytest.approx(0.36)


# =====================================================================================
# 2. Study API (server.py + package.py)
# =====================================================================================

class TestAPI:
    def test_only_approved_packages_with_five_layers_start(self, sb):
        write_package(sb / "packages", "pkgShort", n_layers=3)
        assert "error" in api("/api/start", participant="bad id!", group="adaptive", package="pkgA")
        assert "error" in api("/api/start", participant="P01", group="adaptive", package="")
        assert "not found" in api("/api/start", participant="P01", group="adaptive", package="nope")["error"]
        assert "not approved" in api("/api/start", participant="P01", group="adaptive", package="pkgP")["error"]
        assert "3 layers" in api("/api/start", participant="P01", group="adaptive", package="pkgShort")["error"]
        assert "error" in api("/api/start", participant="P01", group="control", package="pkgA")
        assert not (sb / "participants" / "P01.json").exists()

    def test_start_then_resume_keeps_the_first_group(self, sb):
        r = api("/api/start", participant="P01", group="adaptive", package="pkgA")
        assert r["resumed"] is False and r["group"] == "adaptive" and r["task"] == "practice"
        r = api("/api/start", participant="P01", group="fixed", package="pkgA")
        assert r["resumed"] is True and r["group"] == "adaptive"
        saved = json.loads((sb / "participants" / "P01.json").read_text())
        assert saved["group"] == "adaptive"

    def test_unknown_task_is_refused(self, sb):
        api("/api/start", participant="P01", group="adaptive", package="pkgA")
        assert "error" in api("/api/task", task="homework")

    @pytest.mark.parametrize("task", ["baseline", "post", "retention", "transfer"])
    def test_formal_tests_give_no_help(self, sb, task):
        api("/api/start", participant="P01", group="adaptive", package="pkgA")
        st = api("/api/task", task=task)
        assert st["helped"] is False
        assert "error" in api("/api/check")                   # no Check
        assert "error" in api("/api/probe/start")             # no probes
        pts = [[200.0, 150.0], [240.0, 150.0], [280.0, 150.0]]
        r = api("/api/stroke", pts=pts, w=3)
        assert "part" not in r and "err" not in r            # no feedback
        assert np.allclose(r["pts"], pts, atol=0.2)           # no correction
        ev = events(sb, "P01", "stroke")[-1]
        assert ev["helped"] is False and ev["corrected"] is False

    def test_practice_corrects_only_at_L3(self, sb):
        api("/api/start", participant="P01", group="adaptive", package="pkgA")
        layer0 = server.S["layers"][0]
        y, x = np.argwhere(layer0 == 1)[0]                   # a head guide pixel
        k = 1 / server.SCALE
        pts = [[(x + 3) * k, (y + 3) * k], [(x + 4) * k, (y + 3) * k]]   # 3 px off the guide
        r = api("/api/stroke", pts=pts, w=3)
        assert "part" in r                                    # practice gives feedback
        assert not np.allclose(r["pts"], pts, atol=0.2)       # L3: pulled toward the guide
        server.S["ctrl"].level = {p: L2 for p in PARTS}
        r = api("/api/stroke", pts=pts, w=3)
        assert np.allclose(r["pts"], pts, atol=0.2)           # L2: never corrected

    def test_probes_run_only_in_practice_and_hide_all_help(self, sb):
        api("/api/start", participant="P01", group="adaptive", package="pkgA")
        assert "error" in api("/api/probe/submit", strokes=[])   # no probe running
        st = api("/api/probe/start")
        assert st["probe"] == 0 and st["helped"] is False
        assert "error" in api("/api/check")
        r = api("/api/stroke", pts=[[100, 100], [140, 100]], w=3)
        assert "part" not in r
        assert events(sb, "P01", "probe_stroke")

    def test_adaptive_group_fades_after_two_successful_probes(self, sb):
        api("/api/start", participant="P01", group="adaptive", package="pkgA")
        good = perfect_strokes(server.S["layers"][0])
        api("/api/probe/start")
        r1 = api("/api/probe/submit", strokes=good)
        assert all(v["outcome"] == "success" for v in r1["results"].values()), r1["results"]
        assert r1["changes"] == []
        api("/api/probe/start")
        r2 = api("/api/probe/submit", strokes=good)
        assert {c["part"] for c in r2["changes"]} == set(PARTS)
        assert all(c["from"] == L3 and c["to"] == L2 for c in r2["changes"])
        assert [p["level"] for p in r2["levels"]] == [L2] * len(PARTS)

    def test_fixed_group_keeps_strong_help(self, sb):
        api("/api/start", participant="P02", group="fixed", package="pkgA")
        good = perfect_strokes(server.S["layers"][0])
        for _ in range(4):
            api("/api/probe/start")
            r = api("/api/probe/submit", strokes=good)
            assert r["changes"] == []
        assert [p["level"] for p in r["levels"]] == [L3] * len(PARTS)

    def test_empty_probe_counts_as_failure(self, sb):
        api("/api/start", participant="P01", group="adaptive", package="pkgA")
        api("/api/probe/start")
        r = api("/api/probe/submit", strokes=[])
        assert r["results"] and all(v["outcome"] == "failure" for v in r["results"].values())

    def test_probe_log_records_timeout(self, sb):
        api("/api/start", participant="P01", group="adaptive", package="pkgA")
        api("/api/probe/start")
        api("/api/probe/submit", strokes=[], timeout=True)
        api("/api/probe/start")
        api("/api/probe/submit", strokes=[])
        probes = events(sb, "P01", "probe")
        assert [p["timeout"] for p in probes] == [True, False]
        assert probes[0]["probe_layer"] == "block-in"

    @pytest.mark.parametrize("group", ["adaptive", "fixed"])
    def test_test_drawing_is_saved_in_its_group_final_folder(self, sb, group):
        api("/api/start", participant="P01", group=group, package="pkgA")
        api("/api/task", task="baseline")
        r = api("/api/export", strokes=[{"pts": [[10, 10], [300, 300]], "w": 12}], timeout=True)
        m = re.fullmatch(rf"{group}/final/{NAME}", r["file"])
        assert m and m["pid"] == "P01" and m["group"] == group and m["stage"] == "baseline"
        assert m["pass"] is None and m["key"] == r["code"]
        img = np.asarray(Image.open(sb / "exports" / r["file"]))
        assert img.shape == (480, 480) and img.min() == 0 and img.max() == 255
        rows = index_rows(sb)
        assert len(rows) == 1 and rows[0]["file"] == r["file"]
        assert {k: rows[0][k] for k in ("participant", "group", "kind", "stage", "pass", "target", "key")} == \
            {"participant": "P01", "group": group, "kind": "test", "stage": "baseline", "pass": "",
             "target": "pkgA", "key": r["code"]}
        ev = events(sb, "P01", "export")[-1]
        assert ev["timeout"] is True and ev["n_strokes"] == 1 and ev["file"] == r["file"]
        assert not (sb / "exports" / "key.csv").exists()
        assert list((sb / "exports").glob("*.png")) == []           # nothing loose in exports/

    def test_practice_steps_go_to_layer_folders_and_the_combined_final_to_final(self, sb):
        api("/api/start", participant="P01", group="fixed", package="pkgA")
        line = lambda k: [{"pts": [[40 + 80 * k, 40], [40 + 80 * k, 440]], "w": 3 + k}]   # one per step
        r = api("/api/practice/step", strokes=line(0))
        assert r["pass"] == 1 and len(r["files"]) == 1
        assert re.fullmatch(rf"fixed/block-in/{NAME}", r["files"][0])
        files = r["files"]
        for k in range(1, 5):
            api("/api/layer", layer=k)
            r = api("/api/practice/step", strokes=line(k))
            files += r["files"]
        assert len(r["files"]) == 2                                 # last step: + the combined final
        parsed = [re.fullmatch(rf"fixed/(?P<folder>[a-z-]+)/{NAME}", f) for f in files]
        assert all(parsed)
        assert [(p["folder"], p["stage"], p["pass"]) for p in parsed] == \
            [(n, n, "1") for n in ghost.LAYERS] + [("final", "final", "1")]
        steps = [np.asarray(Image.open(sb / "exports" / f)) for f in files[:5]]
        final = np.asarray(Image.open(sb / "exports" / files[5]))
        assert all(s.shape == (480, 480) and s.min() == 0 and s.max() == 255 for s in steps)
        assert (final == np.minimum.reduce(steps)).all()           # final = every step on one page
        api("/api/layer", layer=0)                                  # Practise again / next block
        assert api("/api/practice/step", strokes=line(0))["pass"] == 2
        api("/api/layer", layer=1)
        assert api("/api/practice/step", strokes=line(1))["pass"] == 2
        rows = index_rows(sb)
        assert [(r["kind"], r["stage"], r["pass"]) for r in rows] == \
            [("practice", n, "1") for n in ghost.LAYERS] + [("practice", "final", "1")] + \
            [("practice", "block-in", "2"), ("practice", "skeleton", "2")]
        assert not (sb / "exports" / "adaptive").exists()           # nothing in the other group
        ev = events(sb, "P01", "practice_step")
        assert [e["pass_no"] for e in ev] == [1, 1, 1, 1, 1, 2, 2]

    def test_passes_are_counted_per_participant(self, sb):
        for pid in ("P01", "P02"):
            api("/api/start", participant=pid, group="adaptive", package="pkgA")
            for _ in range(2 if pid == "P01" else 1):
                api("/api/layer", layer=0)
                r = api("/api/practice/step", strokes=[{"pts": [[50, 50], [90, 90]], "w": 3}])
        assert r["pass"] == 1                                       # P02 starts at pass 1
        assert sorted(re.fullmatch(NAME, Path(x["file"]).name)["pass"] + x["participant"]
                      for x in index_rows(sb)) == ["1P01", "1P02", "2P01"]

    def test_practice_images_only_at_the_end_of_a_practice_step(self, sb):
        assert "error" in api("/api/practice/step", strokes=[])     # no session yet
        api("/api/start", participant="P01", group="adaptive", package="pkgA")
        api("/api/probe/start")
        assert "error" in api("/api/practice/step", strokes=[])     # during a probe
        api("/api/probe/submit", strokes=[])
        api("/api/task", task="post")
        assert "error" in api("/api/practice/step", strokes=[])     # in a formal test
        assert index_rows(sb) == [] and not (sb / "exports" / "adaptive").exists()

    def test_erase_accepts_undo_clear_and_eraser_only(self, sb):
        api("/api/start", participant="P01", group="adaptive", package="pkgA")
        for kind in ("undo", "clear", "eraser"):
            assert api("/api/erase", kind=kind, strokes=[{"pts": [[1, 1], [2, 2]]}]) == {"ok": True}
        assert "error" in api("/api/erase", kind="smudge", strokes=[])
        logged = [e["event"] for e in events(sb, "P01") if e["event"] in ("undo", "clear", "eraser")]
        assert logged == ["undo", "clear", "eraser"]
        assert events(sb, "P01", "eraser")[0]["removed"] == [[[1, 1], [2, 2]]]

    def test_hidden_target_is_never_sent_in_a_study_session(self, sb):
        api("/api/start", participant="P01", group="adaptive", package="pkgA")
        server.S["gray"] = np.ones((300, 300), np.float32)    # even if a target were loaded
        assert api("/api/state")["target_img"] is None
        if server.SHOW_TARGET:                                # prepare mode (no package) still shows it
            server.S["package"] = None
            assert api("/api/state")["target_img"] is not None

    def test_every_logged_event_has_its_context(self, sb):
        api("/api/start", participant="P01", group="fixed", package="pkgA")
        api("/api/stroke", pts=[[100, 100], [150, 120]], w=3)
        api("/api/probe/start")
        api("/api/probe/submit", strokes=[])
        api("/api/task", task="post")
        api("/api/export", strokes=[])
        ev = events(sb, "P01")
        assert {e["event"] for e in ev} >= {"session_start", "stroke", "probe_start", "probe", "task", "export"}
        for e in ev:
            assert e["participant"] == "P01" and e["group"] == "fixed" and e["target"] == "pkgA"
            assert e["t"] > 0 and e["time"]


# =====================================================================================
# 3. Participant view in a real browser (participant.html + live server)
# =====================================================================================

@pytest.fixture(scope="session")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api", reason="pip install playwright")
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Chromium for Playwright is missing (python -m playwright install chromium): {e}")
        yield b
        b.close()


@pytest.fixture
def live(sb):
    """The real HTTP server (pages + API) on a free port, serving the sandbox."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def ui(browser, live):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    page = ctx.new_page()
    page.on("dialog", lambda d: d.accept())                   # the gear's confirm()
    page.goto(live + "/participant")
    page.wait_for_function("document.querySelectorAll('#pkg option').length >= 2")
    yield page
    ctx.close()


# --- browser helpers ---

def cover_title(page):
    return page.evaluate("() => document.getElementById('cover').classList.contains('hidden')"
                         " ? null : document.getElementById('coverTitle').textContent")


def wait_cover(page, title, timeout=15000):
    page.wait_for_function(
        "t => !document.getElementById('cover').classList.contains('hidden')"
        " && document.getElementById('coverTitle').textContent === t", arg=title, timeout=timeout)


def press(page, label):
    page.locator("#coverBtns").get_by_role("button", name=label, exact=True).click()


def setup(page, pid, task, group="adaptive", pkg="pkgA", test_min="10", probe_sec="0"):
    page.fill("#pid", pid)
    page.select_option("#group", group)
    page.select_option("#pkg", pkg)
    page.select_option("#taskSel", task)
    page.fill("#testMin", test_min)
    page.fill("#probeSec", probe_sec)
    page.click("#go")


def enter(page):
    """After Start: pass the returning-participant screen and the task's start screen."""
    page.wait_for_selector("#app:not(.hidden)")
    if cover_title(page) == "Returning participant":
        press(page, "Continue")
    if cover_title(page):
        press(page, "Start")


def draw(page, pts):
    """Drag the mouse through pts, given as fractions of the canvas."""
    if page.locator("#cover").count():                          # participant view only
        page.wait_for_selector("#cover.hidden", state="attached")
    box = page.locator("#ink").bounding_box()
    xy = [(box["x"] + fx * box["width"], box["y"] + fy * box["height"]) for fx, fy in pts]
    page.mouse.move(*xy[0])
    page.mouse.down()
    for x, y in xy[1:]:
        page.mouse.move(x, y, steps=8)
    page.mouse.up()


def stroke(page, y=0.5, x0=0.3, x1=0.7):
    """Draw one horizontal stroke and wait until the server has it."""
    with page.expect_response(lambda r: r.url.endswith("/api/stroke")):
        draw(page, [(x0, y), (x1, y)])


def finish_test(page):
    stroke(page)
    page.click("#main")
    wait_cover(page, "Thank you")
    press(page, "OK")
    page.wait_for_selector("#setup:not(.hidden)")


def run_practice_pass(page, step_check=None):
    """One pass through the five steps; redraw on every probe screen."""
    for k in range(5):
        stroke(page, y=0.3 + 0.08 * k)
        page.click("#main")
        if ghost.LAYERS[k] in PROBE_LAYERS:
            wait_cover(page, "Now on your own")
            with page.expect_response(lambda r: r.url.endswith("/api/probe/start")):
                press(page, "OK, start")
            stroke(page)
            page.click("#main")
        if k < 4:
            page.wait_for_function("n => document.getElementById('stepName').textContent"
                                   ".startsWith('Step ' + n + ' ')", arg=k + 2)
        if step_check:
            step_check(page)
    wait_cover(page, "All steps done")


def participant_text(page):
    """Everything the participant can read on the tablet right now (not the setup)."""
    return page.evaluate("""() => [document.getElementById('app').innerText,
        document.getElementById('cover').classList.contains('hidden') ? ''
          : document.getElementById('coverCard').innerText].join(' ')""")


def researcher_transfer(page, live, pid, group, pkg, shapes=None):
    """The Transfer test as run in the researcher app (/): start/resume, Task = transfer,
    draw, Finish test drawing."""
    page.goto(live + "/")
    page.wait_for_function("pk => [...document.querySelectorAll('#pkg option')].some(o => o.value === pk)",
                           arg=pkg)
    page.fill("#pid", pid)
    page.select_option("#group", group)
    page.select_option("#pkg", pkg)
    with page.expect_response(lambda r: r.url.endswith("/api/start")):
        page.click("#start")
    with page.expect_response(lambda r: r.url.endswith("/api/task")):
        page.select_option("#task", "transfer")
    for pts in shapes or [[(0.3, 0.5), (0.7, 0.5)]]:
        with page.expect_response(lambda r: r.url.endswith("/api/stroke")):
            draw(page, pts)
    with page.expect_response(lambda r: r.url.endswith("/api/export")):
        page.click("#export")


def ring(cx, cy, rx, ry=None, n=28, a0=0.0, a1=360.0):
    """An ellipse (or arc) as canvas fractions, for drawing face-like strokes."""
    ry = rx if ry is None else ry
    return [(cx + rx * np.cos(np.radians(a)), cy + ry * np.sin(np.radians(a)))
            for a in np.linspace(a0, a1, n)]


# Face-like strokes per construction step, and the pen size used for that step.
FACE = {
    "block-in": (2, [ring(0.5, 0.42, 0.2, 0.25), ring(0.42, 0.38, 0.04, 0.02), ring(0.58, 0.38, 0.04, 0.02),
                     [(0.5, 0.42), (0.47, 0.5), (0.53, 0.5)], [(0.44, 0.57), (0.56, 0.57)],
                     [(0.44, 0.66), (0.42, 0.8)], [(0.56, 0.66), (0.58, 0.8)]]),
    "skeleton": (1, [[(0.5, 0.15), (0.5, 0.7)], [(0.32, 0.38), (0.68, 0.38)], [(0.42, 0.5), (0.58, 0.5)],
                     [(0.4, 0.57), (0.6, 0.57)], [(0.25, 0.82), (0.75, 0.82)]]),
    "contours": (3, [ring(0.5, 0.43, 0.19, 0.24, a0=200, a1=520), ring(0.42, 0.38, 0.035, 0.015),
                     ring(0.58, 0.38, 0.035, 0.015), ring(0.5, 0.56, 0.06, 0.02, a0=10, a1=170),
                     [(0.3, 0.3), (0.36, 0.2), (0.5, 0.16), (0.64, 0.2), (0.7, 0.3)]]),
    "shading": (8, [[(0.34, 0.45 + 0.03 * i), (0.4, 0.43 + 0.03 * i)] for i in range(5)]),
    "shadow": (12, [[(0.32, 0.24 + 0.025 * i), (0.68, 0.24 + 0.025 * i)] for i in range(3)]),
}
TEST_FACE = FACE["contours"][1]


def draw_strokes(page, shapes):
    for pts in shapes:
        with page.expect_response(lambda r: r.url.endswith("/api/stroke")):
            draw(page, pts)


def copy_real_package(sb):
    """Copy an approved 5-layer package of the real app (e.g. 'kid with t shirt') into the
    sandbox, so BOT01 practises on a real portrait. None when the app has none."""
    for d in sorted((APP / "packages").glob("*/")):
        try:
            meta = json.loads((d / "meta.json").read_text())
            with np.load(d / "layers.npz") as z:
                ok = z["layers"].shape[0] == 5
        except (OSError, ValueError, KeyError):
            continue
        if ok and meta.get("review", {}).get("status") == "approved":
            shutil.copytree(d, sb / "packages" / d.name)
            return d.name
    return None


@contextmanager
def api_responses(page):
    got = []
    page.on("response", lambda r: got.append(r) if "/api/" in r.url else None)
    yield got


class TestBrowser:
    # --- setup screen ---

    def test_setup_lists_approved_portraits_and_the_participant_tasks(self, ui):
        opts = ui.eval_on_selector_all("#pkg option", "os => os.map(o => o.value)")
        assert sorted(opts) == ["pkgA", "pkgB"]                # pending pkgP is not offered
        assert ui.input_value("#taskSel") == "warmup"           # the first step of I3
        tasks = ui.eval_on_selector_all("#taskSel option", "os => os.map(o => o.value)")
        assert tasks == ["warmup", "baseline", "practice", "post", "retention"]   # Transfer: researcher app
        assert ui.locator("#pkgT").count() == 0
        assert ui.input_value("#testMin") == "10" and ui.input_value("#probeSec") == "15"

    def test_transfer_runs_in_the_researcher_app(self, browser, live, sb):
        page = browser.new_page()
        researcher_transfer(page, live, "P01", "adaptive", "pkgB")
        page.close()
        ev = events(sb, "P01", "export")
        assert len(ev) == 1 and ev[0]["task"] == "transfer" and ev[0]["target"] == "pkgB"
        row = index_rows(sb)[0]
        assert (row["stage"], row["target"], row["kind"]) == ("transfer", "pkgB", "test")
        assert re.fullmatch(rf"adaptive/final/P01_adaptive_transfer_[0-9a-f]{{8}}\.png", row["file"])
        assert (sb / "exports" / row["file"]).exists()

    def test_portrait_choice_is_kept_between_tasks(self, ui):
        setup(ui, "P01", "warmup", pkg="pkgA")                  # not the default (newest) one
        ui.wait_for_selector("#app:not(.hidden)")
        ui.click("#main")
        press(ui, "OK")
        ui.wait_for_selector("#setup:not(.hidden)")
        ui.wait_for_function("document.querySelectorAll('#pkg option').length >= 2")
        assert ui.input_value("#pkg") == "pkgA"

    # --- warm-up, pens, eraser ---

    def test_warm_up_logs_nothing(self, ui, sb):
        with api_responses(ui) as got:
            setup(ui, "P01", "warmup")
            ui.wait_for_selector("#app:not(.hidden)")
            draw(ui, [(0.2, 0.5), (0.8, 0.5)])
            ui.click("#eraser")
            draw(ui, [(0.5, 0.3), (0.5, 0.7)])
            ui.click("#undo")
            ui.click("#main")
            wait_cover(ui, "Warm-up done")
        assert all(r.url.endswith("/api/packages") for r in got)
        assert list((sb / "sessions").iterdir()) == []

    def test_six_pen_sizes_and_the_chosen_size_is_used(self, ui, sb):
        sizes = ui.eval_on_selector_all(".pen", "bs => bs.map(b => +b.dataset.w)")
        assert sizes == PEN_SIZES
        setup(ui, "P01", "practice")
        enter(ui)
        ui.click(".pen[data-w='12']")
        assert ui.evaluate("pen") == 12
        stroke(ui)
        assert events(sb, "P01", "stroke")[-1]["w"] == 12

    def test_eraser_rubs_out_part_of_a_line_and_is_logged(self, ui, sb):
        setup(ui, "P01", "practice")
        enter(ui)
        stroke(ui, y=0.5, x0=0.2, x1=0.8)
        ui.click("#eraser")
        assert "on" in ui.get_attribute("#eraser", "class") and ui.evaluate("tool") == "eraser"
        with ui.expect_response(lambda r: r.url.endswith("/api/erase")):
            draw(ui, [(0.5, 0.35), (0.5, 0.65)])                 # rub across the middle
        assert ui.evaluate("strokes.length") == 2                # the line is now two pieces
        ev = events(sb, "P01", "eraser")
        assert len(ev) == 1 and ev[0]["removed"] and ev[0]["probe"] is False
        ui.click("#undo")                                        # Undo removes the last piece
        assert ui.evaluate("strokes.length") == 1

    def test_every_new_screen_starts_with_the_pen(self, ui):
        setup(ui, "P01", "practice")
        enter(ui)
        stroke(ui)
        ui.click("#eraser")
        ui.click("#main")                                        # block-in ends with a probe
        wait_cover(ui, "Now on your own")
        press(ui, "OK, start")
        ui.wait_for_function("phase === 'probe'")
        assert ui.evaluate("tool") == "pen"
        assert "on" not in (ui.get_attribute("#eraser", "class") or "").split()

    # --- formal tests and their timer ---

    def test_baseline_start_screen_clock_and_saved_drawing(self, ui, sb):
        setup(ui, "P01", "baseline")
        ui.wait_for_selector("#app:not(.hidden)")
        wait_cover(ui, "Baseline drawing")
        assert "You have 10 minutes" in ui.text_content("#coverText")
        assert ui.text_content("#clock") == ""                   # not running before Start
        press(ui, "Start")
        assert ui.text_content("#clock").startswith("Time left 10:00") or \
            ui.text_content("#clock").startswith("Time left 9:5")
        assert ui.is_hidden("#check")                            # no help in tests
        finish_test(ui)
        ev = events(sb, "P01", "export")
        assert len(ev) == 1 and ev[0]["timeout"] is False and ev[0]["task"] == "baseline"
        assert [f.name for f in (sb / "exports" / "adaptive" / "final").iterdir()] == \
            [Path(ev[0]["file"]).name]
        assert Path(ev[0]["file"]).name.startswith("P01_adaptive_baseline_")

    def test_test_is_saved_automatically_when_time_is_up(self, ui, sb):
        setup(ui, "P01", "post", test_min="0.05")                # 3 seconds
        enter(ui)
        stroke(ui)
        wait_cover(ui, "Thank you", timeout=15000)
        ev = events(sb, "P01", "export")
        assert len(ev) == 1 and ev[0]["timeout"] is True and ev[0]["n_strokes"] == 1

    def test_time_up_during_a_save_does_not_save_twice(self, ui, sb):
        """'I'm finished' is pressed, the save is slow, and the clock runs out meanwhile:
        only one drawing may reach the server."""
        sent = []

        def slow_export(route):
            sent.append(time.time())
            if len(sent) == 1:
                time.sleep(6)                                    # the clock (6 s) runs out now
            route.continue_()
        ui.route("**/api/export", slow_export)
        setup(ui, "P01", "post", test_min="0.1")
        enter(ui)
        stroke(ui)
        ui.click("#main")
        wait_cover(ui, "Thank you", timeout=15000)
        ui.wait_for_timeout(1500)                                # room for a second save to show up
        assert len(sent) == 1
        assert len(events(sb, "P01", "export")) == 1

    def test_leaving_a_test_stops_its_timer(self, ui, sb):
        setup(ui, "P01", "retention", test_min="0.05")
        enter(ui)
        ui.click("#gear")                                        # confirm() is accepted
        ui.wait_for_selector("#setup:not(.hidden)")
        ui.wait_for_timeout(5000)                                # past the 3 s limit
        assert events(sb, "P01", "export") == []
        assert ui.evaluate("timer") is None

    def test_no_limit_means_no_clock(self, ui, sb):
        setup(ui, "P01", "post", test_min="0")
        ui.wait_for_selector("#app:not(.hidden)")
        if cover_title(ui) == "Returning participant":
            press(ui, "Continue")
        assert "Take your time" in ui.text_content("#coverText")
        press(ui, "Start")
        ui.wait_for_timeout(1500)
        assert ui.text_content("#clock") == "" and ui.evaluate("timer") is None

    # --- practice, probes, end screen ---

    def test_practice_pass_probes_only_the_line_steps(self, ui, sb):
        setup(ui, "P01", "practice")
        enter(ui)
        run_practice_pass(ui)
        probes = events(sb, "P01", "probe")
        assert [p["probe_layer"] for p in probes] == PROBE_LAYERS
        assert all(p["timeout"] is False for p in probes)
        assert [b.strip() for b in ui.locator("#coverBtns button").all_inner_texts()] == \
            ["Practise again", "I'm done"]

    def test_end_screen_im_done_keeps_the_screen_and_asks_for_the_researcher(self, ui, sb):
        setup(ui, "P01", "practice")
        enter(ui)
        run_practice_pass(ui)
        press(ui, "I'm done")
        assert "hand the tablet back" in ui.text_content("#note")
        assert ui.is_visible("#app") and ui.is_hidden("#setup")  # nothing ends on its own

    def test_practise_again_starts_over_and_probes_again(self, ui, sb):
        setup(ui, "P01", "practice")
        enter(ui)
        run_practice_pass(ui)
        press(ui, "Practise again")
        ui.wait_for_function("document.getElementById('stepName').textContent.startsWith('Step 1 ')")
        assert ui.evaluate("strokes.length") == 0
        run_practice_pass(ui)
        assert len(events(sb, "P01", "probe")) == 6              # 3 per pass

    def test_probe_is_submitted_automatically_when_time_is_up(self, ui, sb):
        setup(ui, "P01", "practice", probe_sec="2")
        enter(ui)
        stroke(ui)
        ui.click("#main")
        wait_cover(ui, "Now on your own")
        press(ui, "OK, start")
        ui.wait_for_function("document.getElementById('clock').textContent.startsWith('Time left')")
        ui.wait_for_function("document.getElementById('stepName').textContent.startsWith('Step 2 ')",
                             timeout=15000)
        probe = events(sb, "P01", "probe")
        assert len(probe) == 1 and probe[0]["timeout"] is True
        assert ui.text_content("#clock") == ""                   # cleared on the next step

    def test_help_level_changes_are_never_shown(self, ui, sb):
        """Even when the server reports a fading change, the tablet says nothing about it."""
        def add_change(route):
            resp = route.fetch()
            body = resp.json()
            body["changes"] = [{"part": "eyes", "from": 3, "to": 2},
                               {"part": "nose", "from": 2, "to": 3}]
            route.fulfill(response=resp, json=body)
        ui.route("**/api/probe/submit", add_change)
        setup(ui, "P01", "practice")
        enter(ui)
        seen = []
        run_practice_pass(ui, step_check=lambda p: seen.append(participant_text(p).lower()))
        seen.append(participant_text(ui).lower())
        for text in seen:
            for word in BLINDING_WORDS:
                assert word not in text, f"'{word}' shown to the participant: {text[:200]}"

    def test_intro_and_returning_screens_do_not_name_the_condition(self, ui, sb):
        setup(ui, "P01", "baseline", group="fixed")
        enter(ui)
        finish_test(ui)
        setup(ui, "P01", "practice", group="fixed")              # same pick: group not named
        ui.wait_for_selector("#app:not(.hidden)")
        wait_cover(ui, "Returning participant")
        text = ui.text_content("#coverText").lower()
        assert "fixed" not in text and "adaptive" not in text
        press(ui, "Continue")
        wait_cover(ui, "Let's draw a portrait")
        intro = ui.text_content("#coverText").lower()
        assert all(w not in intro for w in BLINDING_WORDS)

    def test_a_different_group_pick_is_flagged_and_ignored(self, ui, sb):
        setup(ui, "P01", "baseline", group="adaptive")
        enter(ui)
        finish_test(ui)
        setup(ui, "P01", "practice", group="fixed")
        ui.wait_for_selector("#app:not(.hidden)")
        wait_cover(ui, "Returning participant")
        assert "Group kept: adaptive" in ui.text_content("#coverText")
        assert ui.input_value("#group") == "adaptive"
        assert json.loads((sb / "participants" / "P01.json").read_text())["group"] == "adaptive"

    def test_target_image_never_reaches_the_tablet(self, ui, sb):
        with api_responses(ui) as got:
            setup(ui, "P01", "practice")
            enter(ui)
            run_practice_pass(ui)
        bodies = [r.json() for r in got if r.ok and "json" in (r.headers.get("content-type") or "")]
        assert any("target_img" in b for b in bodies)
        assert all(b.get("target_img") is None for b in bodies)

    def test_full_session_in_i3_order(self, ui, browser, live, sb):
        """Warm-up, baseline, practice x2, post-test, retention (participant view), then
        transfer (researcher app), as in I3 v0.4."""
        setup(ui, "P01", "warmup")
        ui.wait_for_selector("#app:not(.hidden)")
        ui.click("#main")
        press(ui, "OK")
        ui.wait_for_selector("#setup:not(.hidden)")
        setup(ui, "P01", "baseline")
        enter(ui)
        finish_test(ui)
        for _ in range(2):                                       # practice block 1 and 2
            setup(ui, "P01", "practice")
            enter(ui)
            run_practice_pass(ui)
            press(ui, "I'm done")
            ui.click("#gear")
            ui.wait_for_selector("#setup:not(.hidden)")
        for task in ("post", "retention"):
            setup(ui, "P01", task)
            enter(ui)
            finish_test(ui)
        researcher = browser.new_page()
        researcher_transfer(researcher, live, "P01", "adaptive", "pkgB")
        researcher.close()
        assert len(events(sb, "P01", "practice_step")) == 10     # 5 steps x 2 blocks
        exports = events(sb, "P01", "export")
        assert [e["task"] for e in exports] == ["baseline", "post", "retention", "transfer"]
        assert [e["target"] for e in exports] == ["pkgA", "pkgA", "pkgA", "pkgB"]
        assert len(events(sb, "P01", "probe")) == 6              # 3 + 3: the v0.4 schedule
        tests = [r for r in index_rows(sb) if r["kind"] == "test"]
        assert [r["stage"] for r in tests] == ["baseline", "post", "retention", "transfer"]
        assert all((sb / "exports" / r["file"]).exists() for r in tests)
        assert {e["group"] for e in events(sb, "P01")} == {"adaptive"}

    def test_bot01_full_session_saves_step_and_final_images(self, ui, browser, live, sb):
        """BOT01 takes the whole I3 session like a participant: warm-up, baseline, practice
        block 1 and 2 (face-like strokes per step, a pen size per step, probe redraws),
        post-test and retention in the participant view, then Transfer in the researcher app.
        Checks every saved image; copies them to tests/output/BOT01/ to look at
        (overview.png shows all of them on one sheet)."""
        pid = "BOT01"
        trained = copy_real_package(sb) or "pkgA"          # the real portrait when there is one
        ui.reload()
        ui.wait_for_function("pk => [...document.querySelectorAll('#pkg option')].some(o => o.value === pk)",
                             arg=trained)

        def formal_test(task):
            setup(ui, pid, task, pkg=trained)
            enter(ui)
            draw_strokes(ui, TEST_FACE)
            ui.click("#main")
            wait_cover(ui, "Thank you")
            press(ui, "OK")
            ui.wait_for_selector("#setup:not(.hidden)")

        # warm-up (nothing saved)
        setup(ui, pid, "warmup", pkg=trained)
        ui.wait_for_selector("#app:not(.hidden)")
        draw(ui, ring(0.5, 0.5, 0.2))
        ui.click("#main")
        press(ui, "OK")
        ui.wait_for_selector("#setup:not(.hidden)")
        formal_test("baseline")
        for _block in (1, 2):                               # practice block 1 and 2
            setup(ui, pid, "practice", pkg=trained)
            enter(ui)
            for k, layer in enumerate(ghost.LAYERS):
                width, shapes = FACE[layer]
                ui.click(f".pen[data-w='{width}']")
                draw_strokes(ui, shapes)
                ui.click("#main")
                if layer in PROBE_LAYERS:
                    wait_cover(ui, "Now on your own")
                    with ui.expect_response(lambda r: r.url.endswith("/api/probe/start")):
                        press(ui, "OK, start")
                    draw_strokes(ui, shapes[:2])            # the unassisted redraw
                    ui.click("#main")
                if k < 4:
                    ui.wait_for_function("n => document.getElementById('stepName').textContent"
                                         ".startsWith('Step ' + n + ' ')", arg=k + 2)
            wait_cover(ui, "All steps done")
            press(ui, "I'm done")
            ui.click("#gear")
            ui.wait_for_selector("#setup:not(.hidden)")
        formal_test("post")
        formal_test("retention")
        researcher = browser.new_page()
        researcher_transfer(researcher, live, pid, "adaptive", "pkgB", shapes=TEST_FACE)
        researcher.close()

        exports = sb / "exports"
        rows = index_rows(sb)
        assert {r["participant"] for r in rows} == {pid} and {r["group"] for r in rows} == {"adaptive"}
        assert not (exports / "fixed").exists()
        # the folders: one per layer plus final, inside the group
        assert sorted(d.name for d in (exports / "adaptive").iterdir()) == sorted(ghost.LAYERS + ["final"])
        # every file is in the folder of its stage and named <pid>_<group>_<stage>[_pass<N>]_<key>.png
        for r in rows:
            m = re.fullmatch(rf"adaptive/(?P<folder>[a-z-]+)/{NAME}", r["file"])
            assert m and m["pid"] == pid and m["group"] == "adaptive" and m["stage"] == r["stage"]
            assert m["folder"] == (r["stage"] if r["stage"] in ghost.LAYERS else "final")
            assert (m["pass"] or "") == r["pass"] and m["key"] == r["key"]
            assert (exports / r["file"]).exists()
        assert len({r["key"] for r in rows}) == len(rows)          # no name is used twice

        # practice: each layer folder holds pass1 and pass2; final holds both combined finals
        practice = [r for r in rows if r["kind"] == "practice"]
        for layer in ghost.LAYERS:
            assert sorted(r["pass"] for r in practice if r["stage"] == layer) == ["1", "2"]
        for n in ("1", "2"):
            steps = [np.asarray(Image.open(exports / next(r["file"] for r in practice
                                                          if r["stage"] == layer and r["pass"] == n)))
                     for layer in ghost.LAYERS]
            final = np.asarray(Image.open(exports / next(r["file"] for r in practice
                                                         if r["stage"] == "final" and r["pass"] == n)))
            assert all(s.shape == (480, 480) and s.min() == 0 for s in steps)   # ink in every step
            assert (final == np.minimum.reduce(steps)).all()                    # final = all steps
            assert (final == 0).sum() > max((s == 0).sum() for s in steps)
        assert len(practice) == 12                                  # (5 steps + final) x 2 passes
        assert len(events(sb, pid, "practice_step")) == 10
        assert len(events(sb, pid, "probe")) == 6

        # formal tests: in final/, the task as the stage, no pass
        tests = [r for r in rows if r["kind"] == "test"]
        assert [r["stage"] for r in tests] == ["baseline", "post", "retention", "transfer"]
        assert [r["target"] for r in tests] == [trained] * 3 + ["pkgB"]
        assert all(r["pass"] == "" for r in tests)
        assert sorted(f.name for f in (exports / "adaptive" / "final").iterdir()) == \
            sorted(Path(r["file"]).name for r in rows if r["stage"] not in ghost.LAYERS)
        assert list(exports.glob("*.png")) == [] and not (exports / "key.csv").exists()
        save_bot_output(sb, pid)


OUT = APP / "tests" / "output"


def save_bot_output(sb, pid):
    """Copy a bot run to tests/output/<pid>/ (git-ignored): exports/ exactly as the study
    machine would have it, the session log, and overview.png with every drawing on one
    sheet (one row per practice pass: the 5 steps + final; then the 4 tests)."""
    out = OUT / pid
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(sb / "exports", out / "exports")
    shutil.copy(sb / "sessions" / f"{pid}.jsonl", out / f"{pid}.jsonl")
    rows = index_rows(out)
    tile, lab = 240, 22
    sheet_rows = []
    for n in sorted({r["pass"] for r in rows if r["kind"] == "practice"}):
        order = ghost.LAYERS + ["final"]
        mine = sorted((r for r in rows if r["kind"] == "practice" and r["pass"] == n),
                      key=lambda r: order.index(r["stage"]))
        sheet_rows.append([(f"pass{n} {r['stage']}", out / "exports" / r["file"]) for r in mine])
    sheet_rows.append([(f"test {r['stage']}", out / "exports" / r["file"])
                       for r in rows if r["kind"] == "test"])
    cols = max(len(r) for r in sheet_rows)
    sheet = Image.new("L", (cols * tile, len(sheet_rows) * (tile + lab)), 255)
    d = ImageDraw.Draw(sheet)
    for i, row in enumerate(sheet_rows):
        for j, (label, path) in enumerate(row):
            x, y = j * tile, i * (tile + lab)
            sheet.paste(Image.open(path).convert("L").resize((tile, tile)), (x, y + lab))
            d.rectangle([x, y + lab, x + tile - 1, y + lab + tile - 1], outline=180)
            d.text((x + 6, y + 5), label, fill=0)
    sheet.save(out / "overview.png")
