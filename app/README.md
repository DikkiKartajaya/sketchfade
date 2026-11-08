# SketchFade app: Construction Ghost (B) + Fading Shadow (A)

Implementation of the final proposal (Lecture 4). B decides *what* to show
(construction layers of a hidden, validated and reviewed target); A decides *how
much* to show (levels L3–L0 per face part, set by unassisted probes).

Built on the ShadowDraw demo in `../shadowdraw-demo/demo` (its engine and face
database are imported, not copied).

## Run on a GPU server (e.g. RTX 6000 Ada)

The study runs here. The whole app runs on the server; the tablet's browser
reaches it through an SSH tunnel (the server only listens on 127.0.0.1, nothing
is exposed).

```bash
# laptop: copy the project (app + shadowdraw-demo, which holds the face database)
scp -r sketchfade user@gpu-server:~/

# server: environment
cd ~/sketchfade/app
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build, see pytorch.org
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.get_device_name(0))"          # must print the GPU
python generate.py                     # downloads models once, test should take ~1-2 s

# server: before recruitment, precompute the study targets (see "Prepare")
CUDA_VISIBLE_DEVICES=0 python prepare.py --prompts prompts.txt --seeds 6 --keep 1
python package.py list
python package.py review <id> --approve --reviewer "Illustrator name"

# server: run the study app
tmux new -s sketchfade                 # keeps running after you disconnect
CUDA_VISIBLE_DEVICES=0 python server.py 9100

# laptop / tablet: tunnel, then open http://127.0.0.1:9100/
ssh -N -L 9100:127.0.0.1:9100 user@gpu-server

# laptop: fetch the study data afterwards
scp -r user@gpu-server:~/sketchfade/app/{sessions,participants,exports,packages,logs} .
```

On a laptop without a GPU, `python server.py` (or double-click `run.bat`) still
works: generation takes about a minute on a CPU, and without `diffusers`/`peft`
the app uses the ShadowDraw retrieval face as target (no shading layer).

## 1 · Prepare targets (B, before recruitment)

Each step of the proposal's pipeline, and where it lives:

| Step | Code | What happens |
|---|---|---|
| Input | UI *Prepare targets* or `prepare.py` | rough sketch + prompt + recorded seed |
| Target generation | `generate.py` | SD 1.5 + ControlNet scribble + LCM-LoRA, 4 steps, pretrained only |
| Target validation | `ghost.find_landmarks` | eye line + eye centers, nose, mouth, head outline, each with a confidence; weakest < `CONF_MIN` → next seed (up to 3 in the UI) |
| Guide construction | `ghost.build_layers` | part masks and `block-in` → `skeleton` → `contours` → `shading`, placed on the landmarks |
| Expert review | `package.py review` | illustrator checks order and difficulty; approve or reject |
| Frozen package | `packages/<id>/` | `layers.npz`, `target.png`, `guides.png` (reviewer sheet), `meta.json` |

Only **approved** packages can be opened in a study session, and a package is
never regenerated. Use one package for practice and baseline/post/retention,
and an **unseen** package for the transfer test.

**Calibrate `CONF_MIN` on the GPU.** The current value (0.20) was set on
ShadowDraw database faces, where it rejects about 12 %. Generated targets have
denser edges, so run `python prepare.py --dry-run ...` on the GPU server during
the pilot, look at the printed confidence distribution and at a few
`guides.png` sheets, and adjust `CONF_MIN` in `ghost.py`.

## 2 · Study session (A)

Start a session with **participant id**, **group** (`adaptive` or `fixed`) and
an approved **package**. A returning participant (day 2, day 7) keeps the group
and the per-part levels stored in `participants/<id>.json`.

### Practice

| Level | Shown for that part | Code |
|---|---|---|
| L3 Strong | overlay at 0.60 + stroke correction (raw stroke kept) | `GUIDE_ALPHA`, `correct` |
| L2 Reduced | overlay at 0.25 | `GUIDE_ALPHA` |
| L1 On request | nothing; **Check** shows it at 0.60 for 2 s | `CHECK_ALPHA`, `CHECK_MS` |
| L0 Independent | only the side reference | |

The **side reference** (small image beside the canvas) shows the current
construction step at every level, in probes and in tests; it is the drawing
prompt, not an overlay.

The fixed group stays at L3 for every part. Both groups start at L3.

### Probes (both groups, same schedule)

Pressing **→** ends the layer with an **unassisted probe**: the canvas is
cleared (blank area), all help is hidden, and the user redraws that step. **Probe
this layer** runs one on demand, e.g. after the last layer. `fading.score_probe`
scores each part of the probed layer:

- **two-way error:** mean of drawn → guide (drawn pixels belong to the part of
  their nearest guide pixel) and guide → drawn, each point capped at 30 px, as a
  fraction of the 300 px frame;
- **coverage:** share of the part's guide pixels with a drawn pixel within 6 px;
- filled layers (shading) are judged by their outline.

| Outcome | Rule |
|---|---|
| success | error ≤ 0.03 and coverage ≥ 0.80 |
| failure | error ≥ 0.06 or coverage < 0.50 |
| neutral | anything in between (breaks a streak) |

Two consecutive successes lower that part's level by one (down to L0); two
consecutive failures raise it (up to L3). The fixed group logs the same
outcomes but its levels never change. All thresholds are constants at the top
of `fading.py`, to be checked in the 3-user pilot.

### Formal tests

Tasks `baseline`, `post`, `retention` and `transfer` disable overlays,
correction, Check and feedback (no error, no level panel) in both groups. The
side reference shows the `contours` layer of the session's package. **Finish
test drawing** saves the drawing as `exports/<code>.png` with a random code;
`exports/key.csv` maps codes to participant, group, task and target, so raters
only see the codes.

## Study logger

`sessions/<participant>.jsonl`: one JSON line per event, each with `t`,
`participant`, `group`, `task`, `target`, `layer`:

| `event` | Extra fields |
|---|---|
| `session_start`, `task`, `layer` | `levels` |
| `stroke`, `probe_stroke` | `part`, `level`, `err_px`, `helped`, `corrected`, `w`, `raw` points, `shown` (corrected) points |
| `check` | `reveals` |
| `probe_start`, `probe` | `probe_layer`, per-part `results` (`err`, `cov`, `outcome`), `levels_before`, `levels_after`, `changes`, `strokes` |
| `export` | blinded `code` |
| `target`, `package_saved` | prepare mode (`sessions/_prepare.jsonl`): source, seed, prompt, landmark confidence |

`logs/sketchfade.log` has the same in readable form plus every request and
the full traceback of any server error:

```
INFO  stroke layer=block-in part=head err=2.3px n=61 level=L3 corrected task=practice
INFO  probe layer=block-in head:s(e=0.008,c=1.00) eyes:n(e=0.031,c=0.74) ... | head=L3 ...
INFO  level head: L3 -> L2
```

## API

All `POST`, JSON, points in 480×480 canvas space; see the docstring of
`server.py`. Every state carries the overlay (`ghost`), the side reference
(`ref`), the per-part levels and streaks, the task and whether help is on.

## Files

| File | Role |
|---|---|
| `generate.py` | B: sketch + prompt → hidden target (SD + ControlNet + LCM) |
| `ghost.py` | B: landmarks + confidence check, 4 labeled construction layers |
| `package.py` | B: frozen guide packages, expert review CLI |
| `prepare.py` | B: batch-generate and validate study targets on the GPU |
| `fading.py` | A: probe scoring (two-way error + coverage), levels L3–L0, controller |
| `server.py` | HTTP API, sessions, tasks, probes, blinded export, study logger |
| `static/index.html` | Drawing UI |

## Simplifications (next steps)

- Landmarks come from edge profiles of the target, not a trained face-landmark
  model; the confidence check is a heuristic to calibrate in the pilot.
- Layers come from simple image processing (edges, blurred tone), not SAM.
- Faces only (the layout comes from the face database).
- One session at a time per server.
