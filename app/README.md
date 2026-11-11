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
works: generation takes about 10 s per seed on a CPU, and without
`diffusers`/`peft` the app uses the ShadowDraw retrieval face as target (no
shading or shadow layer).

**Device.** `generate.device()` picks a CUDA GPU when it has at least
`MIN_VRAM_GB` (4 GB) and the installed torch build can run on it, else Apple
MPS, else the CPU; the startup log says which and why. On Windows, `run.bat`
replaces a CPU-only torch with the CUDA build (`cu130`) once when it finds
`nvidia-smi`.

## 1 · Prepare targets (B, before recruitment)

Each step of the proposal's pipeline, and where it lives:

| Step | Code | What happens |
|---|---|---|
| Input | UI *Prepare targets* or `prepare.py` | rough sketch + prompt + drawing style (`pencil` by default, `realistic` photo, `anime`) + recorded seed; the sketch strokes are logged (`pin` event) |
| Target generation | `generate.py` | SD 1.5 + ControlNet scribble + LCM-LoRA, 4 steps, pretrained only; the style only changes the prompt. The sketch sets the layout only (ControlNet 0.5 for the first half of the steps, CFG 1.5 + negative prompt): at full strength the head outline became a hood or mask around the face |
| Target validation | `ghost.generated_layers` | MediaPipe face landmarks (478 points) of the generated image; per part, the share of its landmark outline lying on an image edge; no face or weakest < `CONF_MIN` → next seed (up to 3 in the UI) |
| Guide construction | `ghost.generated_layers` | The figure (face + hair, neck + shoulders) comes from MediaPipe's multiclass selfie segmenter (16 MB); when it misses a stylized drawing, from what differs from the background, split at the chin. `block-in` (cranium circle over the skull and hair, face ellipse, eyes, lips, nose wedge, neck and shoulders as straight lines) → `skeleton` (axis from the top of the skull to the base of the neck, eye/nose/mouth lines and the shoulder line, along the face's tilt) → `contours` (the image's own edges around the figure) → `shading` (mid tones) → `shadow` (darkest tones; dark hair is one mass), labeled by part: `head` (with hair), `eyes`, `nose`, `mouth`, `neck` (with shoulders) |
| Expert review | UI *Prepare targets → Expert review* or `package.py review` | illustrator checks order and difficulty; approve or reject (with the reviewer's name) |
| Frozen package | `packages/<id>/` | `layers.npz`, `target.png`, `guides.png` (reviewer sheet), `meta.json` |

Only **approved** packages can be opened in a study session, and a package is
never regenerated. Use one package for practice and baseline/post/retention,
and an **unseen** package for the transfer test.

**Calibrate `CONF_MIN` on the GPU.** The current value (0.20) was set on
ShadowDraw database faces, where it rejects about 12 %. Generated targets use
the landmark-on-edge confidence above (a few test targets scored 0.5–0.99 per
part), so run `python prepare.py --dry-run ...` on the GPU server during the
pilot, look at the printed confidence distribution and at a few `guides.png`
sheets, and adjust `CONF_MIN` in `ghost.py`. Packages saved before the
`shadow` layer existed (4 layers) cannot be opened; prepare them again.

The ShadowDraw shadow under the rough sketch (a blend of candidate faces) is
off for now (`SHADOWDRAW` in `static/index.html`).

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

The **Layers panel** beside the canvas (after Chen et al. 2025, *Interactive
Drawing Guidance for Anime Illustrations with Diffusion Model*, which puts
guidance sketches beside the canvas and lets the user place one on the
canvas background) shows every construction layer as a thumbnail. Ticking a
layer shows it on the canvas, unticking hides it; *Show all* and *Clear
background* do all at once. Visible layers stack, each in its own color, and
each is drawn at the part's help level above (so L1/L0 parts stay hidden).
Moving to a layer shows it; L3 correction only pulls toward the current layer
while it is shown. Every change is logged (`visible` event).

In probes and tests the panel is replaced by the **side reference**, which
shows the step to draw; it is the drawing prompt, not an overlay.

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
- filled layers (shading, shadow) are judged by their outline.

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
| `visible` | `visible` (layers shown on the canvas) |
| `stroke`, `probe_stroke` | `part`, `level`, `err_px`, `helped`, `corrected`, `w`, `raw` points, `shown` (corrected) points, `visible` |
| `check` | `reveals` |
| `probe_start`, `probe` | `probe_layer`, per-part `results` (`err`, `cov`, `outcome`), `levels_before`, `levels_after`, `changes`, `strokes` |
| `export` | blinded `code` |
| `pin`, `target`, `package_saved` | prepare mode (`sessions/_prepare.jsonl`): rough sketch `strokes` (points in 480 space, pen width), prompt, style, source, seed, landmark confidence |

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
| `generate.py` | B: sketch + prompt + style → hidden target (SD + ControlNet + LCM), device choice |
| `ghost.py` | B: face landmarks + confidence check, 5 labeled construction layers |
| `package.py` | B: frozen guide packages, expert review CLI |
| `prepare.py` | B: batch-generate and validate study targets on the GPU |
| `fading.py` | A: probe scoring (two-way error + coverage), levels L3–L0, controller |
| `server.py` | HTTP API, sessions, tasks, probes, blinded export, study logger |
| `static/index.html` | Drawing UI |

## Simplifications (next steps)

- Retrieval targets (no generator) still use landmarks from edge profiles; the
  confidence check is a heuristic to calibrate in the pilot.
- Contours and tone layers come from simple image processing (edges, blurred
  tone inside the segmented figure), not SAM. The neck part is not validated
  (a target may be cropped at the chin).
- Faces only (the layout comes from the face database).
- One session at a time per server.
