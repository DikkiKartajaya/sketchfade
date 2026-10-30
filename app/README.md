# SketchFade — Fading Shadow (A) + Construction Ghost (B)

Base code for proposals **A** and **B** combined: B decides *what* to show
(construction layers of a hidden target), A decides *how much* to show
(per-part help that fades as the user's skill grows).

Built on the ShadowDraw demo in `../shadowdraw-demo/demo` (its engine and face
database are imported, not copied).

## Run

```bash
pip install -r requirements.txt
python generate.py        # once: downloads SD 1.5 + ControlNet + LCM (~3 GB), test image
python server.py          # http://127.0.0.1:9100/
```

or double-click `run.bat`. Without `diffusers`/`peft` installed the app still
runs, using the ShadowDraw retrieval face as target (no shading layer).

## Run on a GPU server (e.g. RTX 6000 Ada)

The whole app runs on the server; the browser on the laptop reaches it through
an SSH tunnel (the server only listens on 127.0.0.1, nothing is exposed).

```bash
# laptop: copy the project (app + shadowdraw-demo, which holds the face database)
scp -r sketchfade user@gpu-server:~/

# server
cd ~/sketchfade/app
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build, see pytorch.org
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.get_device_name(0))"          # must print the GPU
python generate.py                     # downloads models once, test should take ~1-2 s
tmux new -s sketchfade                 # keeps running after you disconnect
CUDA_VISIBLE_DEVICES=0 python server.py 9100

# laptop: tunnel, then open http://127.0.0.1:9100/
ssh -N -L 9100:127.0.0.1:9100 user@gpu-server

# laptop: fetch the study data afterwards
scp -r user@gpu-server:~/sketchfade/app/{logs,sessions,targets,skill.json} .
```

## Flow

1. **Sketch** a rough face → live ShadowDraw shadow (paper's retrieval).
2. Type a **prompt** and press **Generate target** → slide 7's pipeline:
   the rough sketch (ControlNet scribble) + prompt drive Stable Diffusion 1.5
   with LCM (4 steps) to a face image, like DeepFaceDrawing / AniFaceDrawing,
   but the image stays **hidden** (saved to `targets/` for the study).
   **Regenerate** = new seed (the user stays in charge of the target).
   On a CPU this takes about a minute; on a CUDA GPU about a second.
3. Draw **layer by layer** (← / →): `block-in` → `skeleton` → `contours` →
   `shading`, all derived from the hidden target. Only the current layer
   appears as a ghost.
4. Every stroke is scored against the current layer (mean distance to the
   nearest guide pixel). The score updates the skill of the part it belongs to
   (`head`, `eyes`, `nose`, `mouth`), and each part's help level fades on its own:

   | skill      | help level                | what the user sees for that part        |
   |------------|---------------------------|-----------------------------------------|
   | < 0.4      | shadow + correction       | ghost of current layer, strokes snapped |
   | 0.4 – 0.6  | construction lines only   | faint block-in / skeleton               |
   | 0.6 – 0.8  | check on request          | nothing until **Check** is pressed      |
   | ≥ 0.8      | no help                   | nothing                                 |

5. Sliders: **Pen width** (1–12 px, per new stroke) and **AI help strength**
   (0–100 %, scales the shadow / ghost opacity and the stroke-correction pull).
   Both are logged per stroke (`w`, `aihelp`).
6. **Test mode** hides all help (pre/post test without AI); strokes are still
   scored and logged but do not change the skill.
7. **Save session** writes the stroke log to `sessions/*.json`. Skill is kept in
   `skill.json` across sessions (delete it to start a new participant).

## Run log

Everything the server does is printed to the console and appended to
`logs/sketchfade.log`: each request (status, ms), pin / target choice, every
stroke (layer, part, error, help level, skill), help-level changes, and the
full traceback of any server error (the UI then shows "server error").

```
INFO  stroke layer=block-in part=head err=2.3px n=61 help='shadow + correction' corrected | head=0.48 ...
INFO  help  head: 'shadow + correction' -> 'construction lines only'
ERROR POST /api/stroke failed
Traceback (most recent call last): ...
```

## Files

| File                | Role                                                        |
|---------------------|-------------------------------------------------------------|
| `generate.py`       | B: sketch + prompt → hidden target (SD + ControlNet + LCM)  |
| `ghost.py`          | B: target image → 4 labeled construction layers             |
| `fading.py`         | A: stroke error, skill per part, help-level policy, ghost   |
| `server.py`         | HTTP API + session state, uses `ShadowEngine` from the demo |
| `static/index.html` | Drawing UI                                                  |

## Simplifications (next steps)

- Layers come from simple image processing (edges, blurred tone), not SAM.
- Block-in / skeleton use fixed face proportions, placed where ShadowDraw
  aligns the sketch and snapped to the target's eye / nose / mouth edges.
- Faces only (the layout comes from the face database).
- Single user, single session in memory.
