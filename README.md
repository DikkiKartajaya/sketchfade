<div align="center">

# ✏️ SketchFade

**Adaptive Fading of Generated Construction Guides for Novice Drawing**

AI as a drawing teacher, not a drawing replacement: SketchFade generates
*the steps, not the picture*, and steps back as you improve.

![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20%7C%20CPU-EE4C2C?logo=pytorch&logoColor=white)
![Diffusers](https://img.shields.io/badge/%F0%9F%A4%97%20diffusers-SD%201.5%20%2B%20ControlNet%20%2B%20LCM-yellow)
![Status](https://img.shields.io/badge/status-research%20prototype-8a8172)

<img src="docs/pipeline.png" alt="Rough sketch and prompt, hidden generated target, then block-in, skeleton, contours and shading layers" width="860">

</div>

---

## Why

AI drawing tools such as **ShadowDraw**, **PortraitSketch**, **DeepFaceDrawing** and
**AniFaceDrawing** guide, correct or finish a novice's drawing, and the result
looks better *now*. But performance during support can differ from later
retention or transfer: the AI drew it, the user only steered. Scaffolding theory
suggests adapting support, then fading it and handing over responsibility.

> **How can AI help novices draw better now, and still draw well without it later?**

| Work | Provides | Fades per part | Tested without help |
|---|---|:---:|---|
| ShadowDraw (2011) | Live contour shadow | No | No |
| PortraitSketch (2014) | Stroke correction | No | No |
| Drawing Assistant (2013) | Guides from photos | No | Partly |
| DrawMyPhoto (2019) | Stages and feedback | No | Left as future work |
| AniFaceDrawing (2023) | Generated portraits | No | No |
| **SketchFade** | **Generated guides as steps** | **Yes** | **Delayed test and transfer** |

**Gap:** little evidence on whether per-part fading adds value when the guide
content is the same, tested with delayed, unassisted drawing.

## Research questions

- **RQ1** Does adaptive fading improve drawing *without AI* compared with fixed guidance?
- **RQ2** How do users experience adaptive fading: ownership, control, usefulness and workload?

**Scope:** adult beginners, simplified frontal portraits, stylus tablet.

SketchFade combines two ideas:

| | Idea | Decides | How |
|---|---|---|---|
| **B** | **Construction Ghost** | *what* to show | Generate a hidden target, then reveal only the **next construction step**. |
| **A** | **Fading Shadow** | *how much* to show | Reduce help **for each face part** as *unassisted* performance improves. |

## Framework

<p align="center"><img src="docs/architecture.png" alt="Architecture: B prepares reviewed guide packages before recruitment; the tablet interface shows help per part; A estimates performance from unassisted probes and sets the level per part; a study logger records all events" width="860"></p>

### B · Construction Ghost (prepared before recruitment)

1. **Input:** a rough sketch, a text prompt and a recorded seed.
2. **Target generation:** Stable Diffusion 1.5 + ControlNet *scribble* + LCM-LoRA
   (4 steps, pretrained only, no new training). The target is **never shown**.
3. **Target validation:** face landmarks (eye line and eye centers, nose, mouth,
   head outline) with a confidence each; a target whose weakest landmark is
   below the threshold is rejected and the next seed is tried.
4. **Guide construction:** part masks (`head`, `eyes`, `nose`, `mouth`) and four
   layers in the order artists build a portrait: `block-in` → `skeleton` (axes)
   → `contours` → `shading`, placed on the detected landmarks.

   <p align="center"><img src="docs/targets.png" alt="Two hidden targets generated from prompts" width="520"></p>

5. **Expert review:** an illustrator checks the layer order and difficulty, then
   the **guide package is frozen**, so every participant sees the same guides.

### A · Fading Shadow (adaptive group only)

Help is set **per part**, on four levels:

| Level | Support for that part |
|---|---|
| **L3 Strong** | guide overlay at 0.60 opacity + stroke correction |
| **L2 Reduced** | guide overlay at 0.25 opacity |
| **L1 On request** | nothing, until the user presses **Check** (brief reveal) |
| **L0 Independent** | side reference only |

An assisted trace does not show learning, so **an unassisted redraw decides the
help**. At fixed checkpoints (the end of each layer, same in both groups) the
user redraws the step in a blank area with no help. The **performance
estimator** scores each part with a two-way distance error and coverage:

- **Success:** error ≤ 0.03 and coverage ≥ 0.80
- **Failure:** error ≥ 0.06 or coverage < 0.50
- **Two successes** in a row lower that part's level; **two failures** raise it.

Raw strokes are always kept apart from corrected strokes, and every event is
logged with participant, task, target, part and assistance level.

## User study

| Group | Help during practice |
|---|---|
| **Fixed guidance** | L3 stays available for every part |
| **Adaptive fading** | level changes per part from the probe results |

**Participants:** about 12 adult novices in 6 baseline-matched pairs, after a
3-user pilot. **Kept equal:** guides, interface, practice time, probe schedule.

| Day 0 · 50 min | Day 2 · 45 min | Day 7 · 35 min |
|---|---|---|
| Consent, stylus practice | Practice block 2 | Delayed retention test |
| Unassisted baseline | Immediate post-test | Unseen portrait transfer |
| Practice block 1 | NASA-TLX, ownership | Interview |

Formal tests disable overlays, snapping, Check and feedback in both groups.

### Evaluation

| Outcome | Measure | Role |
|---|---|---|
| Retention | delayed minus baseline | Primary, RQ1 |
| Immediate gain | post-test minus baseline | Secondary |
| Near transfer | unseen portrait score | Secondary |
| Ownership | items and interview | Exploratory, RQ2 |
| Workload | NASA-TLX | Exploratory, RQ2 |
| Policy behavior | levels, transitions | Feasibility |

Test drawings are exported under blinded codes and scored by **two blinded
raters** on four criteria (1–5), with ICC for reliability. Analysis: paired gain
difference, exact randomization test, bootstrap 95 % interval. With six pairs this
is a feasibility study: we report estimates and individual trajectories, and a
null result would not show equal effectiveness.

## Quick start

The study runs on a GPU server (targets in about 1 s); the tablet's browser
reaches it through an SSH tunnel, see [`app/README.md`](app/README.md#run-on-a-gpu-server-eg-rtx-6000-ada).

```bash
git clone https://github.com/DikkiKartajaya/sketchfade.git
cd sketchfade/app
pip install -r requirements.txt     # on the GPU server: install the CUDA build of torch first
python generate.py                  # once: downloads the models (~3 GB), test target

# before recruitment: precompute, check and review the study targets
python prepare.py "a young woman with short hair" "an old man with a beard" --seeds 6
python package.py list
python package.py review <id> --approve --reviewer "Illustrator name"

python server.py                    # → http://127.0.0.1:9100/
```

| Hardware | Time per target |
|---|---|
| NVIDIA GPU (e.g. RTX 6000 Ada) | ~1 s |
| CPU (Intel Core Ultra 5, measured) | ~46 s |

Without `diffusers` the app still runs on the ShadowDraw database face as the
target (no shading layer), which is enough to try the interface.

## Using the app

| Control | What it does |
|---|---|
| **Participant id**, **group**, **package** + **Start / resume** | open a study session; a returning participant keeps their group and levels |
| **Task** | `practice`, or a formal test: `baseline`, `post`, `retention`, `transfer` |
| **← / →** | previous layer / end this layer with an unassisted probe, then go on |
| **Probe this layer** | run a probe now (e.g. after the last layer) |
| **Check** | brief reveal of the current step for parts at L1 |
| **Finish test drawing** | export the test drawing under a blinded code |
| **Prepare targets** | sketch + prompt → generate → landmark check → **Save package** |

Study data stays on the study machine (ignored by git):

| Path | Contents |
|---|---|
| `app/sessions/<participant>.jsonl` | every event: strokes (raw and corrected), probes with per-part error, coverage and outcome, level changes, Check, tasks |
| `app/participants/<participant>.json` | group and current level and streak per part |
| `app/exports/<code>.png`, `key.csv` | blinded test drawings for the raters, and the key |
| `app/packages/<id>/` | frozen guide packages (layers, reviewer sheet, metadata, review) |
| `app/logs/sketchfade.log` | readable server log with full tracebacks |

## Repository

```
sketchfade/
├── app/                    SketchFade (A + B)
│   ├── generate.py         B · sketch + prompt → hidden target (SD + ControlNet + LCM)
│   ├── ghost.py            B · landmarks + confidence, 4 labeled construction layers
│   ├── package.py          B · frozen guide packages and expert review (CLI)
│   ├── prepare.py          B · batch-generate study targets on the GPU server
│   ├── fading.py           A · probe scoring, levels L3–L0, fading controller
│   ├── server.py           HTTP API, sessions, tasks, probes, study logger
│   └── static/index.html   drawing UI
├── shadowdraw-demo/demo/   ShadowDraw reimplementation + face database (retrieval, layout)
└── docs/                   README figures
```

## Schedule and roles

| Week 1 | Week 2 | Week 3 | Week 4 |
|---|---|---|---|
| Integrate interface, guides, logging | Guide review, 3-user pilot, calibration | Recruit; run the first two sessions | Day 7, ratings, analysis, report |

| Member | Lead role |
|---|---|
| Berlian Angga Kusuma | Tablet interface and interaction flow |
| Dikki Yusuf Kartajaya | Construction rules, fading integration, study procedure, analysis |
| Muhammad Ariful Furqon | Generation, stroke-error computation, backend records |

All members review the pilot, inspect data quality and discuss interpretation.
**Collaborator needed:** an illustrator or drawing instructor for guide review and rating.

## Built on

- Y. J. Lee, C. L. Zitnick, M. F. Cohen. **ShadowDraw**: Real-Time User Guidance for Freehand Drawing. SIGGRAPH 2011.
- J. Xie et al. **PortraitSketch**: Face Sketching Assistance for Novices. UIST 2014.
- E. Iarussi, A. Bousseau, T. Tsandilas. **The Drawing Assistant**. UIST 2013.
- **DrawMyPhoto**. 2019.
- S.-Y. Chen et al. **DeepFaceDrawing**: Deep Generation of Face Images from Sketches. SIGGRAPH 2020.
- Z. Huang et al. **AniFaceDrawing**: Anime Portrait Exploration during Your Sketching. SIGGRAPH 2023.
- D. Dixon, M. Prasad, T. Hammond. **iCanDraw?** CHI 2010.
- L. Zhang, A. Rao, M. Agrawala. **ControlNet**. ICCV 2023 · S. Luo et al. **LCM-LoRA**. 2023 · R. Rombach et al. **Stable Diffusion**. CVPR 2022.

## Team

Course project for *I665E Human–AI Interaction Exercise II*, JAIST.

**Berlian Angga Kusuma** (s2610055) · **Dikki Yusuf Kartajaya** (s2610043) · **Muhammad Ariful Furqon** (s2620005)
