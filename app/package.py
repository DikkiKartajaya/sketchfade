"""B · frozen guide packages: targets prepared and reviewed before recruitment.

A package is one validated target with its construction layers, saved once and
never regenerated, so every participant in both groups sees the same guides.

  packages/<id>/layers.npz   (5, 300, 300) int8 label images
  packages/<id>/target.png   the hidden target (for the reviewer, never shown)
  packages/<id>/guides.png   all five layers side by side (for the reviewer)
  packages/<id>/meta.json    prompt, style, seed, source, landmark confidence, review

Expert review (an illustrator checks layer order and difficulty):

  python package.py list
  python package.py review <id> --approve --reviewer "Name" [--note "..."]
  python package.py review <id> --reject  --reviewer "Name" --note "why"

Only approved packages can be used in a study session.
"""

import argparse
import json
import os
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "packages")
COLORS = [(40, 60, 140), (200, 90, 40), (40, 140, 90), (150, 50, 140), (150, 120, 30)]  # per part


def _dir(pid):
    if not pid or os.sep in pid or "/" in pid or pid.startswith("."):
        raise ValueError(f"bad package id {pid!r}")
    return os.path.join(ROOT, pid)


def _sheet(layers):
    """Reviewer sheet: the layers left to right, colored by part."""
    n, h, w = layers.shape
    img = np.full((h, w * n, 3), 247, np.uint8)
    for k in range(n):
        for p, c in enumerate(COLORS):
            img[:, k * w : (k + 1) * w][layers[k] == p + 1] = c
    return Image.fromarray(img)


def save(layers, gray, meta):
    """Freeze one target. Returns the new package id."""
    pid = time.strftime("%Y%m%d_%H%M%S") + f"_s{meta.get('seed', 0)}"
    d = _dir(pid)
    os.makedirs(d)
    np.savez_compressed(os.path.join(d, "layers.npz"), layers=layers)
    if gray is not None:
        Image.fromarray((gray * 255).astype(np.uint8)).save(os.path.join(d, "target.png"))
    _sheet(layers).save(os.path.join(d, "guides.png"))
    meta = {**meta, "id": pid, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "review": {"status": "pending"}}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return pid


def meta(pid):
    with open(os.path.join(_dir(pid), "meta.json")) as f:
        return json.load(f)


def load(pid):
    """(layers, meta) of one package."""
    with np.load(os.path.join(_dir(pid), "layers.npz")) as z:
        return z["layers"], meta(pid)


def list_all():
    if not os.path.isdir(ROOT):
        return []
    out = []
    for pid in sorted(os.listdir(ROOT)):
        try:
            out.append(meta(pid))
        except (OSError, ValueError):
            continue
    return out


def review(pid, approve, reviewer, note=""):
    m = meta(pid)
    m["review"] = {"status": "approved" if approve else "rejected",
                   "reviewer": reviewer, "note": note,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(_dir(pid), "meta.json"), "w") as f:
        json.dump(m, f, indent=1)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list packages and their review status")
    r = sub.add_parser("review", help="record the expert review of a package")
    r.add_argument("id")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--approve", action="store_true")
    g.add_argument("--reject", action="store_true")
    r.add_argument("--reviewer", required=True)
    r.add_argument("--note", default="")
    a = ap.parse_args()
    if a.cmd == "list":
        for m in list_all():
            conf = min(m.get("confidence", {"-": 0}).values())
            print(f"{m['id']:28} {m['review']['status']:9} conf={conf:.2f} "
                  f"{m.get('source', '?'):9} {m.get('prompt', '')!r}")
    else:
        m = review(a.id, a.approve, a.reviewer, a.note)
        print(f"{a.id}: {m['review']['status']} by {a.reviewer}")


if __name__ == "__main__":
    main()
