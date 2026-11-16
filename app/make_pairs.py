"""Side-by-side images: each final drawing next to its reference.

Run after a session (the app does not need to be running, and nothing in it changes):

  python make_pairs.py                    # every participant
  python make_pairs.py --participant P01  # one participant

For every drawing in exports/<group>/final/ that is listed in exports/index.csv (the
formal tests baseline / post / retention / transfer, and the combined final of each
practice pass) it writes

  exports/<group>/compare/<same name as the drawing>_compare.png
      left: the drawing, right: its reference, with a caption

and the reference of each portrait once, as exports/references/<package>_reference.png.

The reference is the one the formal tests show beside the canvas: the portrait's
contours layer as grey lines (server.py, TEST_REF_LAYER). Existing compare images are
rewritten, so running the script again is safe. Drawings saved before index.csv
existed (random names in exports/) are skipped.
"""

import argparse
import csv
import os

import numpy as np
from PIL import Image, ImageDraw

import fading
import ghost

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 480                                  # saved drawings are 480 x 480
REF_LAYER = ghost.LAYERS.index("contours")  # same as server.TEST_REF_LAYER
CAPTION = 30                                # caption strip height (px)


def reference(packages, exports, target):
    """The portrait's reference (480 x 480, grey lines on white); saved once per portrait."""
    path = os.path.join(exports, "references", f"{target}_reference.png")
    if not os.path.exists(path):
        with np.load(os.path.join(packages, target, "layers.npz")) as z:
            layers = z["layers"]
        ref = fading.guide_mask(layers[REF_LAYER]) > 0
        img = Image.fromarray(np.where(ref, 70, 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path)
    return Image.open(path).convert("L")


def pair(drawing, ref, caption):
    """drawing | reference, each with a label, a caption on top."""
    gap = 16
    img = Image.new("L", (2 * SIZE + gap, SIZE + 2 * CAPTION), 255)
    d = ImageDraw.Draw(img)
    d.text((8, 9), caption, fill=0)
    for x, (part, label) in enumerate([(drawing, "drawing"), (ref, "reference")]):
        left = x * (SIZE + gap)
        img.paste(part.convert("L").resize((SIZE, SIZE)), (left, CAPTION))
        d.rectangle([left, CAPTION, left + SIZE - 1, CAPTION + SIZE - 1], outline=170)
        d.text((left + 8, CAPTION + SIZE + 9), label, fill=90)
    return img


def make_pairs(app_dir=HERE, participant=None):
    """Write the compare images; returns their paths relative to exports/."""
    exports = os.path.join(app_dir, "exports")
    packages = os.path.join(app_dir, "packages")
    index = os.path.join(exports, "index.csv")
    if not os.path.exists(index):
        print("no exports/index.csv yet: nothing to compare")
        return []
    with open(index, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["stage"] not in ghost.LAYERS                       # final/ only
                and (participant is None or r["participant"] == participant)]
    done = []
    for r in rows:
        src = os.path.join(exports, r["file"])
        if not os.path.exists(src):
            print(f"skipped {r['file']}: file not found")
            continue
        try:
            ref = reference(packages, exports, r["target"])
        except OSError:
            print(f"skipped {r['file']}: portrait {r['target']!r} not found in packages/")
            continue
        kind = f"practice pass {r['pass']}, all steps" if r["kind"] == "practice" else f"{r['stage']} test"
        caption = f"{r['participant']} | {r['group']} | {kind} | portrait {r['target']} | {r['time']}"
        name = os.path.basename(r["file"])[:-4] + "_compare.png"
        rel = f"{r['group']}/compare/{name}"
        os.makedirs(os.path.join(exports, r["group"], "compare"), exist_ok=True)
        pair(Image.open(src), ref, caption).save(os.path.join(exports, rel))
        done.append(rel)
    print(f"{len(done)} compare image(s) in exports/<group>/compare/")
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participant", help="only this participant ID")
    make_pairs(participant=ap.parse_args().participant)
