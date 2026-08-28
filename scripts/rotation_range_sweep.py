#!/usr/bin/env python3
"""Does a wider training-time RandomRotation extend the robust-angle plateau?
(README.md future work: "Widen the rotation-augmentation range".)

The brittleness characterization (scripts/characterize_rotation_brittleness.py,
CHANGELOG 2026-08-24) found the plateau's edges track the *literal* training
augmentation range: the default checkpoint (+/-10.8 deg aug) holds accuracy to
roughly +/-11 deg then falls off a cliff; the no-aug released checkpoint falls
off almost immediately. This script trains nothing -- it takes a set of
checkpoints trained at different `--rotation-factor` values and sweeps each one
over rotation angle, so we can see whether the plateau edge moves out with the
training range or just gets a gentler slope.

    python scripts/rotation_range_sweep.py \
        models/clock_rot_f0.03_s0.keras models/clock_rot_f0.06_s0.keras ...

Checkpoint filenames of the form ...f<factor>... are auto-labelled with the
+/- degree range (factor * 360); otherwise the file stem is used.
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import clockmodel as cm
from characterize_rotation_brittleness import ANGLES, sweep

PLATEAU_ACC = 0.95  # "robust" = top-1 at or above this


def plateau_range(results):
    """Widest contiguous angle interval containing 0 with acc >= PLATEAU_ACC."""
    lo = hi = 0
    for a in sorted(a for a in ANGLES if a <= 0)[::-1]:
        if results[a][0] >= PLATEAU_ACC:
            lo = a
        else:
            break
    for a in sorted(a for a in ANGLES if a >= 0):
        if results[a][0] >= PLATEAU_ACC:
            hi = a
        else:
            break
    return lo, hi


def label_for(path):
    m = re.search(r"f(\d*\.?\d+)", Path(path).stem)
    if m:
        deg = float(m.group(1)) * 360.0
        return f"+/-{deg:.1f} deg aug"
    return Path(path).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--per-class", type=int, default=1,
                    help="test images sampled per class (default 1 -> 144)")
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "images"
                                        / "rotation_range_sweep.png"))
    args = ap.parse_args()

    names = cm.class_names()
    rng = np.random.default_rng(0)
    images, labels = [], []
    for cls in names:
        paths = sorted((cm.DATA_DIR / "test" / cls).glob("*.jpg"))
        for p in rng.choice(paths, size=min(args.per_class, len(paths)),
                            replace=False):
            images.append(cm.load_image(p))
            labels.append(cls)
    print(f"sampled {len(images)} test images across {len(names)} classes\n")

    fig, ax = plt.subplots(figsize=(9, 5))
    summary = []
    for ckpt in args.checkpoints:
        lbl = label_for(ckpt)
        print(f"--- {ckpt}  ({lbl}) ---")
        model = cm.load_model(ckpt)
        results = sweep(model, names, images, labels)
        lo, hi = plateau_range(results)
        summary.append((lbl, ckpt, results[0][0], lo, hi))
        xs = sorted(results)
        ax.plot(xs, [results[a][0] * 100 for a in xs], marker="o",
                markersize=3, label=lbl)
        print()

    ax.axhline(PLATEAU_ACC * 100, color="grey", ls="--", lw=0.8,
               label=f"{PLATEAU_ACC:.0%} plateau threshold")
    ax.set_xlabel("test-time rotation (degrees)")
    ax.set_ylabel("top-1 accuracy (%)")
    ax.set_title(f"Does wider rotation aug widen the robust plateau? "
                 f"({len(images)} test images)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print(f"wrote {args.out}\n")

    print(f"{'training aug':>18} | {'upright acc':>11} | "
          f"{'>=95% plateau':>14} | plateau width")
    print("-" * 70)
    for lbl, ckpt, up, lo, hi in summary:
        print(f"{lbl:>18} | {up:>11.4f} | "
              f"{f'[{lo:+d}, {hi:+d}] deg':>14} | {hi - lo} deg")


if __name__ == "__main__":
    main()
