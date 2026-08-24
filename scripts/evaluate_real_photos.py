#!/usr/bin/env python3
"""Evaluate the clock-reading model on real (non-synthetic) clock photos.

Investigates a new axis of difficulty vs. the synthetic training/eval set:
off-axis viewing angles, non-upright hands, imprecise time settings, varied
lighting. Motivated by the rotation-brittleness finding (AGENTS.md) -- real
photos are rarely perfectly upright the way every synthetic render is.

Dataset: kongaskristjan/real-clocks (CC0, https://pxhere.com), 92 real
clock photos with ground-truth time embedded in the filename
(`<hash>_<hour>_<minute>.jpg`). Not committed to the repo (gitignored, like
data/); download with:

    kaggle datasets download -d kongaskristjan/real-clocks -p real_data --unzip

The model's 144 output classes are 5-minute increments, but real photos'
true times aren't restricted to multiples of 5 -- so "accuracy" here means
two different things, both reported: exact match against the label rounded
to the nearest 5-minute class (the model's actual task), and circular
minutes-off error against the *unrounded* true time (how close the reading
actually is, independent of the class grid).
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

FILENAME_RE = re.compile(r"^[0-9a-f]+_(\d{2})_(\d{2})\.jpg$")


def parse_label(path):
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"can't parse hour_minute from {path.name}")
    hour, minute = int(m.group(1)), int(m.group(2))
    return hour, minute


def nearest_5min_label(hour, minute, names):
    rounded_minute = int(round(minute / 5)) * 5
    hour_adj = hour + rounded_minute // 60
    rounded_minute %= 60
    hour_label = hour_adj % 12
    if hour_label == 0:
        hour_label = 12
    label = f"{hour_label}-{rounded_minute:02d}"
    assert label in names, f"{label} not a known class (from {hour}:{minute:02d})"
    return label


def circular_minutes_off(hour, minute, pred_label):
    true_min = (hour % 12) * 60 + minute
    ph, pm = cm.label_to_time(pred_label).split(":")
    pred_min = (int(ph) % 12) * 60 + int(pm)
    diff = abs(pred_min - true_min)
    return min(diff, 720 - diff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "real_data"),
                     help="directory with train/ and val/ subdirs of real photos")
    ap.add_argument("--model", default=None)
    ap.add_argument("--figure", default=str(REPO_ROOT / "docs" / "images" / "real_photo_predictions.png"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    paths = sorted(data_dir.glob("*/*.jpg"))
    if not paths:
        sys.exit(f"no images found under {data_dir}/*/*.jpg -- download the dataset first (see module docstring)")

    names = cm.class_names()
    model = cm.load_model(args.model) if args.model else cm.load_model()

    records = []
    for path in paths:
        hour, minute = parse_label(path)
        img = cm.load_image(path)
        probs = model.predict(img[None, ...], verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = names[pred_idx]
        pred_prob = float(probs[pred_idx])
        top5_idx = np.argsort(probs)[-5:]
        true_rounded = nearest_5min_label(hour, minute, names)
        err = circular_minutes_off(hour, minute, pred_label)
        records.append(dict(
            path=path, hour=hour, minute=minute, true_rounded=true_rounded,
            pred_label=pred_label, pred_prob=pred_prob,
            top5_hit=names.index(true_rounded) in top5_idx, err=err,
        ))

    n = len(records)
    top1 = np.mean([r["pred_label"] == r["true_rounded"] for r in records])
    top5 = np.mean([r["top5_hit"] for r in records])
    errs = np.array([r["err"] for r in records])
    within5 = np.mean(errs <= 5)
    within30 = np.mean(errs <= 30)

    print(f"real photos       : {n} images (kongaskristjan/real-clocks)")
    print(f"top-1 accuracy     : {top1:.4f}  (vs. ~0.9958 on the synthetic test split)")
    print(f"top-5 accuracy     : {top5:.4f}")
    print(f"median |error|     : {np.median(errs):.0f} min")
    print(f"mean |error|       : {errs.mean():.1f} min")
    print(f"within 5 min       : {within5:.4f}")
    print(f"within 30 min      : {within30:.4f}")
    print(f"mean confidence    : {np.mean([r['pred_prob'] for r in records]):.3f}")

    wrong = [r for r in records if r["pred_label"] != r["true_rounded"]]
    print(f"\n{len(wrong)} misread(s) out of {n}:")
    for r in sorted(wrong, key=lambda r: -r["err"])[:20]:
        print(f"  {r['path'].name:30s} true ~{r['hour']}:{r['minute']:02d} "
              f"(class {r['true_rounded']})  ->  pred {cm.label_to_time(r['pred_label'])} "
              f"(p={r['pred_prob']:.3f}, {r['err']} min off)")

    # Qualitative figure: a mix of best and worst predictions.
    by_err = sorted(records, key=lambda r: r["err"])
    sample = by_err[:6] + by_err[-6:]
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for ax, r in zip(axes.flat, sample):
        img = cm.load_image(r["path"]) / 255.0
        ok = r["pred_label"] == r["true_rounded"]
        ax.imshow(img)
        ax.set_title(
            f"true ~{r['hour']}:{r['minute']:02d} -> pred {cm.label_to_time(r['pred_label'])} "
            f"({r['pred_prob']:.2f}, {r['err']} min off)",
            color="black" if ok else "crimson", fontsize=8,
        )
        ax.axis("off")
    plt.suptitle(f"Real-photo predictions: 6 best + 6 worst (of {n}, kongaskristjan/real-clocks)")
    plt.tight_layout()
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.figure, dpi=130)
    print(f"\nwrote {args.figure}")


if __name__ == "__main__":
    main()
