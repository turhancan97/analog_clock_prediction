#!/usr/bin/env python3
"""Auto-surface likely dataset defects (and genuinely hard images) by their
confidence signature, instead of hand-diagnosing per class the way the two
known defects were originally found.

Two flags, one per known defect shape (see DATASET.md / CHANGELOG.md):

  low-conf         p(top-1) < --low-conf
                   diffuse, unsure predictions -- the blank-dial defect
                   (p ~ 0.06-0.10) and genuinely ambiguous renders.

  confident-wrong  wrong  AND  p(top-1) > --high-conf  AND  |err| >= --min-err
                   the model is sure and sure-wrong -- the +/-3h15m hand-shift
                   defect, or a mislabelled folder.

Confident-wrong predictions that share the same *signed* minute offset are a
rendering-bug signature (e.g. a cluster all at +195 min): reported as a
histogram, with any offset of >= --cluster hits called out as systematic.

    python scripts/flag_suspects.py --split test
    python scripts/flag_suspects.py --split all --csv suspects.csv

Works with either head (softmax or the circular regression head); low-conf is
skipped for the regression head, which has no probability.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import clockmodel as cm


def _minute(label):
    """'10-35' -> minutes past 12 on a 12-hour dial."""
    h, m = label.split("-")
    return (int(h) % 12) * 60 + int(m)


def _listing(split):
    """(path, class_name) for every image in the split(s), in class_names() order."""
    names = cm.class_names()
    splits = ("train", "valid", "test") if split == "all" else (split,)
    rows = []
    for sp in splits:
        for cls in names:
            for p in sorted((cm.DATA_DIR / sp / cls).glob("*.jpg")):
                rows.append((str(p), cls))
    return rows


def _decode(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(tf.cast(img, tf.float32), cm.IMG_SIZE)
    img.set_shape((*cm.IMG_SIZE, 3))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test",
                    choices=["train", "valid", "test", "all"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--low-conf", type=float, default=0.15)
    ap.add_argument("--high-conf", type=float, default=0.55)
    ap.add_argument("--min-err", type=int, default=15,
                    help="minutes; a confident-wrong flag must be at least this far off")
    ap.add_argument("--cluster", type=int, default=3,
                    help="confident-wrong offset seen >= this many times = systematic")
    ap.add_argument("--limit", type=int, default=40, help="rows printed per section")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    names = cm.class_names()
    model = cm.load_model(args.model) if args.model else cm.load_model()

    rows = _listing(args.split)
    if not rows:
        sys.exit(f"no images found for split={args.split}")
    paths = [p for p, _ in rows]
    y_true = np.array([names.index(c) for _, c in rows])
    print(f"scoring {len(paths)} images (split={args.split})")

    ds = (tf.data.Dataset.from_tensor_slices(paths)
          .map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(64).prefetch(tf.data.AUTOTUNE))
    preds = model.predict(ds, verbose=1)
    y_pred = cm.output_to_class_idx(preds)
    is_softmax = preds.shape[-1] == cm.NUM_CLASSES
    conf = (preds[np.arange(len(preds)), y_pred] if is_softmax
            else np.full(len(y_pred), np.nan))

    tmin = np.array([_minute(names[i]) for i in y_true])
    pmin = np.array([_minute(names[i]) for i in y_pred])
    signed = (pmin - tmin + 360) % 720 - 360                 # [-360, 360)
    abserr = np.minimum(np.abs(pmin - tmin), 720 - np.abs(pmin - tmin))
    wrong = y_pred != y_true

    low = (np.flatnonzero(conf < args.low_conf) if is_softmax
           else np.array([], dtype=int))
    cwrong = np.flatnonzero(
        wrong & (abserr >= args.min_err)
        & (conf > args.high_conf if is_softmax else True))

    acc = 1.0 - wrong.mean()
    print(f"\ntop-1 accuracy : {acc:.4f}   ({wrong.sum()} wrong of {len(rows)})")

    def _dump(idx, title):
        print(f"\n{'=' * 72}\n{title}  ({len(idx)})\n{'=' * 72}")
        order = idx[np.argsort(conf[idx])] if is_softmax else \
            idx[np.argsort(-abserr[idx])]
        for i in order[:args.limit]:
            c = f"p={conf[i]:.3f}  " if is_softmax else ""
            mark = " " if wrong[i] else "*"   # * = prediction actually matches the label
            print(f"  {mark} {cm.label_to_time(names[y_true[i]]):>5} -> "
                  f"{cm.label_to_time(names[y_pred[i]]):>5}  {c}"
                  f"{abserr[i]:>3.0f} min  {Path(paths[i]).relative_to(cm.DATA_DIR)}")
        if len(idx) > args.limit:
            print(f"  ... and {len(idx) - args.limit} more")

    if is_softmax:
        _dump(low, f"LOW CONFIDENCE  (p < {args.low_conf})  -- '*' = still correct")
    else:
        print("\n(low-confidence flag skipped: regression head has no probability)")
    _dump(cwrong, f"CONFIDENT & WRONG  (|err| >= {args.min_err} min"
                  + (f", p > {args.high_conf}" if is_softmax else "") + ")")

    if len(cwrong):
        hist = Counter(int(signed[i]) for i in cwrong)
        print(f"\n{'=' * 72}\nconfident-wrong by signed offset (minutes)\n{'=' * 72}")
        for off, n in sorted(hist.items(), key=lambda kv: -kv[1]):
            tag = "  <-- systematic (rendering-defect signature)" if n >= args.cluster else ""
            print(f"  {off:+5d} min : {n:3d}{tag}")

    flagged = sorted(set(low.tolist()) | set(cwrong.tolist()))
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "true", "pred", "conf", "abs_err_min",
                        "signed_off_min", "flag"])
            for i in flagged:
                flags = []
                if i in set(low.tolist()):
                    flags.append("low-conf")
                if i in set(cwrong.tolist()):
                    flags.append("confident-wrong")
                w.writerow([
                    Path(paths[i]).relative_to(cm.DATA_DIR), names[y_true[i]],
                    names[y_pred[i]],
                    f"{conf[i]:.4f}" if is_softmax else "",
                    int(abserr[i]), int(signed[i]), "+".join(flags)])
        print(f"\nwrote {len(flagged)} flagged rows to {args.csv}")

    print(f"\n{len(flagged)} images flagged "
          f"({len(low)} low-conf, {len(cwrong)} confident-wrong, "
          f"{len(set(low.tolist()) & set(cwrong.tolist()))} both)")


if __name__ == "__main__":
    main()
