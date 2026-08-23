#!/usr/bin/env python3
"""Evaluate the clock-reading model on a data split."""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
import clockmodel as cm


def minutes_off(pred_labels, true_labels):
    """Circular error in minutes on a 12-hour dial (max 360)."""
    def to_min(l):
        h, m = l.split("-")
        return (int(h) % 12) * 60 + int(m)

    diff = np.abs([to_min(p) - to_min(t) for p, t in zip(pred_labels, true_labels)])
    return np.minimum(diff, 720 - diff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    names = cm.class_names()
    model = cm.load_model(args.model) if args.model else cm.load_model()
    ds = cm.make_dataset(args.split, batch_size=args.batch_size)

    probs = model.predict(ds, verbose=1)
    y_true = np.concatenate([y.numpy().argmax(1) for _, y in ds])
    y_pred = probs.argmax(1)

    acc = (y_pred == y_true).mean()
    top5 = np.mean([t in row for row, t in zip(np.argsort(probs, 1)[:, -5:], y_true)])
    err = minutes_off([names[i] for i in y_pred], [names[i] for i in y_true])

    print(f"\nsplit          : {args.split}  ({len(y_true)} images)")
    print(f"top-1 accuracy : {acc:.4f}")
    print(f"top-5 accuracy : {top5:.4f}")
    print(f"median |error| : {np.median(err):.0f} min")
    print(f"mean |error|   : {err.mean():.1f} min")
    print(f"within 5 min   : {(err <= 5).mean():.4f}")

    wrong = np.flatnonzero(y_pred != y_true)
    if len(wrong):
        print(f"\n{len(wrong)} misread(s):")
        for i in wrong[:20]:
            print(f"  true {cm.label_to_time(names[y_true[i]]):>5}"
                  f"  ->  pred {cm.label_to_time(names[y_pred[i]]):>5}"
                  f"  (p={probs[i, y_pred[i]]:.3f}, {err[i]:.0f} min off)")
        if len(wrong) > 20:
            print(f"  ... and {len(wrong) - 20} more")


if __name__ == "__main__":
    main()
