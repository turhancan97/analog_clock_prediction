#!/usr/bin/env python3
"""Summarise one checkpoint for the backbone ablation (CHANGELOG.md): param
count, test + dataset-wide top-1 accuracy, mean circular error in minutes, and
CPU inference latency. Run once per trained backbone and paste the rows into
the ablation table.

    python scripts/backbone_ablation.py models/clock_b0.keras
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import clockmodel as cm


def _circular_min_err(pred_idx, true_idx, names):
    def to_min(l):
        h, m = l.split("-")
        return (int(h) % 12) * 60 + int(m)

    d = np.abs([to_min(names[p]) - to_min(names[t])
                for p, t in zip(pred_idx, true_idx)])
    return np.minimum(d, 720 - d)


def _accuracy(model, splits, names):
    yt, yp = [], []
    for split in splits:
        ds = cm.make_dataset(split, batch_size=64)
        preds = model.predict(ds, verbose=0)
        yt.append(np.concatenate([y.numpy().argmax(1) for _, y in ds]))
        yp.append(cm.output_to_class_idx(preds))
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    err = _circular_min_err(yp, yt, names)
    return (yp == yt).mean(), err.mean(), len(yt)


def _cpu_latency(model, n=200):
    """ms per image, batch size 1, single image reused (measures compute, not
    the data pipeline). Forced onto CPU -- the deployment-relevant number."""
    x = np.random.uniform(0, 255, (1, *cm.IMG_SIZE, 3)).astype("float32")
    with tf.device("/CPU:0"):
        f = tf.function(lambda t: model(t, training=False))
        f(tf.constant(x))  # trace / warm up
        start = time.perf_counter()
        for _ in range(n):
            f(tf.constant(x))
        return (time.perf_counter() - start) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--no-latency", action="store_true")
    args = ap.parse_args()

    names = cm.class_names()
    model = cm.load_model(args.checkpoint)

    test_acc, test_err, n_test = _accuracy(model, ["test"], names)
    full_acc, full_err, n_full = _accuracy(model, ["train", "valid", "test"], names)
    lat = None if args.no_latency else _cpu_latency(model)

    print(f"\ncheckpoint      : {args.checkpoint}")
    print(f"params          : {model.count_params():,}")
    print(f"test top-1      : {test_acc:.4f}  ({n_test} images)")
    print(f"dataset top-1   : {full_acc:.4f}  ({n_full} images, {round((1-full_acc)*n_full)} errors)")
    print(f"mean |error|    : test {test_err:.1f} min / dataset {full_err:.1f} min")
    if lat is not None:
        print(f"CPU latency     : {lat:.1f} ms/image  (batch 1, /CPU:0)")

    print(f"\n| ckpt | params | test top-1 | dataset top-1 | mean err (min) | CPU ms |")
    print(f"| {Path(args.checkpoint).stem} | {model.count_params():,} | "
          f"{test_acc:.4f} | {full_acc:.4f} | {full_err:.1f} | "
          f"{'-' if lat is None else f'{lat:.1f}'} |")


if __name__ == "__main__":
    main()
