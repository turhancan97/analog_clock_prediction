#!/usr/bin/env python3
"""Did fine-tuning on real photos change *where* the model looks, or just
recalibrate it? Grad-CAM on real held-out photos for two checkpoints:

  default   models/clock_model_rot54_s0.keras       (synthetic only)
  real-mix  models/clock_realism_realmix.keras      (+ 138 real training photos)

Outputs a qualitative panel (image | default CAM | real-mix CAM, with each
model's prediction) and a quantitative comparison of heatmap *focus* -- the
fraction of Grad-CAM mass in the hottest 10% of pixels -- across every real
test photo. A more focused heatmap means the model is committing to a region;
a diffuse one means it has not found the hands.

    python scripts/gradcam_real.py
    python scripts/gradcam_real.py --n 12 --split real-test
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import clockmodel as cm

LAYER = "top_activation"
MODELS = {
    "default": cm.MODELS_DIR / "clock_model_rot54_s0.keras",
    "real-mix": cm.MODELS_DIR / "clock_realism_realmix.keras",
}


def _rows(split):
    import pandas as pd
    df = pd.read_csv(cm.REAL_MANIFEST)
    want = split[len("real-"):]
    if want != "all":
        df = df[df["split"] == want]
    names = set(cm.class_names())
    return [(str(REPO_ROOT / r.path), r.label,
             f"{int(r.hour) % 12 or 12}:{int(r.minute):02d}")
            for r in df.itertuples() if r.label in names]


def _cam(grad_model, img):
    x = tf.convert_to_tensor(img[None, ...])
    with tf.GradientTape() as tape:
        conv, preds = grad_model(x)
        idx = int(tf.argmax(preds[0]))
        score = preds[:, idx]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), idx, float(preds[0][idx])


def _focus(cam):
    """Fraction of total heatmap mass in the hottest 10% of pixels."""
    v = np.sort(cam.ravel())[::-1]
    k = max(1, len(v) // 10)
    return v[:k].sum() / (v.sum() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="real-test",
                    choices=["real-test", "real-train", "real-all"])
    ap.add_argument("--n", type=int, default=9, help="photos in the qualitative panel")
    ap.add_argument("--figure", default=str(REPO_ROOT / "docs" / "images" / "gradcam_real.png"))
    args = ap.parse_args()

    names = cm.class_names()
    rows = _rows(args.split)
    print(f"{len(rows)} photos (split={args.split})")

    models, grad_models = {}, {}
    for k, p in MODELS.items():
        m = cm.load_model(p)
        models[k] = m
        grad_models[k] = keras.Model(m.inputs, [m.get_layer(LAYER).output, m.output])

    focus = {k: [] for k in MODELS}
    records = []
    for path, label, exact in rows:
        img = cm.load_image(path)
        row = {"path": path, "label": label, "exact": exact}
        for k in MODELS:
            cam, idx, conf = _cam(grad_models[k], img)
            focus[k].append(_focus(cam))
            row[k] = dict(cam=cam, pred=names[idx], conf=conf,
                          correct=names[idx] == label)
        records.append(row)

    print(f"\nGrad-CAM focus (mass in hottest 10% of pixels), mean +/- sd:")
    for k in MODELS:
        f = np.array(focus[k])
        print(f"  {k:9s} {f.mean():.3f} +/- {f.std():.3f}")
    # paired: is real-mix more/less focused on the same images?
    d = np.array(focus["real-mix"]) - np.array(focus["default"])
    print(f"  real-mix - default (paired): {d.mean():+.3f} "
          f"({(d > 0).mean():.0%} of photos more focused)")
    for k in MODELS:
        acc = np.mean([r[k]["correct"] for r in records])
        print(f"  {k:9s} top-1 on this split: {acc:.3f}")

    # qualitative panel: a spread of cases, real-mix-correct first
    records.sort(key=lambda r: (not r["real-mix"]["correct"], not r["default"]["correct"]))
    pick = records[:args.n]
    fig, axes = plt.subplots(len(pick), 3, figsize=(7.5, 2.5 * len(pick)))
    if len(pick) == 1:
        axes = axes[None, :]
    for ax_row, r in zip(axes, pick):
        base = cm.load_image(r["path"]) / 255.0
        ax_row[0].imshow(base)
        ax_row[0].set_ylabel(f"label {cm.label_to_time(r['label'])}\n(exact {r['exact']})",
                             fontsize=8, rotation=0, ha="right", va="center", labelpad=28)
        for ax, k in zip(ax_row[1:], ("default", "real-mix")):
            ax.imshow(base)
            hm = tf.image.resize(r[k]["cam"][..., None], cm.IMG_SIZE).numpy().squeeze()
            ax.imshow(hm, cmap="jet", alpha=0.45)
            ok = r[k]["correct"]
            ax.set_title(f"{k}: {cm.label_to_time(r[k]['pred'])} "
                         f"({r[k]['conf']:.2f})",
                         fontsize=8, color="#2f6a91" if ok else "#cf4832")
        for ax in ax_row:
            ax.set_xticks([]); ax.set_yticks([])
    axes[0][0].set_title("photo", fontsize=8)
    fig.suptitle(f"Grad-CAM on real photos: synthetic-only vs. real-fine-tuned "
                 f"({args.split})", fontsize=10)
    fig.tight_layout()
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=130)
    print(f"\nwrote {args.figure}")


if __name__ == "__main__":
    main()
