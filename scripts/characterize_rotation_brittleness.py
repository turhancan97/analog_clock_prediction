#!/usr/bin/env python3
"""Characterize the rotation-brittleness cliff (README.md future work #4).

Prior knowledge (AGENTS.md, from an earlier ad-hoc, uncommitted check on the
*released* checkpoint): 0 deg -> 100%, 3 deg -> ~69%, 6 deg -> 0%, despite
predictions being exactly invariant to 90/180/270 degree rotation. This
script makes that reproducible and extends it two ways:

1. A finer-grained angle sweep (with the current default checkpoint, which
   was trained with +/-10.8 degree RandomRotation augmentation, AND the
   released checkpoint, which saw no rotation augmentation at all) --
   does training-time augmentation widen the surviving angle range, or just
   soften the cliff's slope without moving it?
2. Grad-CAM at increasing rotation angles for one example, to see whether
   attention drifts off the hands, diffuses, or locks onto something else
   entirely as accuracy collapses.

Images are rotated with scipy.ndimage.rotate (reshape=False, mode="nearest",
bilinear interpolation) so corners are edge-extended rather than filled
black -- a black wedge would be a big, unrealistic visual artifact of its
own and confound "rotation" with "sudden black region in frame".
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from scipy.ndimage import rotate as ndi_rotate

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import clockmodel as cm

ANGLES = [-90, -60, -45, -30, -20, -15, -10, -8, -6, -5, -4, -3, -2, -1,
          0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 45, 60, 90]


def rotate_image(img, angle_deg):
    rotated = ndi_rotate(img, angle_deg, axes=(1, 0), reshape=False,
                          mode="nearest", order=1)
    return np.clip(rotated, 0, 255).astype(np.float32)


def circular_minutes_off(true_label, pred_label):
    def to_min(l):
        h, m = cm.label_to_time(l).split(":")
        return (int(h) % 12) * 60 + int(m)
    diff = abs(to_min(pred_label) - to_min(true_label))
    return min(diff, 720 - diff)


def sweep(model, names, images, labels):
    """Returns {angle: (accuracy, mean_confidence, mean_minutes_off)}."""
    results = {}
    for angle in ANGLES:
        batch = np.stack([rotate_image(img, angle) for img in images])
        probs = model.predict(batch, verbose=0)
        pred_idx = probs.argmax(1)
        pred_labels = [names[i] for i in pred_idx]
        acc = np.mean([p == t for p, t in zip(pred_labels, labels)])
        conf = np.mean(probs[np.arange(len(probs)), pred_idx])
        err = np.mean([circular_minutes_off(t, p) for t, p in zip(labels, pred_labels)])
        results[angle] = (acc, conf, err)
        print(f"  angle {angle:+4d} deg: acc={acc:.3f}  mean_conf={conf:.3f}  mean_err={err:.1f} min")
    return results


GRADCAM_LAYER = "top_activation"


def make_gradcam(model, img_float, pred_index=None):
    grad_model = keras.Model(model.inputs, [model.get_layer(GRADCAM_LAYER).output, model.output])
    img_batch = tf.convert_to_tensor(img_float[None, ...])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_score = preds[:, pred_index]
    grads = tape.gradient(class_score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index, preds[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1,
                     help="images sampled per class from the test split (default 1 -> 144 images)")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "docs" / "images"))
    args = ap.parse_args()

    names = cm.class_names()
    rng = np.random.default_rng(0)

    images, labels = [], []
    for cls in names:
        paths = sorted((cm.DATA_DIR / "test" / cls).glob("*.jpg"))
        for p in rng.choice(paths, size=min(args.per_class, len(paths)), replace=False):
            images.append(cm.load_image(p))
            labels.append(cls)
    print(f"sampled {len(images)} test images across {len(names)} classes")

    default_model = cm.load_model()
    baseline_model = cm.load_model(cm.DATA_DIR / "time-99.68.h5")

    print("\n--- default checkpoint (clock_model_unfrozen_aug_80ep.keras, trained with +/-10.8deg aug) ---")
    default_results = sweep(default_model, names, images, labels)

    print("\n--- released checkpoint (time-99.68.h5, no rotation augmentation) ---")
    baseline_results = sweep(baseline_model, names, images, labels)

    # --- Figure 1: accuracy vs. rotation angle, both checkpoints ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, results, color in [
        ("default (80-epoch, rotation-aug)", default_results, "#4C72B0"),
        ("released (time-99.68.h5, no aug)", baseline_results, "#DD8452"),
    ]:
        xs = sorted(results)
        ys = [results[a][0] * 100 for a in xs]
        ax.plot(xs, ys, marker="o", markersize=4, label=label, color=color)
    ax.axvspan(-10.8, 10.8, alpha=0.08, color="#4C72B0", label="default's training rotation range (+/-10.8deg)")
    ax.set_xlabel("rotation (degrees)")
    ax.set_ylabel("top-1 accuracy (%)")
    ax.set_title(f"Rotation brittleness: accuracy vs. angle ({len(images)} test images)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = Path(args.out_dir) / "rotation_cliff.png"
    plt.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"\nwrote {fig_path}")

    # --- Figure 2: Grad-CAM at increasing rotation, default checkpoint ---
    example_idx = 0
    example_img, example_label = images[example_idx], labels[example_idx]
    gradcam_angles = [0, 3, 6, 10, 20, 45]
    fig, axes = plt.subplots(1, len(gradcam_angles), figsize=(3 * len(gradcam_angles), 3.6))
    for ax, angle in zip(axes, gradcam_angles):
        rotated = rotate_image(example_img, angle)
        heatmap, pred_idx, probs = make_gradcam(default_model, rotated)
        heatmap_big = tf.image.resize(heatmap[..., None], cm.IMG_SIZE).numpy().squeeze()
        ax.imshow(rotated / 255.0)
        ax.imshow(heatmap_big, cmap="jet", alpha=0.45)
        ax.axis("off")
        pred_label = names[pred_idx]
        ok = pred_label == example_label
        ax.set_title(f"{angle:+d} deg\npred {cm.label_to_time(pred_label)} ({probs[pred_idx]:.2f})",
                     color="black" if ok else "crimson", fontsize=9)
    plt.suptitle(f"Grad-CAM vs. rotation angle (true {cm.label_to_time(example_label)}, default checkpoint)")
    plt.tight_layout()
    fig_path = Path(args.out_dir) / "rotation_gradcam.png"
    plt.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
