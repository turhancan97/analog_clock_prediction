"""Generate the static PNGs embedded in README.md.

One-off script, not part of the train/eval/predict pipeline. Re-run and
recommit the PNGs under docs/images/ if the default checkpoint changes or the
dataset defects get fixed upstream.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import clockmodel as cm

OUT_DIR = REPO_ROOT / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.facecolor"] = "white"

model = cm.load_model()
names = cm.class_names()
rng = np.random.default_rng(0)


def predict_one(path):
    img = cm.load_image(path)
    probs = model.predict(img[None, ...], verbose=0)[0]
    idx = int(np.argmax(probs))
    return img, names[idx], float(probs[idx])


# --- 1. Sample predictions grid -------------------------------------------
sample_classes = rng.choice(names, size=12, replace=False)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for ax, cls in zip(axes.flat, sample_classes):
    img_path = sorted((cm.DATA_DIR / "test" / cls).glob("*.jpg"))[0]
    img, pred_label, prob = predict_one(img_path)
    ok = pred_label == cls
    ax.imshow(img / 255.0)
    ax.set_title(
        f"true {cm.label_to_time(cls)} -> pred {cm.label_to_time(pred_label)} ({prob:.2f})",
        color="black" if ok else "crimson", fontsize=9,
    )
    ax.axis("off")
plt.suptitle("Sample predictions (test split, one image per class)")
plt.tight_layout()
plt.savefig(OUT_DIR / "sample_predictions.png", dpi=130)
plt.close(fig)
print("wrote sample_predictions.png")


# --- 2. Grad-CAM comparison -------------------------------------------------
GRADCAM_LAYER = "top_activation"


def make_gradcam(img_float, pred_index=None):
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


def show_gradcam(ax, path, true_label, title_prefix):
    img = cm.load_image(path)
    heatmap, pred_idx, probs = make_gradcam(img)
    heatmap_big = tf.image.resize(heatmap[..., None], cm.IMG_SIZE).numpy().squeeze()
    ax.imshow(img / 255.0)
    ax.imshow(heatmap_big, cmap="jet", alpha=0.45)
    ax.axis("off")
    pred_label = names[pred_idx]
    ok = pred_label == true_label
    title = (
        f"{title_prefix}\ntrue {cm.label_to_time(true_label)} -> "
        f"pred {cm.label_to_time(pred_label)} ({probs[pred_idx]:.2f})"
    )
    ax.set_title(title, color="black" if ok else "crimson", fontsize=9)


normal_gradcam_path = sorted((cm.DATA_DIR / "test" / "9-25").glob("*.jpg"))[0]

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
show_gradcam(axes[0], normal_gradcam_path, "9-25", "correct, confident")
show_gradcam(axes[1], cm.DATA_DIR / "train" / "12-30" / "36.jpg", "12-30", "blank-dial defect")
show_gradcam(axes[2], cm.DATA_DIR / "valid" / "8-50" / "36.jpg", "8-50", "shift defect (mislabeled)")
plt.suptitle("Grad-CAM: where the model looks")
plt.tight_layout()
plt.savefig(OUT_DIR / "gradcam_comparison.png", dpi=130)
plt.close(fig)
print("wrote gradcam_comparison.png")


# --- 3. Dataset defects side by side ---------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(10.5, 4))
for ax, (path, title) in zip(axes, [
    (cm.DATA_DIR / "train" / "9-25" / "36.jpg", "normal render\n(9-25, index 36)"),
    (cm.DATA_DIR / "train" / "12-30" / "36.jpg", "blank-dial defect\n(labeled 12-30, index 36)"),
    (cm.DATA_DIR / "train" / "2-50" / "0.jpg", "±3h15m shift defect\n(labeled 2-50, hands show ~6:05)"),
]):
    img = cm.load_image(path) / 255.0
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
plt.suptitle("Dataset rendering defects (see DATASET.md)")
plt.tight_layout()
plt.savefig(OUT_DIR / "dataset_defects.png", dpi=130)
plt.close(fig)
print("wrote dataset_defects.png")


# --- 4. Checkpoint accuracy comparison --------------------------------------
# Figures from AGENTS.md's GPU Slurm results table (2026-08-23). The 20-epoch
# checkpoint was deleted after being superseded, so these are the recorded
# measurements, not recomputed here.
checkpoints = ["released\n(time-99.68.h5)", "20-epoch\nrotation-aug", "80-epoch\nrotation-aug (default)"]
accuracy = [99.57, 99.49, 99.77]
errors = [62, 73, 33]

fig, ax1 = plt.subplots(figsize=(7, 4.5))
color = "#4C72B0"
bars = ax1.bar(checkpoints, accuracy, color=color)
ax1.set_ylabel("dataset-wide accuracy (%)", color=color)
ax1.set_ylim(99.0, 100.0)
ax1.tick_params(axis="y", labelcolor=color)
for bar, acc, err in zip(bars, accuracy, errors):
    ax1.annotate(f"{acc:.2f}%\n({err} errors)", (bar.get_x() + bar.get_width() / 2, acc),
                 ha="center", va="bottom", fontsize=9)
ax1.set_title("Full-dataset accuracy by checkpoint (14,400 images)")
plt.tight_layout()
plt.savefig(OUT_DIR / "checkpoint_accuracy.png", dpi=130)
plt.close(fig)
print("wrote checkpoint_accuracy.png")
