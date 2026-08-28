"""Shared helpers: class labels, model loading, image preprocessing."""
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data"
MODELS_DIR = REPO_DIR / "models"
IMG_SIZE = (150, 150)          # what the released model expects
NUM_CLASSES = 144
MINUTES_PER_CLASS = 5          # 720 min on a 12h dial / 144 classes

# Class i is i*5 minutes past 12 o'clock, i.e. angle 2*pi*i/144 around the
# dial. CIRCULAR_TARGETS[i] = (sin, cos) of that angle -- the regression
# target for the circular head (train.py --head circular), which respects
# that 11:55 and 12:00 are adjacent where the flat 144-way softmax does not.
_ANGLES = 2.0 * np.pi * np.arange(NUM_CLASSES) / NUM_CLASSES
CIRCULAR_TARGETS = np.stack([np.sin(_ANGLES), np.cos(_ANGLES)], axis=1).astype("float32")


def output_to_class_idx(preds):
    """Model output -> class index, dispatching on the head. (batch, 144)
    softmax probabilities -> argmax; (batch, 2) (sin, cos) -> nearest class."""
    preds = np.asarray(preds)
    if preds.shape[-1] == NUM_CLASSES:
        return preds.argmax(-1)
    if preds.shape[-1] == 2:
        ang = np.mod(np.arctan2(preds[..., 0], preds[..., 1]), 2.0 * np.pi)
        return np.mod(np.round(ang / (2.0 * np.pi / NUM_CLASSES)).astype(int),
                      NUM_CLASSES)
    raise ValueError(f"unexpected model output width {preds.shape[-1]}")

# Best checkpoint measured so far (2026-08-28): EfficientNetB3, seed 0, with
# +/-54 deg RandomRotation augmentation (--rotation-factor 0.15). 99.82% on
# the full dataset and robust across the entire +/-90 deg rotation sweep, vs
# the +/-10.8 deg predecessor's 99.77% and +/-11 deg plateau -- see
# CHANGELOG.md 2026-08-28. Gitignored like all *.keras artifacts; falls back
# through the previous default to the released checkpoint on a fresh clone.
for _candidate in ("clock_model_rot54_s0.keras",
                   "clock_model_unfrozen_aug_80ep.keras"):
    DEFAULT_MODEL = MODELS_DIR / _candidate
    if DEFAULT_MODEL.exists():
        break
else:
    DEFAULT_MODEL = DATA_DIR / "time-99.68.h5"


class _CompatDepthwiseConv2D(keras.layers.DepthwiseConv2D):
    """Keras 2.8 wrote a `groups` entry that Keras 3 no longer accepts."""

    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return cls(**config)


def class_names(data_dir=DATA_DIR):
    """The 144 labels in the order the model's output units use.

    The model was trained off the `labels` column of clocks.csv, so the order is
    alphabetical over the UNDERSCORE form ('10_00' ... '1_00' ... '9_55'). That
    differs from sorting the hyphenated directory names, because '_' sorts after
    digits while '-' sorts before them -- sorting the wrong form silently costs
    ~33% accuracy by shifting whole hours. It is also not the `class index`
    column, which is ordered by hour then minute.
    """
    labels = pd.read_csv(Path(data_dir) / "clocks.csv")["labels"].unique()
    return [l.replace("_", "-") for l in sorted(labels)]


def load_model(path=DEFAULT_MODEL, compile=False):
    return keras.saving.load_model(
        path, compile=compile,
        custom_objects={"DepthwiseConv2D": _CompatDepthwiseConv2D},
    )


def load_image(path, size=IMG_SIZE):
    """One image as a float32 [0, 255] array. Rescaling lives inside the model."""
    img = tf.io.decode_image(tf.io.read_file(str(path)), channels=3,
                             expand_animations=False)
    return tf.image.resize(tf.cast(img, tf.float32), size).numpy()


def make_dataset(split, batch_size=32, shuffle=False, data_dir=DATA_DIR):
    """A tf.data pipeline over data/<split>/, label-aligned with class_names()."""
    ds = keras.utils.image_dataset_from_directory(
        Path(data_dir) / split,
        labels="inferred", label_mode="categorical",
        class_names=class_names(data_dir),
        image_size=IMG_SIZE, batch_size=batch_size,
        shuffle=shuffle, interpolation="bilinear",
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def label_to_time(label):
    """'10-35' or '10_35' -> '10:35'."""
    hour, minute = label.replace("_", "-").split("-")
    return f"{int(hour)}:{minute}"


def decode_predictions(probs, names=None, top=1):
    """(batch, 144) probabilities -> list of [(label, time, prob), ...] per row."""
    names = names or class_names()
    out = []
    for row in np.atleast_2d(probs):
        idx = np.argsort(row)[::-1][:top]
        out.append([(names[i], label_to_time(names[i]), float(row[i])) for i in idx])
    return out
