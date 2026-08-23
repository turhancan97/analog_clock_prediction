"""Shared helpers: class labels, model loading, image preprocessing."""
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPO_DIR = Path(__file__).resolve().parent.parent
IMG_SIZE = (150, 150)          # what the released model expects
NUM_CLASSES = 144

# Best checkpoint measured so far (2026-08-23): rotation-aug + 80-epoch
# fine-tune, 99.77% on the full dataset vs the released time-99.68.h5's
# 99.57% -- see CHANGELOG.md. Gitignored like all *.keras artifacts; falls
# back to the released checkpoint if it isn't present (e.g. fresh clone).
DEFAULT_MODEL = REPO_DIR / "clock_model_unfrozen_aug_80ep.keras"
if not DEFAULT_MODEL.exists():
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
