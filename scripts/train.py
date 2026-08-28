#!/usr/bin/env python3
"""Train a clock reader. Default backbone is EfficientNetB3 (the released
model's topology); ``--backbone`` swaps in a smaller one for the size/latency
ablation (see CHANGELOG.md, backbone ablation)."""
import argparse
import sys
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import clockmodel as cm

_TWO_PI = 2.0 * np.pi


@keras.saving.register_keras_serializable(package="clock")
class MeanMinutesError(keras.metrics.Metric):
    """Mean circular error in minutes for the (sin, cos) regression head."""

    def __init__(self, name="min_err", **kw):
        super().__init__(name=name, **kw)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        at = tf.atan2(y_true[:, 0], y_true[:, 1])
        ap = tf.atan2(y_pred[:, 0], y_pred[:, 1])
        d = tf.abs(at - ap)
        d = tf.minimum(d, _TWO_PI - d) / _TWO_PI * 720.0
        self.total.assign_add(tf.reduce_sum(d))
        self.count.assign_add(tf.cast(tf.shape(d)[0], tf.float32))

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

# name -> keras.applications constructor. All of these ship an input-scaling
# layer inside the returned model (EfficientNet* and MobileNetV3* default to
# include_preprocessing=True), so they take raw [0, 255] floats just like the
# released checkpoint -- do not pre-scale. ResNet50V2 is the exception and
# gets an explicit preprocessing layer below; "simplecnn" is built from
# scratch by build_simplecnn().
APPLICATIONS = {
    "efficientnetb3": keras.applications.EfficientNetB3,
    "efficientnetb0": keras.applications.EfficientNetB0,
    "mobilenetv3small": keras.applications.MobileNetV3Small,
    "resnet50v2": keras.applications.ResNet50V2,
}
BACKBONES = tuple(APPLICATIONS) + ("simplecnn",)


def _head(features, dropout, head="softmax"):
    """Shared trunk (GlobalMaxPool -> BN -> 256 -> Dropout) then either the flat
    144-way softmax or a 2-unit (sin, cos) circular-regression output."""
    x = keras.layers.GlobalMaxPooling2D(name="max_pool")(features)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    if head == "circular":
        x = keras.layers.Dense(2, name="sincos")(x)
        return keras.layers.UnitNormalization(name="unit")(x)
    return keras.layers.Dense(cm.NUM_CLASSES, activation="softmax")(x)


def build_simplecnn(dropout, head="softmax", trainable_backbone=True):
    """A ~1M-param ConvNet trained from scratch -- no ImageNet prior, which
    this clean synthetic line-art arguably doesn't benefit from anyway."""
    inp = keras.Input(shape=(*cm.IMG_SIZE, 3))
    x = keras.layers.Rescaling(1.0 / 255)(inp)
    for filters in (32, 64, 128, 128, 256):
        x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling2D()(x)
    return keras.Model(inp, _head(x, dropout, head))


def build_model(backbone="efficientnetb3", dropout=0.3, learning_rate=1e-3,
                head="softmax", trainable_backbone=True):
    """Chosen backbone -> shared head. Compiled with Adamax (released setup).

    head="softmax": 144-way classification, categorical_crossentropy.
    head="circular": (sin, cos) regression on the dial angle, MSE + MeanMinutesError.
    """
    if backbone == "simplecnn":
        model = build_simplecnn(dropout, head, trainable_backbone)
    else:
        base = APPLICATIONS[backbone](
            include_top=False, weights="imagenet",
            input_shape=(*cm.IMG_SIZE, 3),
        )
        base.trainable = trainable_backbone
        inp = base.input
        feats = base.output
        if backbone == "resnet50v2":
            # ResNet50V2 has no built-in preprocessing; the others do.
            inp = keras.Input(shape=(*cm.IMG_SIZE, 3))
            scaled = keras.applications.resnet_v2.preprocess_input(
                keras.layers.Identity()(inp))
            feats = base(scaled)
        model = keras.Model(inp, _head(feats, dropout, head))

    if head == "circular":
        loss, metrics = "mse", [MeanMinutesError()]
    else:
        loss, metrics = "categorical_crossentropy", ["accuracy"]
    model.compile(
        optimizer=keras.optimizers.Adamax(learning_rate=learning_rate),
        loss=loss, metrics=metrics,
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=5,
                    help="EarlyStopping patience on the monitored metric")
    ap.add_argument("--backbone", choices=BACKBONES, default="efficientnetb3")
    ap.add_argument("--head", choices=("softmax", "circular"), default="softmax",
                    help="softmax = 144-way; circular = (sin, cos) angle regression")
    ap.add_argument("--seed", type=int, default=None,
                    help="set for reproducible weights/augmentation/shuffling")
    ap.add_argument("--rotation-factor", type=float, default=0.15,
                    help="RandomRotation factor (fraction of 2pi; 0.15 = +/-54 deg). "
                         "0.15 is the default -- it widens the rotation-robust "
                         "plateau to the full +/-90 deg sweep at no upright cost "
                         "(CHANGELOG 2026-08-28). 0.03 = +/-10.8 deg was the old default.")
    ap.add_argument("--out", default=str(cm.MODELS_DIR / "clock_model.keras"))
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="train only the head (much faster, lower ceiling)")
    args = ap.parse_args()
    Path(args.out).resolve().parent.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        keras.utils.set_random_seed(args.seed)

    train_ds = cm.make_dataset("train", args.batch_size, shuffle=True)
    valid_ds = cm.make_dataset("valid", args.batch_size)

    # Augment layers expect raw [0, 255]; input scaling lives inside the model.
    augment = keras.Sequential([
        keras.layers.RandomRotation(args.rotation_factor, fill_mode="nearest"),
        keras.layers.RandomZoom(0.05, fill_mode="nearest"),
        keras.layers.RandomTranslation(0.05, 0.05, fill_mode="nearest"),
    ], name="augment")
    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)

    if args.head == "circular":
        # one-hot label -> (sin, cos) regression target for that class angle
        targets = tf.constant(cm.CIRCULAR_TARGETS)
        to_circ = lambda x, y: (x, tf.gather(targets, tf.argmax(y, axis=1)))
        train_ds = train_ds.map(to_circ, num_parallel_calls=tf.data.AUTOTUNE)
        valid_ds = valid_ds.map(to_circ, num_parallel_calls=tf.data.AUTOTUNE)
        monitor, mode = "val_min_err", "min"
    else:
        monitor, mode = "val_accuracy", "max"

    model = build_model(args.backbone, args.dropout, args.learning_rate,
                        head=args.head, trainable_backbone=not args.freeze_backbone)
    model.summary()
    print(f"\nbackbone: {args.backbone}   head: {args.head}   "
          f"rotation_factor: {args.rotation_factor}   "
          f"params: {model.count_params():,}")

    model.fit(
        train_ds, validation_data=valid_ds, epochs=args.epochs,
        callbacks=[
            keras.callbacks.ModelCheckpoint(args.out, monitor=monitor, mode=mode,
                                            save_best_only=True, verbose=1),
            keras.callbacks.EarlyStopping(monitor=monitor, mode=mode,
                                          patience=args.patience,
                                          restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=2, verbose=1),
        ],
    )

    # Head-agnostic test report: decode whatever the head emits to class indices.
    test_ds = cm.make_dataset("test", args.batch_size)
    preds = model.predict(test_ds, verbose=0)
    y_true = np.concatenate([y.numpy().argmax(1) for _, y in test_ds])
    y_pred = cm.output_to_class_idx(preds)
    names = cm.class_names()
    to_min = lambda i: (int(names[i].split("-")[0]) % 12) * 60 + int(names[i].split("-")[1])
    d = np.abs([to_min(p) - to_min(t) for p, t in zip(y_pred, y_true)])
    err = np.minimum(d, 720 - d)
    print(f"\nbest model saved to {args.out}")
    print(f"test top-1     : {(y_pred == y_true).mean():.4f}")
    print(f"test mean |err|: {err.mean():.1f} min")


if __name__ == "__main__":
    main()
