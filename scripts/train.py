#!/usr/bin/env python3
"""Train an EfficientNetB3 clock reader, matching the released model's setup."""
import argparse
import sys
from pathlib import Path

import keras
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import clockmodel as cm


def build_model(dropout=0.3, learning_rate=1e-3, trainable_backbone=True):
    """Same topology as data/time-99.68.h5: B3 -> GlobalMaxPool -> BN -> 256 -> 144."""
    backbone = keras.applications.EfficientNetB3(
        include_top=False, weights="imagenet",
        input_shape=(*cm.IMG_SIZE, 3),
    )
    backbone.trainable = trainable_backbone

    x = keras.layers.GlobalMaxPooling2D(name="max_pool")(backbone.output)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(cm.NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(backbone.input, out)
    model.compile(
        # Adamax is what the released checkpoint was trained with.
        optimizer=keras.optimizers.Adamax(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--out", default=str(cm.MODELS_DIR / "clock_model.keras"))
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="train only the head (much faster, lower ceiling)")
    args = ap.parse_args()
    Path(args.out).resolve().parent.mkdir(parents=True, exist_ok=True)

    train_ds = cm.make_dataset("train", args.batch_size, shuffle=True)
    valid_ds = cm.make_dataset("valid", args.batch_size)

    # EfficientNet's own Rescaling/Normalization layers expect raw [0, 255].
    augment = keras.Sequential([
        keras.layers.RandomRotation(0.03, fill_mode="nearest"),
        keras.layers.RandomZoom(0.05, fill_mode="nearest"),
        keras.layers.RandomTranslation(0.05, 0.05, fill_mode="nearest"),
    ], name="augment")
    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)

    model = build_model(args.dropout, args.learning_rate,
                        trainable_backbone=not args.freeze_backbone)
    model.summary()

    model.fit(
        train_ds, validation_data=valid_ds, epochs=args.epochs,
        callbacks=[
            keras.callbacks.ModelCheckpoint(args.out, monitor="val_accuracy",
                                            save_best_only=True, verbose=1),
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                          restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=2, verbose=1),
        ],
    )

    loss, acc = model.evaluate(cm.make_dataset("test", args.batch_size), verbose=1)
    print(f"\nbest model saved to {args.out}")
    print(f"test accuracy: {acc:.4f}  (loss {loss:.4f})")


if __name__ == "__main__":
    main()
