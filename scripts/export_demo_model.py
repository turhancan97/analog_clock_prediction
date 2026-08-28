#!/usr/bin/env python3
"""Export a trained `simplecnn` checkpoint to a compact JSON blob that the
browser inference demo (docs/demo/) embeds and runs with TensorFlow.js.

There is no TF.js conversion step: the demo rebuilds this exact architecture
in tfjs-layers and calls setWeights(). This script only has to dump the weight
tensors, in Keras `get_weights()` order, symmetric-int8 quantised (per tensor,
scale = max|w| / 127). ~0.64 M params -> ~0.65 MB int8 -> ~0.87 MB base64.

    python scripts/export_demo_model.py models/clock_demo_simplecnn.keras \\
        --out docs/demo/model.json

Also prints the accuracy cost of the quantisation (re-loads the dequantised
weights and re-evaluates the test split).
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import clockmodel as cm

# Layers that carry weights, in model order. Everything else (Rescaling, ReLU,
# MaxPool, GlobalMaxPool, Dropout) is structural and rebuilt in JS.
WEIGHTED = ("Conv2D", "BatchNormalization", "Dense")


def _collect(model):
    """Weight tensors in TF.js `model.weights` order: every trainable weight
    across all layers first (in layer order), then every non-trainable weight
    (the BN moving_mean / moving_variance). Keras groups by layer instead, so
    this reorder is what makes setWeights() line up on the JS side."""
    trainable, non_trainable = [], []
    for layer in model.layers:
        if layer.__class__.__name__ not in WEIGHTED:
            continue
        trainable += [w.numpy().astype("float32") for w in layer.trainable_weights]
        non_trainable += [w.numpy().astype("float32") for w in layer.non_trainable_weights]
    return trainable + non_trainable


def _scale(w):
    return float(np.abs(w).max()) / 127.0 or 1e-8


def _quantize(tensors):
    scales, blobs, shapes = [], [], []
    for w in tensors:
        s = _scale(w)
        q = np.clip(np.round(w / s), -127, 127).astype(np.int8)
        scales.append(s)
        shapes.append(list(w.shape))
        blobs.append(q.tobytes())
    return scales, shapes, b"".join(blobs)


def _quant_dequant(w):
    s = _scale(w)
    return (np.clip(np.round(w / s), -127, 127) * s).astype("float32")


def _dequantized_model(model):
    """int8-round-trip every weight in place, to measure the accuracy cost."""
    for layer in model.layers:
        if layer.__class__.__name__ in WEIGHTED:
            layer.set_weights([_quant_dequant(w) for w in layer.get_weights()])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=str(cm.REPO_DIR / "docs" / "demo" / "model.json"))
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    model = cm.load_model(args.checkpoint)
    assert model.input_shape[1:] == (*cm.IMG_SIZE, 3), model.input_shape
    assert model.output_shape[-1] == cm.NUM_CLASSES

    tensors = _collect(model)
    scales, shapes, blob = _quantize(tensors)
    n_params = sum(int(np.prod(s)) for s in shapes)

    payload = {
        "arch": "simplecnn",
        "img_size": list(cm.IMG_SIZE),
        "num_classes": cm.NUM_CLASSES,
        "class_names": cm.class_names(),
        "filters": [32, 64, 128, 128, 256],
        "scales": scales,
        "shapes": shapes,
        "int8_b64": base64.b64encode(blob).decode("ascii"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, separators=(",", ":")))
    kb = Path(args.out).stat().st_size / 1024
    print(f"wrote {args.out}  ({n_params:,} params, {len(blob)/1024:.0f} KB int8, "
          f"{kb:.0f} KB json)")

    if args.no_eval:
        return
    names = cm.class_names()
    ds = cm.make_dataset("test", batch_size=64)
    y_true = np.concatenate([y.numpy().argmax(1) for _, y in ds])
    p_fp = cm.load_model(args.checkpoint).predict(ds, verbose=0).argmax(1)
    p_q = _dequantized_model(cm.load_model(args.checkpoint)) \
        .predict(ds, verbose=0).argmax(1)
    print(f"test top-1  fp32: {(p_fp == y_true).mean():.4f}   "
          f"int8: {(p_q == y_true).mean():.4f}")


if __name__ == "__main__":
    main()
