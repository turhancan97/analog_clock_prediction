"""Tests for src/clockmodel.py.

Split into two tiers:
- Pure-logic tests (labels, config-patching) need neither the dataset nor a
  trained checkpoint, so they always run, including in CI.
- Data/model-backed tests (class ordering, accuracy regression) need
  data/ (gitignored, a symlink to shared storage) and/or a trained
  checkpoint (gitignored). They skip automatically when those aren't
  present -- e.g. in CI, which has neither -- and are meant to catch
  regressions when run locally.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import clockmodel as cm

DATA_AVAILABLE = cm.DATA_DIR.exists() and (cm.DATA_DIR / "clocks.csv").exists()
MODEL_AVAILABLE = cm.DEFAULT_MODEL.exists()

requires_data = pytest.mark.skipif(not DATA_AVAILABLE, reason="data/ not available")
requires_model = pytest.mark.skipif(not MODEL_AVAILABLE, reason="no trained checkpoint available")


# --- Pure logic --------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    ("10-35", "10:35"),
    ("10_35", "10:35"),
    ("1-00", "1:00"),
    ("9-05", "9:05"),
])
def test_label_to_time(label, expected):
    assert cm.label_to_time(label) == expected


def test_decode_predictions_top1():
    names = ["0-00", "0-05", "0-10"]
    probs = np.array([[0.1, 0.7, 0.2]])
    out = cm.decode_predictions(probs, names=names, top=1)
    assert out == [[("0-05", "0:05", 0.7)]]


def test_decode_predictions_top_n_ranked_by_probability():
    names = ["0-00", "0-05", "0-10"]
    probs = np.array([[0.1, 0.7, 0.2]])
    out = cm.decode_predictions(probs, names=names, top=3)
    labels = [label for label, _, _ in out[0]]
    assert labels == ["0-05", "0-10", "0-00"]


def test_compat_depthwise_conv2d_drops_groups_key():
    # Regression test for the Keras-2.8-to-3 compat gotcha (AGENTS.md #2):
    # Keras 2.8 wrote a `groups` key into every DepthwiseConv2D config that
    # Keras 3's DepthwiseConv2D.from_config rejects outright.
    config = {"name": "dw_conv", "kernel_size": (3, 3), "groups": 40}
    layer = cm._CompatDepthwiseConv2D.from_config(dict(config))
    assert layer.name == "dw_conv"


# --- Data-backed ---------------------------------------------------------

@requires_data
def test_class_names_count_and_uniqueness():
    names = cm.class_names()
    assert len(names) == cm.NUM_CLASSES
    assert len(set(names)) == cm.NUM_CLASSES


@requires_data
def test_class_names_order_is_alphabetical_over_underscore_labels():
    # Regression test for the class-ordering trap (AGENTS.md #1): the model
    # was trained off clocks.csv's underscore-form `labels` column, sorted
    # alphabetically. Sorting the hyphenated directory names instead silently
    # costs ~33% accuracy (every error exactly one hour off).
    names = cm.class_names()
    underscore_sorted = sorted(l.replace("-", "_") for l in names)
    assert [l.replace("-", "_") for l in names] == underscore_sorted

    hyphen_sorted = sorted(names)
    assert names != hyphen_sorted, (
        "class_names() must NOT match sorting the hyphenated directory names "
        "-- that ordering silently scores ~66.6%, see AGENTS.md"
    )


@requires_data
def test_class_names_does_not_match_csv_class_index_order():
    import pandas as pd
    df = pd.read_csv(cm.DATA_DIR / "clocks.csv")
    csv_index_order = [
        l.replace("_", "-") for l in
        df.sort_values("class index")["labels"].unique()
    ]
    names = cm.class_names()
    assert names != csv_index_order, (
        "class_names() must NOT match clocks.csv's `class index` column "
        "-- that ordering silently scores ~65%, see AGENTS.md"
    )


@requires_model
@requires_data
def test_default_checkpoint_test_accuracy_regression():
    # Locks in the measured accuracy from AGENTS.md/README.md (~99.72% top-1
    # on the test split for the default checkpoint, the +/-54 deg rotation-aug
    # model from 2026-08-28) so a future change to preprocessing, class
    # ordering, or model loading gets caught instead of silently shipping a
    # worse model.
    names = cm.class_names()
    model = cm.load_model()
    ds = cm.make_dataset("test", batch_size=64)

    probs = model.predict(ds, verbose=0)
    y_true = np.concatenate([y.numpy().argmax(1) for _, y in ds])
    y_pred = cm.output_to_class_idx(probs)

    acc = (y_pred == y_true).mean()
    assert acc >= 0.995, f"test top-1 accuracy dropped to {acc:.4f} (expected >= 0.995)"
