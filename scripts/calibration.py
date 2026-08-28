#!/usr/bin/env python3
"""Is the model's confidence trustworthy -- does p(top-1) = 0.9 actually mean
~90% correct? Reliability diagram + ECE, plus post-hoc temperature scaling and
a selective-prediction table (what accuracy you get if you only keep
predictions above a confidence threshold and defer the rest to a human).

Temperature is fit once on --fit-split (minimising NLL) and applied everywhere.
The model emits softmax probabilities, not logits, so scaling is done on
log(p): softmax(log(p) / T), which is exact at T = 1 and a valid
sharpen/soften either side.

    python scripts/calibration.py --split test
    python scripts/calibration.py --split real-test          # needs real_data/real_manifest.csv
    python scripts/calibration.py --split test --fit-split valid --figure docs/images/calibration.png

Softmax head only -- the circular regression head has no probability.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import clockmodel as cm

N_BINS = 15
THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def _load_split(name, batch_size=64):
    """-> (y_true, y_pred_probs) for a synthetic split or a real manifest split."""
    if name.startswith("real-"):
        ds = cm.make_real_dataset(name[len("real-"):], batch_size)
    elif name == "all":
        import tensorflow as tf
        parts = [cm.make_dataset(s, batch_size) for s in ("train", "valid", "test")]
        ds = parts[0]
        for p in parts[1:]:
            ds = ds.concatenate(p)
    else:
        ds = cm.make_dataset(name, batch_size)
    return ds


def _score(model, ds):
    probs = model.predict(ds, verbose=1)
    y_true = np.concatenate([y.numpy().argmax(1) for _, y in ds])
    return y_true, probs


def _apply_T(probs, T):
    logp = np.log(np.clip(probs, 1e-12, 1.0)) / T
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def _metrics(probs, y_true):
    pred = probs.argmax(1)
    conf = probs[np.arange(len(probs)), pred]
    correct = (pred == y_true).astype(float)
    nll = -np.log(probs[np.arange(len(probs)), y_true] + 1e-12).mean()
    brier = ((probs - np.eye(cm.NUM_CLASSES)[y_true]) ** 2).sum(1).mean()

    edges = np.linspace(0, 1, N_BINS + 1)
    ece = mce = 0.0
    bin_stats = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            bin_stats.append((lo, hi, 0, np.nan, np.nan))
            continue
        acc_b, conf_b, w = correct[m].mean(), conf[m].mean(), m.mean()
        gap = abs(acc_b - conf_b)
        ece += w * gap
        mce = max(mce, gap)
        bin_stats.append((lo, hi, int(m.sum()), acc_b, conf_b))
    return dict(acc=correct.mean(), conf=conf, correct=correct, ece=ece,
               mce=mce, nll=nll, brier=brier, bins=bin_stats)


def _reliability_panel(ax, stats, title):
    xs = [(lo + hi) / 2 for lo, hi, *_ in stats["bins"]]
    accs = [b[3] for b in stats["bins"]]
    width = 1 / N_BINS
    ax.plot([0, 1], [0, 1], color="#9aa2a6", ls="--", lw=1, zorder=1)
    ax.bar(xs, accs, width=width * 0.92, color="#3a7eab", alpha=0.85,
           edgecolor="#2f6a91", zorder=2)
    for x, b in zip(xs, stats["bins"]):
        if not np.isnan(b[3]) and abs(b[3] - b[4]) > 1e-3:
            ax.plot([x, x], [b[3], b[4]], color="#cf4832", lw=1.4, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_title(f"{title}\nECE {stats['ece']:.3f}  ·  MCE {stats['mce']:.3f}",
                 fontsize=10)
    ax.set_aspect("equal")


def _selective_table(conf, correct):
    print(f"\n{'thresh':>7} {'coverage':>9} {'accuracy':>9} {'deferred':>9}")
    print("-" * 38)
    for t in THRESHOLDS:
        keep = conf >= t
        cov = keep.mean()
        acc = correct[keep].mean() if keep.any() else float("nan")
        print(f"{t:>7.2f} {cov:>8.1%} {acc:>8.2%} {1 - cov:>8.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test",
                    help="test / valid / train / all / real-test / real-train")
    ap.add_argument("--fit-split", default="valid",
                    help="split to fit the temperature on (default valid)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--figure", default=str(cm.REPO_DIR / "docs" / "images" / "calibration.png"))
    args = ap.parse_args()

    model = cm.load_model(args.model) if args.model else cm.load_model()

    print(f"fitting temperature on '{args.fit_split}' ...")
    yf, pf = _score(model, _load_split(args.fit_split))
    if pf.shape[-1] != cm.NUM_CLASSES:
        sys.exit("calibration needs the softmax head (got a non-144-wide output)")
    logpf = np.log(np.clip(pf, 1e-12, 1.0))

    def nll_T(T):
        z = logpf / T
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        return -np.log(p[np.arange(len(p)), yf] + 1e-12).mean()

    T = float(minimize_scalar(nll_T, bounds=(0.3, 10.0), method="bounded").x)
    direction = ("model was over-confident, softening" if T > 1.02
                 else "model was under-confident, sharpening" if T < 0.98
                 else "model was already well-scaled")
    print(f"fitted temperature T = {T:.3f}  ({direction})")

    print(f"\nscoring '{args.split}' ...")
    yt, pt = _score(model, _load_split(args.split))
    before = _metrics(pt, yt)
    after = _metrics(_apply_T(pt, T), yt)

    print(f"\nsplit '{args.split}'  ({len(yt)} images)   top-1 {before['acc']:.4f}")
    print(f"{'':16}{'ECE':>8}{'MCE':>8}{'NLL':>8}{'Brier':>8}")
    print(f"{'before scaling':16}{before['ece']:>8.3f}{before['mce']:>8.3f}"
          f"{before['nll']:>8.3f}{before['brier']:>8.3f}")
    print(f"{'after  (T=%.2f)' % T:16}{after['ece']:>8.3f}{after['mce']:>8.3f}"
          f"{after['nll']:>8.3f}{after['brier']:>8.3f}")

    print("\nselective prediction (before scaling):")
    _selective_table(before["conf"], before["correct"])
    print("\nselective prediction (after scaling):")
    _selective_table(after["conf"], after["correct"])

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    _reliability_panel(axes[0], before, f"{args.split} — raw")
    _reliability_panel(axes[1], after, f"{args.split} — T={T:.2f}")
    fig.suptitle("Reliability: bar = accuracy in bin, red = gap to confidence",
                 fontsize=10)
    fig.tight_layout()
    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=130)
    print(f"\nwrote {args.figure}")


if __name__ == "__main__":
    main()
