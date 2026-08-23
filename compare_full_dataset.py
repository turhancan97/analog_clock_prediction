#!/usr/bin/env python3
"""Per-class error comparison between two checkpoints over the FULL dataset
(train+valid+test combined), replicating the original 11-10 diagnosis scope
instead of the 1440-image test split evaluate.py normally covers."""
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "src")
import clockmodel as cm

MODELS = {
    "baseline (time-99.68.h5)": None,
    "unfrozen_aug (clock_model_unfrozen_aug.keras)": "clock_model_unfrozen_aug.keras",
}


def evaluate_full(model_path):
    names = cm.class_names()
    model = cm.load_model(model_path) if model_path else cm.load_model()

    y_true_all, y_pred_all = [], []
    for split in ("train", "valid", "test"):
        ds = cm.make_dataset(split, batch_size=64)
        probs = model.predict(ds, verbose=0)
        y_true = np.concatenate([y.numpy().argmax(1) for _, y in ds])
        y_pred = probs.argmax(1)
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return names, y_true, y_pred


def report(label, names, y_true, y_pred):
    n = len(y_true)
    wrong = np.flatnonzero(y_pred != y_true)
    acc = 1 - len(wrong) / n

    print(f"\n{'=' * 70}")
    print(f"{label}")
    print(f"{'=' * 70}")
    print(f"images         : {n}")
    print(f"accuracy       : {acc:.4f}  ({len(wrong)} errors)")

    true_err_counts = Counter(names[y_true[i]] for i in wrong)
    print(f"\nerrors by TRUE class (top 10 of {len(true_err_counts)} classes with any error):")
    for label_, count in true_err_counts.most_common(10):
        print(f"  {cm.label_to_time(label_):>6}  {count:3d} errors")

    max_count = true_err_counts.most_common(1)[0][1] if true_err_counts else 0
    top_share = max_count / len(wrong) if wrong.size else 0
    print(f"\nmost concentrated class accounts for {max_count}/{len(wrong)} errors ({top_share:.1%})")
    print(f"errors spread across {len(true_err_counts)} distinct true-classes "
          f"(out of {len(wrong)} total errors, {len(names)} classes)")

    return true_err_counts, wrong


results = {}
for label, path in MODELS.items():
    names, y_true, y_pred = evaluate_full(path)
    counts, wrong = report(label, names, y_true, y_pred)
    results[label] = counts

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
for label, counts in results.items():
    total = sum(counts.values())
    top = counts.most_common(1)[0] if counts else ("-", 0)
    print(f"{label}")
    print(f"  total errors: {total}, worst class: {cm.label_to_time(top[0])} "
          f"({top[1]} errors, {top[1]/total:.1%} of all errors)" if total else "  total errors: 0")
