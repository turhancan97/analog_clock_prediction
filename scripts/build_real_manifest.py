#!/usr/bin/env python3
"""Build `real_data/real_manifest.csv` -- the canonical list of real clock
photos with ground-truth times, rounded to the 144-class 5-minute grid, and a
deterministic train/test split stratified by hour-of-day.

Two sources, both gitignored (images are large); the manifest itself is small
and committed so the split is reproducible without re-downloading:

  1. kongaskristjan/real-clocks   real_data/{train,val}/<hash>_<HH>_<MM>.jpg
  2. vctorsuarezvara/real-images-of-analogclocks
       real_data/vctorsuarezvara/Images/Images/<i>.jpg  +  label.csv
       (label.csv has no header; row i is "<hour>,<minute>" for <i>.jpg)

The split is by hour bucket, not by class -- ~195 photos across 144 classes is
well under one per class, so per-class stratification is impossible. TEST_FRAC
of each hour bucket is held out; that split is never trained on (see
train.py --real-mix, which only ever pulls split == "train").

    python scripts/build_real_manifest.py [--test-frac 0.3] [--seed 0]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import clockmodel as cm

KONGAS_RE = re.compile(r"^[0-9a-f]+_(\d{1,2})_(\d{1,2})\.jpg$")


def kongaskristjan_rows():
    rows = []
    for sub in ("train", "val"):
        for p in sorted((cm.REAL_DATA_DIR / sub).glob("*.jpg")):
            m = KONGAS_RE.match(p.name)
            if not m:
                print(f"  skip (unparseable): {p.name}")
                continue
            rows.append((p, int(m.group(1)), int(m.group(2)), "kongaskristjan"))
    return rows


def vctorsuarezvara_rows():
    base = cm.REAL_DATA_DIR / "vctorsuarezvara"
    label_csv = base / "label.csv"
    img_dir = base / "Images" / "Images"
    if not label_csv.exists() or not img_dir.exists():
        print(f"  vctorsuarezvara source not present ({base}) -- skipping")
        return []
    labels = pd.read_csv(label_csv, header=None, names=["hour", "minute"])
    rows = []
    for i, (hour, minute) in enumerate(zip(labels["hour"], labels["minute"])):
        p = img_dir / f"{i}.jpg"
        if not p.exists():
            print(f"  skip (no image for label row {i}): {p.name}")
            continue
        rows.append((p, int(hour), int(minute), "vctorsuarezvara"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = kongaskristjan_rows() + vctorsuarezvara_rows()
    if not raw:
        sys.exit("no real photos found under real_data/ -- see module docstring")

    records = []
    for path, hour, minute, source in raw:
        label = cm.nearest_5min_label(hour, minute)
        assert label in set(cm.class_names()), f"{label} not a class ({hour}:{minute})"
        records.append(dict(
            path=str(path.relative_to(REPO_ROOT)), source=source,
            hour=hour, minute=minute, label=label,
            hour_bucket=int(label.split("-")[0]),
        ))
    df = pd.DataFrame(records).sort_values("path").reset_index(drop=True)

    # deterministic per-hour-bucket holdout
    rng = np.random.default_rng(args.seed)
    df["split"] = "train"
    for _, grp in df.groupby("hour_bucket"):
        n_test = max(1, round(len(grp) * args.test_frac)) if len(grp) > 1 else 0
        test_idx = rng.choice(grp.index.to_numpy(), size=n_test, replace=False)
        df.loc[test_idx, "split"] = "test"

    df = df.drop(columns="hour_bucket")
    cm.REAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cm.REAL_MANIFEST, index=False)

    n_tr, n_te = (df["split"] == "train").sum(), (df["split"] == "test").sum()
    print(f"\nwrote {cm.REAL_MANIFEST}  ({len(df)} photos: {n_tr} train / {n_te} test)")
    print(df.groupby(["source", "split"]).size().unstack(fill_value=0))
    print(f"\ndistinct classes covered: {df['label'].nunique()} / {cm.NUM_CLASSES}")


if __name__ == "__main__":
    main()
