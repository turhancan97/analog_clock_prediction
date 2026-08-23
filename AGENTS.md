# AGENTS.md

Guidance for AI agents working in this repo. Read this before touching code.

## What this project is

Analog clock reading: a 224×224 clock-face image → one of 144 classes (every
5-minute increment on a 12-hour dial). `data/time-99.68.h5` is a pretrained
EfficientNetB3 checkpoint from Aug 2022 (Keras 2.8) that scores 99.38% on test.

Layout, usage, and model details live in `README.md` — read it too, don't
duplicate it here.

## Environment

- Python is conda at `/shared/results/common/kargin/tck_miniconda3/bin/python3`
  (3.12.8). TF 2.21 / Keras 3.13 / torch 2.10 are already installed.
- **CPU only** — no GPU was visible. Full training runs are slow; budget for it
  and don't kick one off without asking.
- Prefix TF commands with `TF_CPP_MIN_LOG_LEVEL=3` to cut the log noise.
- Not a git repo (as of 2026-08-23). Don't assume git history exists.
- **`data/` is a symlink** to `/shared/sets/datasets/vision/analog-clock/data`.
  Plain `find data ...` silently returns nothing — always `find -L data ...`.
  Provenance and re-download steps are in `DATASET.md`.

## Non-obvious things that will bite you

These are documented at length in `README.md`; the short version:

1. **Class order is alphabetical over the UNDERSCORE labels** from
   `clocks.csv` (`10_00, …, 1_00, …, 9_55`). Sorting the hyphenated directory
   names instead scores 66.6%; the CSV's `class index` column scores 65%.
   Both fail *silently*. Always use `clockmodel.class_names()`.
2. **Keras 3 cannot load the .h5 directly** — Keras 2.8 wrote a `groups` key
   into every `DepthwiseConv2D`. Use `clockmodel.load_model()`, which patches it.
3. **The model takes 150×150 raw `[0, 255]` floats.** Rescaling and
   Normalization are layers *inside* the model. Do not pre-scale to [0,1].

## Model behaviour worth knowing

- The model **reads the dial numerals**, not absolute hand angles — predictions
  are exactly invariant to 90°/180°/270° rotation.
- It is nonetheless **very brittle to small rotations**: 3° costs ~30 points of
  accuracy and 6° destroys it entirely, even though rotating a clock does not
  change the time it shows. Every image in the dataset is upright.
- Consequently **geometric test-time augmentation makes things worse**, measured.
  Don't reach for it. Rotation augmentation during *training* is still untested
  and is the most promising lead.
- The `11-10` class is the single biggest error source (29 of 62 dataset-wide
  errors), and the model misreads 22.5% of its own *training* images for it.
  The labels are correct. See `CHANGELOG.md` for the full diagnosis.

## Working agreements

- **Verify, don't assume.** Both class-ordering bugs above produced plausible
  code that ran fine and was badly wrong. Any change touching labels,
  preprocessing, or model loading must be checked with
  `python evaluate.py --split test` — expect ~0.9938.
- Report numbers you actually measured. Say so explicitly when something is
  untested (e.g. `train.py` has been smoke-tested but never fully run).
- Keep `README.md` (how to use it) and `AGENTS.md` (how to work on it) distinct.

## Maintenance

Update `CHANGELOG.md` at the end of any session that changes the repo, newest
first, with the date and what changed. Update this file when you learn
something a future session would otherwise have to rediscover the hard way.
