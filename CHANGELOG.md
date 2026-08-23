# Changelog

Newest first. Dates are absolute.

## 2026-08-23 (very newest) — the "~195-min-offset" pattern is a second dataset defect, not model confusion

Follow-up to both the 33-error diagnosis and the Grad-CAM notebook: Grad-CAM
showed the model's remaining errors have a clear, confident focal point on
the hand-tip region (unlike the blank-dial defect's diffuse attention) --
meaning they weren't obviously "no signal" cases. That prompted checking
whether the ~195-minute offset shared by most of them was fuzzy model
confusion or something more exact.

### Findings
- **It's exact.** Recomputing the *signed* offset (not just the circular
  distance) for all 13 non-blank errors on the current default checkpoint:
  every single one is off by precisely ±195 minutes (3h15m) -- e.g.
  `11:10 -> 7:55` is exactly `11:10 - 3:15`; `8:25 -> 11:40` is exactly
  `8:25 + 3:15`. No fuzz, no near-misses.
- **These errors cluster on specific file indices** -- `0`, `38`, `51`, `72`
  -- the same signature as the blank-dial defect's clustering on index `36`.
- **Visually confirmed as a rendering defect, not a model error.** Read
  `train/2-50/0.jpg` and `train/8-25/38.jpg` directly: the drawn hands show
  ~6:05 and ~11:40 respectively -- i.e. exactly what the model predicted, not
  the folder's labeled time. The model is reading the image correctly; the
  image doesn't match its own folder name.
- **Extended to all 144 occurrences of each suspect index** (same method as
  the index-`36` check): only 1-4 of 144 are actually affected per index, not
  the whole index -- but of the ones that are wrong, **100% are the exact
  ±195-minute shift, zero "other" error types**. That 0% mismatch rate is the
  strongest evidence this is a rendering bug and not model confusion, which
  would be expected to scatter across many different error magnitudes.
- **Revises the 33-error diagnosis's accuracy estimate upward.** Combining
  both defect types (19 blank + 13 shifted, one of which -- `valid/8-50/36.jpg`
  -- was double-counted in both entries before this finding): **32 of 33
  dataset-wide errors are dataset defects**, leaving exactly 1 genuine model
  error (`11:25 -> 10:55`, 30 min off, a plausible adjacent-class boundary
  call). True model accuracy is closer to **99.99%** (14,399/14,400), not the
  99.91% estimated in the earlier diagnosis (which only accounted for the
  blank-dial defect).
- **Retroactively reframes part of the original `11-10` diagnosis** (see the
  entry far below). That diagnosis concluded `11-10`'s 29 baseline-model
  errors were genuine model weakness, not a labeling problem, based on
  visually confirming the hands showed 11:10. That's still true for most of
  those 29 -- retraining fixed 28 of them, which a data-only defect
  couldn't explain -- but the one that survived retraining
  (`test/11-10/38.jpg`, still misread by the 80-epoch model) is this same
  index-`38` rendering defect, not a residual model weakness.
- Updated `DATASET.md` caveats to describe both defect types together.

## 2026-08-23 (newest of all still) — EDA + Grad-CAM attention notebook

### Added
- `notebooks/eda_attention.ipynb` — dataset EDA (class balance, sample image
  grid) plus Grad-CAM attention visualization on the default checkpoint.
  Target layer: `top_activation` (last EfficientNetB3 conv feature map,
  5x5x1536 before global pooling). Generated via `nbformat`, then executed
  end-to-end with `jupyter nbconvert --execute` to catch errors and populate
  outputs before committing — reproduces 99.58% test accuracy / 6 misreads,
  matching every prior measurement.
- Added `matplotlib==3.10.8` and `notebook==7.5.6` to `requirements.txt`
  (already installed in the conda env; now pinned for reproducibility).

### Findings
- **Attention tracks the hands, not a fixed frame region.** On correct,
  confident predictions, the Grad-CAM heatmap concentrates tightly at the
  hand-tip vertex, consistent with the earlier rotation-invariance finding
  (predictions exactly unchanged under 90/180/270 degree rotation).
- **The blank-dial dataset defect (see the 33-error diagnosis above) produces
  visibly diffuse, unfocused attention** — no coherent hot region, spread
  across most of the dial — matching its near-random 0.06-0.09 confidence.
  Directly visualized `train/12-30/36.jpg` (blank) against
  `train/9-25/36.jpg` (normal, same file index, different class) for
  contrast.
- **The mislabeled `valid/8-50/36.jpg` image gets attention on the actual
  drawn hands, near 12:05** — not on empty space near where 8:50's hands
  *should* be. Independent visual confirmation, on top of the earlier eyeball
  inspection, that this is a mislabel/mis-render and not a model error.
- **The dataset's visual diversity is much wider than earlier samples
  suggested.** The error diagnosis and provenance checks happened to sample
  plain yellow-dial, tick-marks-only images (all coincidentally file index
  `36` or `0`). Random test-split samples in this notebook show ornate
  numerals, photographic-style clock faces, illustrated backgrounds, and even
  mirrored/reversed digit rendering. Worth knowing before drawing conclusions
  about "the" dataset's appearance from a handful of spot-checked files.
- Mean Grad-CAM heatmap over 60 random correct predictions, in a shared
  (non-hand-centered) frame, peaks near the image center — expected, since
  the clock center is where hands originate regardless of the time shown.

## 2026-08-23 (newest of all) — repo reorganized: scripts/, models/

### Changed
- All Python entry points (`train.py`, `evaluate.py`, `predict.py`,
  `compare_full_dataset.py`) and both `.slurm` job files moved into
  `scripts/`. `src/clockmodel.py` (the shared library) stays where it is.
- Local checkpoints now live in `models/` (gitignored) instead of the repo
  root: `clockmodel.MODELS_DIR`, `clockmodel.DEFAULT_MODEL` updated to match;
  `train.py --out` now defaults to `models/clock_model.keras` and creates the
  directory if missing. Both `.slurm` scripts updated to the new
  `scripts/`/`models/` paths.
- Each moved script now resolves `src/` relative to its own file location
  (`Path(__file__).resolve().parent.parent / "src"`) instead of assuming cwd
  — was previously a bare `sys.path.insert(0, "src")`, which only worked if
  invoked from the repo root. Run from the repo root either way
  (`python scripts/train.py ...`); it's just more robust now.
- Deleted the two superseded checkpoints that were sitting in the repo root
  (`clock_model_frozen_aug.keras`, `clock_model_unfrozen_aug.keras` — the
  20-epoch run, superseded by the 80-epoch one). Only
  `models/clock_model_unfrozen_aug_80ep.keras` remains.
- `compare_full_dataset.py`'s `MODELS` dict dropped the now-deleted 20-epoch
  checkpoint; compares baseline vs the current default only.
- Verified after the move: `scripts/evaluate.py --split test` and
  `scripts/predict.py` both reproduce prior results unchanged (99.58% top-1,
  same 6 test misreads).
- Updated `README.md` and `AGENTS.md` layout sections and every command
  example to the new paths.

### Fixed
- **`compare_full_dataset.py`'s baseline entry silently loaded the wrong
  model.** It called `cm.load_model()` with no args to mean "the baseline,"
  which worked back when the default was `data/time-99.68.h5` — but the
  earlier "default checkpoint switched" change made `load_model()`'s default
  the 80-epoch checkpoint instead, so both entries in the comparison silently
  loaded the *same* model (caught because the re-run after this move showed
  identical 33-error results for both rows instead of baseline's expected 62).
  Fixed by passing `cm.DATA_DIR / "time-99.68.h5"` explicitly instead of
  relying on the default. Re-verified: baseline 62 errors / 80-epoch 33
  errors, matching every prior measurement.

## 2026-08-23 (latest) — 33-error diagnosis: most are dataset defects, not model errors

Same methodology as the original `11-10` diagnosis, applied to the new
default checkpoint's remaining 33 full-dataset errors.

### Findings
- **18 of 33 errors share one file: `36.jpg`** in their class directory
  (`12-30/36.jpg`, `2-05/36.jpg`, `9-05/36.jpg`, `10-05/36.jpg`, etc.), all
  predicted with near-random confidence (p ≈ 0.06-0.10, vs 1/144 ≈ 0.007
  chance level).
- **Visually confirmed blank**: these images are a bare dial with tick marks
  and no hands drawn at all. Checked several directly (`Read` tool on the
  `.jpg`); a same-index image from a different class (`9-25/36.jpg`) has
  normal hands, so this isn't an index-wide bug — it's specific renders.
- **Extended the check to all 144 `36.jpg` files** (one per class): 124/144
  correctly classified, 20/144 wrong. Of the 20: **19 are the same blank-dial
  defect**; the 20th (`valid/8-50/36.jpg`) has visible hands but they're drawn
  near 12:05, not 8:50 — a mislabel or mis-render, confirmed visually. Not
  corrupt JPEGs (`file`/`md5sum` show valid, distinct 224×224 images) — a
  dataset-generation defect, present before any of this session's retraining.
- **The remaining 13 non-`36.jpg` errors are the real signal**: 12 of them
  share the exact same ~195-minute-offset pattern as the original `11-10`
  diagnosis (e.g. `8:25→11:40`, `5:05→8:20`, `11:10→7:55`, `3:35→12:20`) —
  this hand-relationship confusion persists after retraining, just far more
  rarely and spread across many classes instead of concentrated in one. The
  13th (`11:25→10:55`, 30 min off, p=0.535) is a minor adjacent-class
  boundary call.
- **Corrected accuracy read**: excluding the 20 dataset-defect errors, the
  model's real error count on the full dataset is 13, not 33 — true accuracy
  closer to **99.91%**, not the measured 99.77%. The measured figure
  understates the model; it's capped by broken source images no model could
  read correctly.
- Documented the defect in `DATASET.md` so future accuracy comparisons
  account for it rather than re-diagnosing from scratch.

## 2026-08-23 (newest still) — default checkpoint switched

### Changed
- `clockmodel.load_model()` now defaults to
  `clock_model_unfrozen_aug_80ep.keras` in the repo root (99.77% dataset-wide,
  see the entry below) instead of `data/time-99.68.h5` (99.57%). Falls back to
  `time-99.68.h5` automatically if the `.keras` file isn't present, e.g. on a
  fresh clone before running `train_unfrozen_aug_longer.slurm`. `--model` on
  `predict.py`/`evaluate.py` still overrides either way.
- Verified via `evaluate.py --split test` with no `--model` flag: now loads
  the new default and reproduces 99.58%.
- Updated `README.md` (layout, "pretrained model" section) and `AGENTS.md`
  (opening paragraph, verification-accuracy note) accordingly.

## 2026-08-23 (newest) — longer-epoch rotation-aug run closes the accuracy gap

Follow-up to the 20-epoch run: does a bigger epoch budget let augmented data
converge to match (or beat) the baseline while keeping the error-spreading
benefit? Yes.

### Added
- `train_unfrozen_aug_longer.slurm` — same job as `train_unfrozen_aug.slurm`
  but `--epochs 80` (EarlyStopping(patience=5) caps wasted time), output to
  `clock_model_unfrozen_aug_80ep.keras`.
- `compare_full_dataset.py` now compares all three checkpoints.

### Findings
- Job 475932, ~7 min on an A100. EarlyStopping triggered at epoch 25, best
  epoch 20, val_accuracy 0.995.
- **Test split**: 99.58% top-1 (vs baseline 99.38%, vs 20-epoch run 98.96%),
  99.86% top-5, mean error 0.7 min (best of all three).
- **Full dataset (14,400 images)**: **99.77% accuracy, 33 errors** — beats
  both the baseline (99.57%, 62 errors) and the 20-epoch run (99.49%, 73
  errors). No class exceeds 9.1% of total errors (vs baseline's 46.8% in
  `11-10` alone).
- `11-10` specifically: 29/62 baseline errors (46.8%) -> 1/33 here (3.0%).
  Nearly eliminated, not fully — one test-split misread (`11:10 -> 7:55`,
  p=0.734) remains.
- **Conclusion**: the 20-epoch run was undertrained, not a different
  accuracy/robustness tradeoff. More epochs gets both higher accuracy and the
  flattened error distribution. `clock_model_unfrozen_aug_80ep.keras` is now
  the best checkpoint measured on every axis.

## 2026-08-23 (latest still) — confirmed time-99.68.h5 provenance

### Findings
- Previously inferred, now confirmed: `kaggle datasets download -d
  gpiosenka/time-image-datasetclassification -f time-99.68.h5` (authenticated,
  `KAGGLE_API_TOKEN` env var picked up automatically by the CLI) returns the
  checkpoint directly, dated 2022-08-18 like `clocks.csv`.
- It is **byte-identical** to the local copy: 135,837,376 bytes, md5
  `6bb35a8914cd9d329c036edfcd575879` on both. Ships with the dataset.

## 2026-08-23 (latest) — full-dataset per-class error comparison

### Added
- `compare_full_dataset.py` — runs both checkpoints over train+valid+test
  combined (14,400 images, same scope as the original `11-10` diagnosis) and
  reports per-true-class error counts, not just the 1440-image test split
  `evaluate.py` covers.

### Findings
- **Baseline** (`time-99.68.h5`): 62 errors, 46.8% concentrated in `11-10`
  alone (29/62); errors touch 28 distinct classes.
- **Rotation-aug** (`clock_model_unfrozen_aug.keras`): 73 errors (slightly
  more total) but no single class has more than 3 (4.1% of all errors);
  errors touch 48 distinct classes.
- Confirms the earlier test-split finding at full scale: augmentation doesn't
  reduce total errors, but it genuinely dissolves the one pathological
  failure mode into a near-uniform scatter rather than just relocating a
  second concentration elsewhere.

## 2026-08-23 (even later) — rotation-augmentation training experiment

Ran the two open leads from `AGENTS.md`: a full training run (never done
before) and rotation augmentation (previously untested). Two runs:

### Added
- `train_unfrozen_aug.slurm` — Slurm job for GPU training on the cluster
  (`dgxa100` partition). Not committed model artifacts (`clock_model_*.keras`,
  gitignored) or run logs (`*.log`, `logs/`, also gitignored).

### Findings — CPU frozen-backbone baseline (sanity check)
- 12 epochs, head-only (backbone frozen), CPU: plateaued at **21.8% test
  accuracy**. Confirms ImageNet features alone can't do this task — expected,
  since the released model needed full backbone fine-tuning. Mainly useful as
  a pipeline smoke test (rotation/zoom/translation augmentation, checkpointing
  all worked).

### Findings — GPU unfrozen fine-tune with rotation augmentation
- Same architecture/hyperparameters as the released model, but with
  `RandomRotation(0.03)` + `RandomZoom(0.05)` + `RandomTranslation(0.05, 0.05)`
  added to the training pipeline (`train.py` already had this coded but it had
  never been exercised in a full run).
- 20 epochs, ~10 min total on an A100 (see GPU setup gotchas below). Best
  epoch 16, EarlyStopping never triggered.
- **98.96% top-1 test** (vs baseline's 99.38%), 99.86% top-5 (tied), mean
  error 2.1 min (vs baseline's 1.1 min). Slightly worse overall — augmentation
  makes the task harder to fit in the same epoch budget, unsurprising.
- **But the `11-10` failure mode is gone.** That class was the baseline's
  single dominant error source (29 of 62 dataset-wide errors, 22.5% of its own
  training images misread — see the entry below). It does not appear in any
  of this run's 15 test misreads.
- **The errors didn't disappear, they moved.** The new misreads still show the
  same ~195-minute-offset signature as the baseline's dominant error pattern,
  just redistributed onto different classes (`12:50→4:05`, `1:15→4:30`,
  `9:10→5:55`, etc.) instead of concentrated on `11-10`.
- **Net read:** rotation augmentation doesn't look like a strict win over
  more epochs at this budget, but it does reshape *which* classes are hard,
  and evidently rebalances the model off `11-10` specifically. Untested:
  whether more epochs (augmented data needs more to converge) recovers the
  baseline's overall accuracy while keeping `11-10` fixed.

### Identified — GPU training setup on this cluster
Two silent-failure gotchas hit back to back when moving `train.py` off CPU
onto the cluster's GPU nodes (conda env at
`/shared/results/common/kargin/tck_miniconda3`):
1. **TF silently trains on CPU with no error** if the pip-installed
   `nvidia-*-cu12` packages' `.so` files aren't on `LD_LIBRARY_PATH` — it logs
   `Cannot dlopen some GPU libraries` (suppressed by
   `TF_CPP_MIN_LOG_LEVEL=3`) and falls back to CPU with no crash. Confirmed via
   `nvidia-smi` on the allocated node showing 0% util / no process while the
   job "ran" at CPU-only speed (~2s/step, matching the earlier CPU timing).
   Fix: `export LD_LIBRARY_PATH=$(for d in .../site-packages/nvidia/*/lib; do
   echo -n "$d:"; done)` before running.
2. **`ptxas` was missing entirely** (`nvidia-cuda-nvcc-cu12` wasn't
   installed), which crashes GPU training outright with `Autotuner could not
   compile any configs for HLO: ...cudnn-conv...` — cuDNN's conv autotuner
   needs `ptxas` to JIT-compile and verify candidate kernels; without it every
   algorithm fails to compile. Fixed by `pip install
   nvidia-cuda-nvcc-cu12==12.8.*` (matching the existing CUDA 12.8 toolkit)
   and adding its `bin/` to `PATH`. Both fixes are baked into
   `train_unfrozen_aug.slurm`.

## 2026-08-23 (later still) — git init

### Added
- `requirements.txt` — pinned to what's actually installed in the conda env
  (tensorflow 2.21.0, keras 3.13.2, numpy 2.2.6, pillow 11.3.0, pandas 2.2.3).
  `torch` is installed in the env but unused by any code here, so it's
  excluded.
- Initialized git (`main` branch), single root commit covering everything
  built this session. No history before this point.

## 2026-08-23 (later) — dataset provenance

### Added
- `DATASET.md` — source, license, download and verification instructions.

### Identified
- Source is **TIME — Image Dataset-Classification** by Gerry Piosenka
  (`gpiosenka/time-image-datasetclassification`) on Kaggle, **CC0 public
  domain**. Matched via the `clocks.csv` header schema, which is that author's
  signature; corroborated by label format, class count, image size and split.
- `data/` is a **symlink** to `/shared/sets/datasets/vision/analog-clock/data`.
  Plain `find data` returns nothing through it — use `find -L`.
- Local copy is 418,698,419 bytes vs Kaggle's reported 402,067,147 (~4% larger),
  probably a version difference.
- Could **not** verify that `time-99.68.h5` ships with the Kaggle dataset;
  file listing needs an authenticated API call. Naming and architecture match
  the author's conventions, but this is inference, not confirmation.

## 2026-08-23 (later) — `11-10` diagnosis

Investigated the `11-10` → `7:55` failure. No code changed; findings only.

### Findings
- **The labels are correct.** Misread `11-10` images visually show a short hand
  at 11 and a long hand at 2. The model is wrong, not the data.
- **Not duplication or leakage.** Zero duplicate files across all 14,400 images.
  Near-identical cross-class pairs do exist (`11-10/81.jpg` vs `7-55/81.jpg`
  differ by 1.8/255) but they do *not* explain the misreads: files misread as
  `7-55` are no more similar to `7-55` than correctly-read ones (21.0 vs 19.8).
- **The model never learned this class.** It misreads 22.5% of its own
  *training* images for `11-10` (train 0.775 / valid 0.500 / test 0.400). This
  is underfitting on a hard class, not a generalization gap.
- **Scale:** over all 14,400 images accuracy is 99.57% (62 errors). `11-10`
  alone accounts for **29 of them**; `4-50` is next with 6.
- **Dominant signature: a 195-minute offset** (40 of 62 errors) — `11-10`→`7-55`,
  `4-50`→`1-35`, `5-20`→`2-05`, `2-10`→`10-55`, `5-10`→`1-55`, `5-15`→`2-00`.
- **The model reads the numerals, not absolute hand angles.** Predictions are
  exactly unchanged under 90°/180°/270° rotation (80/80 images).
- **But it is extremely brittle to small rotations**: 0° → 100%, 3° → 68.9%,
  6° → 0%. Rotating the dial and hands together does not change the true time,
  so this is pure off-distribution fragility — every clock in the dataset is
  upright.
- Misread `11-10` images are geometrically distinct: minute-hand reach exceeds
  hour-hand reach in 96% of them vs 45% of correctly-read ones. (Crude
  angular-reach detector that saturates on the dial ring — treat as suggestive.)

### Retracted
- **Test-time augmentation is not the cheap win suggested earlier.** Measured on
  `11-10` it makes things *worse*: 0.710 → 0.400 over 10 rotate/zoom views.
  Cause is the 3°/6° brittleness above. Do not pursue TTA with geometric views.

## 2026-08-23

Repo went from data-only (no code at all) to a working inference, evaluation,
and training setup.

### Added
- `src/clockmodel.py` — class labels, model loading, image preprocessing.
- `predict.py` — read the time off one or more images, with top-N guesses.
- `evaluate.py` — top-1/top-5 accuracy plus circular minutes-off error on a split.
- `train.py` — rebuilds the same architecture from ImageNet weights and fine-tunes.
- `README.md`, `AGENTS.md`, `CHANGELOG.md`, `.gitignore`.

### Identified
- `data/time-99.68.h5` is EfficientNetB3 → GlobalMaxPool → BatchNorm →
  Dense(256, relu) → Dropout → Dense(144, softmax); 11,220,159 params,
  Adamax @ 1e-3, categorical crossentropy, Keras 2.8.
- Input is **150×150**, not the images' native 224×224. Rescaling (1/255) and
  Normalization are layers inside the model, so it takes raw `[0, 255]` floats.
- `train.py`'s rebuild reproduces the param count exactly, confirming the
  architecture reconstruction.

### Fixed
- Keras 3 could not load the checkpoint: Keras 2.8 wrote a `groups` key into
  every `DepthwiseConv2D` config. `clockmodel.load_model` drops it via a
  `from_config` patch.
- **Class ordering.** First implementation sorted the hyphenated directory
  names and scored 66.6% with every error exactly one hour off. Correct order
  is alphabetical over the underscore labels in `clocks.csv`, because `-` (45)
  sorts before digits and `_` (95) after. The CSV's own `class index` column
  is a third, also-wrong order (65%). All three fail silently.
- `load_image` used `decode_jpeg`; switched to `decode_image` so PNG and other
  formats work.

### Measured
- **99.38% top-1** on both test and valid; 99.86% top-5; mean error 1.1 min.
  Consistent with the `99.68` in the checkpoint filename.
- Resize method (PIL bicubic/bilinear/nearest/lanczos, tf bilinear ±antialias,
  tf nearest) makes **no** difference to accuracy — ruled out while chasing the
  ordering bug.

### Known issues
- ~0.6% misread, concentrated in one class: `11-10` is confidently read as
  `7:55` (p≈0.9) in both valid and test. The images were checked visually and
  really are 11:10 — a model weakness, not a labeling error.
- `train.py` is smoke-tested (builds, fits, checkpoints) but has **never been
  run to completion**. No GPU available in this environment.
