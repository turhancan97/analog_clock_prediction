# Changelog

Newest first. Dates are absolute.

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
