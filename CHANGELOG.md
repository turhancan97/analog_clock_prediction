# Changelog

Newest first. Dates are absolute.

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
