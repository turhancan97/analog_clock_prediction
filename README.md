# Analog Clock Reading

Reads the time off a 224×224 analog clock face image, as one of 144 classes
(every 5-minute increment on a 12-hour dial).

![Sample predictions across a random selection of classes](docs/images/sample_predictions.png)

## Layout

```
data/
  train/ valid/ test/   144 class dirs each ("3-10", "11-45"); 11520 / 1440 / 1440 JPGs
  clocks.csv            manifest: class index, filepaths, labels, data set
  time-99.68.h5         released pretrained EfficientNetB3, Keras 2.8, Aug 2022
models/                 locally trained checkpoints; gitignored
  clock_model_unfrozen_aug_80ep.keras   default (see below), better than time-99.68.h5
src/clockmodel.py       class labels, model loading, preprocessing
scripts/
  predict.py             read the time off given image(s)
  evaluate.py             accuracy + minutes-off error on a split
  train.py                retrain the same architecture from ImageNet weights
  compare_full_dataset.py compare checkpoints' per-class errors over the full dataset
  *.slurm                 GPU training jobs for this cluster
notebooks/
  eda_attention.ipynb     dataset EDA + Grad-CAM: where the model looks
```

`clockmodel.load_model()` defaults to `models/clock_model_unfrozen_aug_80ep.keras`
if present, falling back to `data/time-99.68.h5` otherwise (e.g. on a fresh
clone before you've trained your own). Pass `--model` to any script to
override. All commands below assume you're running from the repo root.

## Usage

```bash
python scripts/predict.py data/test/3-25/29.jpg --top 3
python scripts/evaluate.py --split test
python scripts/train.py --epochs 20 --out models/clock_model.keras
python scripts/evaluate.py --split test --model models/clock_model.keras
```

## The pretrained model

EfficientNetB3 → GlobalMaxPool → BatchNorm → Dense(256, relu) → Dropout →
Dense(144, softmax). 11,220,159 params, Adamax @ 1e-3,
categorical crossentropy. Both the released checkpoint and the locally
retrained ones below use this same architecture.

**Input is 150×150, not 224×224** — the images are 224², so everything gets
resized down. Rescaling (1/255) and Normalization are layers *inside* the model,
so feed it raw float `[0, 255]`; do not scale beforehand.

`time-99.68.h5` (released): **99.38%** top-1 on both test and valid (99.86%
top-5, mean error 1.1 min). Consistent with the `99.68` in the filename.

`clock_model_unfrozen_aug_80ep.keras` (default, locally trained with rotation
augmentation): **99.58%** top-1 test (99.86% top-5, mean error 0.7 min); on
the full 14,400-image dataset, 99.77% vs the released model's 99.57%, with
errors spread across many classes rather than concentrated in one. See
`CHANGELOG.md` (2026-08-23 entries) for the full experiment.

![Full-dataset accuracy by checkpoint](docs/images/checkpoint_accuracy.png)

### Two gotchas

**1. Loading it in Keras 3 fails out of the box.** Keras 2.8 wrote a `groups`
key into every `DepthwiseConv2D` config that Keras 3 rejects. `clockmodel.load_model`
passes a subclass whose `from_config` drops it.

**2. Class order is alphabetical over the UNDERSCORE labels**, i.e.
`10_00, 10_05, …, 1_00, …, 9_55` — the model was trained from the `labels`
column of `clocks.csv`. Two wrong-looking alternatives:

- sorting the hyphenated *directory* names gives a different order, because `-`
  (45) sorts before digits while `_` (95) sorts after. This silently drops
  accuracy to **66.6%**, with every error exactly one hour off.
- the `class index` column in `clocks.csv` is ordered by hour then minute and
  matches neither. It scores **65%**.

Always take labels from `clockmodel.class_names()`.

### Known failure mode (released model, `time-99.68.h5`)

~0.6% of images misread, and most of them are the same one: class `11-10` is
confidently (p≈0.9) read as `7:55` in both valid and test. Spot-checking the
images confirms they really are 11:10 — mostly a model weakness, not a bad
label (retraining fixes all but one instance of it — see `DATASET.md`, which
turned out to be a rendering defect, not the model).

### Known failure mode (default model, `clock_model_unfrozen_aug_80ep.keras`)

Of this model's 33 dataset-wide errors, only **1 is a genuine model
error** (`11:25 -> 10:55`, 30 min off). The other 32 are two dataset
rendering defects — a blank dial with no hands, and hands drawn at exactly
±3h15m from the folder's labeled time — see `DATASET.md` caveats. Real
accuracy is closer to **99.99%**.

![Dataset rendering defects: normal render, blank dial, and shifted hands](docs/images/dataset_defects.png)

## Where the model looks

`notebooks/eda_attention.ipynb` — dataset EDA (class balance, sample images
across the surprisingly wide variety of dial art styles) plus Grad-CAM
visualizations of the model's attention. Confirms visually, not just
statistically: attention concentrates tightly on the hand-tip region for
correct and confident predictions, including on the rendering-defect images —
which is what raised the question of whether those were really "model
confusion" and led to finding the ±3h15m-shift defect above. The blank-dial
defect produces diffuse, unfocused attention with low confidence instead; on
`valid/8-50/36.jpg`, attention locks onto the *actual drawn hands* (near
12:05), not the folder's claimed 8:50.

![Grad-CAM attention: correct prediction vs. blank-dial defect vs. shift defect](docs/images/gradcam_comparison.png)

## Future work

- **Automated tests.** There's no test suite — `evaluate.py --split test` is
  run by hand to catch regressions (per `AGENTS.md`'s "verify, don't assume").
  A regression test asserting accuracy against a fixed threshold, plus a unit
  test locking down `clockmodel.class_names()`'s ordering (the class-ordering
  trap has silently cost ~33% accuracy twice already), would catch both
  automatically.
- **CI.** No `.github/workflows` — nothing runs those tests, or even a lint/
  import smoke test, on push. Blocked on the first item existing.
- **Test on real (non-synthetic) clock photos** — a new axis of difficulty,
  not more tuning on this dataset. Everything measured so far is on
  synthetic, upright, centered, exact-5-minute-mark renders; `DATASET.md`
  already lists three untried candidates (`kongaskristjan/real-clocks`,
  `vctorsuarezvara/real-images-of-analogclocks`,
  `shivajbd/analog-clocks`). Given the rotation-brittleness finding (3°
  costs ~30 points, 6° destroys accuracy entirely), unposed real photos —
  off-axis viewing angles, non-upright hands, imprecise time settings,
  varied lighting — seem likely to break this model in ways the synthetic
  set can't reveal. Worth treating as its own investigation rather than
  squeezing more accuracy out of the current dataset.
- **Model card / public-facing writeup.** The findings here (rotation
  brittleness despite dial-reading invariance, the two dataset rendering
  defects and how Grad-CAM helped tell them apart from real model error, the
  `11-10` failure mode and its fix) are well-documented enough in
  `CHANGELOG.md` to make a solid blog post, Kaggle notebook, or standalone
  model card if this is ever shared publicly.
- **Confidence-based defect auto-flagging.** The two dataset defects found so
  far have distinct confidence signatures — the blank-dial defect is diffuse
  and near-random (p≈0.06–0.10), the ±3h15m shift defect is high-confidence
  but wrong. A script that flags low-confidence predictions and
  confidently-wrong predictions could auto-surface defects like these
  instead of hand-diagnosing per-class each time, and would generalize to
  auditing other synthetic datasets built the same way.
- **Characterize the rotation-brittleness cliff.** Measured (0°→100%,
  3°→~69%, 6°→0%) but not explained — why does a model that reads numerals
  invariant to 90°/180°/270° collapse at 6°? Worth knowing whether the
  numeral-reading mechanism only generalizes to axis-aligned crops/features
  or something else is going on, and whether it explains why training's
  `RandomRotation(0.03)` (≈±5.4°, right at the cliff edge) didn't raise
  overall accuracy despite fixing `11-10`.
- **Confidence calibration.** All accuracy figures so far are top-1/top-5,
  not calibration — whether p=0.9 actually means ~90% correct. Relevant if
  the confidence score is ever used to decide when to trust a prediction vs.
  fall back to a human (e.g. for the real-photo axis above).
- **Model size / deployment.** EfficientNetB3 at 150×150 is a fairly heavy
  backbone for a 144-way classification task this constrained. Every
  training run so far has varied epochs/augmentation, never architecture —
  untested whether a much smaller model, or one distilled from this one,
  holds accuracy if latency or deployment size ever matters.
