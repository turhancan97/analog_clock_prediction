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
  clock_model_rot54_s0.keras            default (see below): ±54° rotation-aug, 99.82% dataset
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

`clockmodel.load_model()` defaults to `models/clock_model_rot54_s0.keras` if
present, falling back through `clock_model_unfrozen_aug_80ep.keras` to
`data/time-99.68.h5` (e.g. on a fresh clone before you've trained your own).
Pass `--model` to any script to override. All commands below assume you're
running from the repo root.

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

`clock_model_rot54_s0.keras` (default, locally trained with ±54° rotation
augmentation, `--rotation-factor 0.15`): **99.72%** top-1 test (99.86% top-5,
mean error 0.5 min); on the full 14,400-image dataset, **99.82%** (26 errors)
vs the released model's 99.57%, errors spread across many classes. It is also
**rotation-robust across the full ±90° test-time sweep** — see the rotation
section below and `CHANGELOG.md` (2026-08-28). The ±10.8° predecessor
`clock_model_unfrozen_aug_80ep.keras` scored 99.58% test / 99.77% dataset
(`CHANGELOG.md` 2026-08-23).

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

### Known failure mode

Of the default model's 26 dataset-wide errors, almost all are two dataset
rendering defects — a blank dial with no hands, and hands drawn at exactly
±3h15m from the folder's labeled time — see `DATASET.md` caveats. Real
accuracy is closer to **99.97%**. (The ±10.8° predecessor had 33 errors,
32 of them the same defects.)

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

## Generalization to real photos: it does not

Every accuracy figure above is on synthetic, upright, centered,
exact-5-minute-mark renders. Tested against 92 real clock photos
(`kongaskristjan/real-clocks`, CC0) with `scripts/evaluate_real_photos.py`:
**5.4% top-1 accuracy** (vs. 99.7% synthetic), mean error 176 minutes, mean
confidence on its own top-1 guess just 0.205. Errors are spread across every
error magnitude rather than one clean systematic offset — this looks like
genuine confusion on out-of-distribution input, consistent with the
rotation-brittleness finding above, not a single fixable bug. See
`DATASET.md` ("Real-photo test set") for how to reproduce and
`CHANGELOG.md` (2026-08-24) for the full diagnosis. (Measured on the
±10.8° checkpoint; worth re-running against the ±54° default now that
rotation robustness is much wider — see future work.)

![Real-photo predictions: six best and six worst](docs/images/real_photo_predictions.png)

## Rotation robustness follows the training augmentation range

`scripts/characterize_rotation_brittleness.py` sweeps test-time rotation
(-90° to +90°) over 144 test images. Each checkpoint holds a flat ~99%
accuracy plateau across roughly its **literal training `RandomRotation`
range**, then falls off a cliff into a high-confidence 0%-accuracy dead zone.
`scripts/rotation_range_sweep.py` measured this across four training ranges
(`CHANGELOG.md` 2026-08-28):

| training aug | upright dataset acc | ≥95% robust plateau |
|---|---|---|
| ±10.8° (`f0.03`) | 99.71% | [−10°, +10°] |
| ±21.6° (`f0.06`) | 99.71% | [−20°, +30°] |
| ±36.0° (`f0.10`) | 99.67% | [−30°, +90°] |
| **±54.0° (`f0.15`, current default)** | **99.82%** | **[−90°, +90°] — the whole sweep** |

Widening the augmentation range widens the robust plateau to match, **at no
upright-accuracy cost** — the ±54° model is both the most accurate and the
most rotation-robust checkpoint trained. The released no-aug checkpoint has
no plateau at all.

![Does wider rotation aug widen the robust plateau?](docs/images/rotation_range_sweep.png)

On the narrow-augmentation checkpoints, confidence stays high (0.4–0.99)
throughout the dead zone — the model doesn't know it's wrong — and Grad-CAM
stays sharply focused on the (rotated) hands even on confidently-wrong
predictions, unlike the diffuse attention seen on the blank-dial dataset
defect above. This is confident *misreading* of a correctly-located hand
position, not lost localization. See `CHANGELOG.md` (2026-08-24, 2026-08-28).

![Grad-CAM at increasing rotation angles: attention stays hand-focused even when wrong](docs/images/rotation_gradcam.png)

## Testing

```bash
python -m pytest tests/ -v
```

`tests/test_clockmodel.py` has two tiers. Pure-logic tests (label formatting,
prediction decoding, the Keras 2.8→3 `DepthwiseConv2D` compat patch) always
run. Data/model-backed tests — locking down `clockmodel.class_names()`'s
ordering against both known-wrong alternatives, and a test-accuracy
regression threshold against the default checkpoint — need `data/` and/or a
trained checkpoint and skip automatically when those aren't present (e.g. in
CI, which has neither since both are gitignored).

CI (`.github/workflows/ci.yml`) runs this suite on every push/PR to `main`,
so it only ever exercises the pure-logic tier; the data/model-backed tests
are for local verification, per `AGENTS.md`'s "verify, don't assume"
guidance.

## Future work

- **Close the real-photo gap.** Now measured, not just predicted (see
  "Generalization to real photos" above): 5.4% top-1 on real clock photos vs.
  ~99.7% synthetic — **though that was measured on the ±10.8° checkpoint;
  re-run `scripts/evaluate_real_photos.py` against the ±54° default first**,
  since much wider rotation robustness is exactly the kind of thing that
  might move it. The 92-image sample here has no angle/lighting/dial-style
  labels, so it can't isolate which factor dominates the failure — a bigger,
  labeled real-photo set (`vctorsuarezvara/real-images-of-analogclocks` is
  untried) would help. Likely still needs its own data collection or
  realistic augmentation (perspective, lighting, hand-drawing variance)
  rather than more epochs on the current synthetic set.
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
- ~~**Widen the rotation-augmentation range.**~~ **Done (2026-08-28).**
  Training at `--rotation-factor 0.15` (±54°) widens the robust plateau to
  the entire ±90° sweep at no upright-accuracy cost (in fact the best
  checkpoint on every axis, 99.82% dataset-wide). This is now the default;
  `clock_model_rot54_s0.keras` is the default checkpoint. Untested past
  ±54° and against real photos (see first item).
- **Confidence calibration.** All accuracy figures so far are top-1/top-5,
  not calibration — whether p=0.9 actually means ~90% correct. Relevant if
  the confidence score is ever used to decide when to trust a prediction vs.
  fall back to a human (e.g. for the real-photo axis above).
- **Model size / deployment.** EfficientNetB3 at 150×150 is a fairly heavy
  backbone for a 144-way classification task this constrained. A backbone
  ablation (2026-08-27, `scripts/train_backbone_ablation.slurm`, see
  `CHANGELOG.md`) found **EfficientNetB0 costs ~0.4–0.5 pt dataset-wide
  accuracy for 2.5× fewer params and 2× the CPU speed** —
  `train.py --backbone efficientnetb0`. MobileNetV3-Small and a 0.64M
  from-scratch CNN trade ~2–3 points of accuracy for 6–7× CPU speedup.
  B3 kept as default; seeded runs (`--seed 0`) reproduce the archived
  0.9977 baseline, so the earlier "recipe drift" was just unseeded variance.
  A **cyclic sin/cos regression head** (`--head circular`, the "better fit"
  idea) was tested and lost badly — ~21–28% exact-bucket accuracy, because
  rotation augmentation corrupts the absolute-angle target (see
  `CHANGELOG.md` 2026-08-27). Remaining: distillation; retry the regression
  head with rotation aug off + an angular loss; `--backbone resnet50v2` is
  currently broken.
