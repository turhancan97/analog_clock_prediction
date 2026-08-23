# Analog Clock Reading

Reads the time off a 224×224 analog clock face image, as one of 144 classes
(every 5-minute increment on a 12-hour dial).

## Layout

```
data/
  train/ valid/ test/   144 class dirs each ("3-10", "11-45"); 11520 / 1440 / 1440 JPGs
  clocks.csv            manifest: class index, filepaths, labels, data set
  time-99.68.h5         released pretrained EfficientNetB3, Keras 2.8, Aug 2022
clock_model_unfrozen_aug_80ep.keras   locally trained, better than the above (see below); gitignored
src/clockmodel.py       class labels, model loading, preprocessing
predict.py              read the time off given image(s)
evaluate.py             accuracy + minutes-off error on a split
train.py                retrain the same architecture from ImageNet weights
```

`clockmodel.load_model()` defaults to `clock_model_unfrozen_aug_80ep.keras` if
present in the repo root, falling back to `data/time-99.68.h5` otherwise (e.g.
on a fresh clone before you've trained your own). Pass `--model` to any script
to override.

## Usage

```bash
python predict.py data/test/3-25/29.jpg --top 3
python evaluate.py --split test
python train.py --epochs 20 --out clock_model.keras
python evaluate.py --split test --model clock_model.keras
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

### Known failure mode

~0.6% of images misread, and most of them are the same one: class `11-10` is
confidently (p≈0.9) read as `7:55` in both valid and test. Spot-checking the
images confirms they really are 11:10 — it's a model weakness, not a bad label.
