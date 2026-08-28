# Changelog

Newest first. Dates are absolute.

## 2026-08-28 — widening the rotation-augmentation range (new default checkpoint)

README future work: "Widen the rotation-augmentation range." The brittleness
characterization (2026-08-24) showed the robust-angle plateau's edges track
the *literal* training `RandomRotation` range. Tested whether widening that
range widens the plateau, and what it costs.

### Method (job 479269, dgxa100, 1h11m)

EfficientNetB3, softmax, `--seed 0` (the recipe that reproduces the archived
baseline), trained at four `--rotation-factor` values, then swept over
test-time rotation with `scripts/rotation_range_sweep.py` (144 test images,
one per class).

### Results

| training aug | upright dataset top-1 (errors) | >=95% robust plateau |
|---|---|---|
| +/-10.8 deg (`f0.03`, old default) | 0.9971 (42) | [-10, +10] deg — cliff to ~0% by +/-30 deg |
| +/-21.6 deg (`f0.06`) | 0.9971 (42) | [-20, +30] deg |
| +/-36.0 deg (`f0.10`) | 0.9967 (48) | [-30, +90] deg |
| **+/-54.0 deg (`f0.15`)** | **0.9982 (26)** | **[-90, +90] deg — robust across the entire sweep** |

- **The plateau edge tracks the training range, confirming 2026-08-24** —
  and at +/-54 deg the model is rotation-robust everywhere tested.
- **It is free.** +/-54 deg gave the *best* upright accuracy of any run
  (0.9982 dataset / 26 errors), beating the previous default's 0.9977. The
  four residual test misreads are all known dataset defects (+/-195 / +/-280
  min, blank-dial p~0.07).
- Training cost flat: all four early-stopped at epoch 32–42, ~15 min each.
- Aliasing quirks, not the trend: the old default is near-perfect at
  *exactly* +/-90 deg (learned 90 deg symmetry) but 0% at +/-45 deg; the
  +/-21.6 deg model has a sharp dip at +/-45 deg.

### Changed

- **`DEFAULT_MODEL` is now `models/clock_model_rot54_s0.keras`** (the `f0.15`
  checkpoint, copied to a stable name). 0.9982 dataset / 0.9972 test, and
  rotation-robust to +/-90 deg. Falls back through
  `clock_model_unfrozen_aug_80ep.keras` to the released `time-99.68.h5` on a
  fresh clone.
- **`train.py --rotation-factor` default is now `0.15`** (was `0.03`).
- `tests/test_clockmodel.py` accuracy-regression threshold raised 0.99 ->
  0.995 (measured 0.9972 test); the test now decodes via
  `cm.output_to_class_idx`.
- `scripts/rotation_range_sweep.py` — new. Sweeps N checkpoints over test-time
  rotation, plots accuracy vs. angle, reports the >=95% plateau width per
  model. Figure: `docs/images/rotation_range_sweep.png`.
- `scripts/train_rotation_range.slurm` — new (job 479269).

### Not done

- `docs/images/rotation_cliff.png` and other figures still show the old
  default; regenerate when convenient.
- Whether this helps the real-photo gap (future work #1) is untested — the
  real-photo failure profile looked like more than rotation.

## 2026-08-27 — backbone ablation, step 1 of the architecture investigation

The "is EfficientNetB3 the right fit?" question (README future work #6).
Step 1 is a pure size/latency ablation: same classifier head, same 80-epoch
recipe, same augmentation — only the backbone changes. Step 2 (head reframe:
cyclic sin/cos regression vs. flat 144-way softmax) is not started.

### Results (job 478705, dgxa100, 41 min for all five)

| backbone | params | test top-1 | dataset top-1 (errors) | mean err (dataset) | CPU ms/img |
|---|---|---|---|---|---|
| efficientnetb3 (retrained control) | 11.2M | 0.9889 | **0.9961** (56) | 0.7 min | 71.7 |
| efficientnetb0 | 4.4M | 0.9882 | **0.9949** (73) | 0.9 min | 35.3 |
| mobilenetv3small | 1.1M | 0.9660 | 0.9871 (186) | 2.7 min | 11.7 |
| simplecnn (from scratch) | 0.64M | 0.9632 | 0.9882 (170) | 2.4 min | 10.0 |
| resnet50v2 | 24.1M | **0.006 — training failed** | 0.007 | 180 min | 50.6 |

- **EfficientNetB0 is a near-free 2.5× shrink.** Dataset-wide accuracy
  0.9949 vs B3's 0.9961 — a 0.12-point gap that is inside the dataset-defect
  noise floor (~32 of the ~56–73 errors are the known blank-dial / ±195-min
  rendering defects, not model errors). Half the params, 2× faster on CPU.
  (Superseded by the 2026-08-27 Step 2 results above: seeded runs put the B0
  gap at ~0.4–0.5 pt, outside defect noise, and with no deployment target
  B3 is kept as the default.)
- **MobileNetV3-Small and the from-scratch simplecnn cost a real ~2–3
  points** (96.3–96.6% test, 98.7–98.8% dataset) but are 6–7× faster on CPU
  and 10–17× smaller. The "deployment at all costs" options, not free ones.
- **A 0.64M from-scratch CNN ties the 1.1M ImageNet-pretrained
  MobileNetV3-Small.** Confirms the hypothesis that ImageNet texture priors
  add little on this clean synthetic line-art — the task is geometric, not
  textural.
- **ResNet50V2 failed to train** — stuck at 0.6%, early-stopped at epoch 6
  restoring epoch 1. This is a bug/instability in the resnet code path (the
  `preprocess_input`-via-`Identity`-layer wiring in `build_model`, and/or
  fully-unfrozen fine-tuning of a 24M-param backbone with Adamax lr=1e-3 from
  step 1), **not an architecture verdict**. Not pursued further: ResNet50V2
  was only a control, and at 2× B3's params it is the wrong direction anyway.

### Added
- `scripts/train.py` now takes `--backbone {efficientnetb3, efficientnetb0,
  mobilenetv3small, resnet50v2, simplecnn}` (default `efficientnetb3`, so the
  existing Slurm scripts are unchanged). `build_model()` gained a `backbone`
  first argument; the classifier head (`GlobalMaxPool → BN → 256 → Dropout →
  144`) is now shared via `_head()`. `simplecnn` is a ~0.64M-param ConvNet
  built from scratch (no ImageNet weights) — `build_simplecnn()`. ResNet50V2
  gets an explicit `resnet_v2.preprocess_input` layer; the EfficientNet and
  MobileNetV3 families keep their built-in input scaling (still raw [0,255]).
- `scripts/backbone_ablation.py` — summarises one checkpoint: param count,
  test + dataset-wide top-1, mean circular error (min), and CPU batch-1
  latency (`/CPU:0`, the deployment-relevant number). Prints a markdown table
  row to paste into the results table.
- `scripts/train_backbone_ablation.slurm` — trains all five backbones at the
  80-epoch budget and runs `backbone_ablation.py --no-latency` on each. Run
  the latency column separately in the interactive CPU shell (GPU-node CPUs
  differ).

### B0 confirmation run (job 478736, `--epochs 150 --patience 12`)
- **B0 reproduces at exactly 0.9882 test / 0.9949 dataset-wide (74 errors)** —
  identical to the ablation run despite the longer budget (early-stopped
  epoch 35, best 23). So B0's plateau is real, not undertraining.
- **Neither B0 nor the retrained B3 control reproduces the archived
  `clock_model_unfrozen_aug_80ep.keras` default's 0.9958 test / 0.9977
  dataset.** The retrained B3 got 0.9889 / 0.9961; B0 got 0.9882 / 0.9949.
  The ~0.7pt test gap is consistent across both backbones → it's a
  training-recipe / seed effect in the current `train.py`, not a backbone
  property. (No seed is set; augmentation is stochastic. The original run
  best'd at epoch 20 after a patience-5 stop at 25; these stop earlier.)
- **Verdict (revised by Step 2): B0 trails B3 by ~0.4–0.5 pt across seeds**
  and the default checkpoint stays B3. With no deployment target the size
  win buys nothing; B0 remains one flag away (`--backbone efficientnetb0`)
  for whenever size does matter.
- Added `--patience` to `train.py` (default 5, unchanged).

### Step 2 results — cyclic regression head loses; seed 0 reproduces the baseline (job 478751, 1h19m)

| run | test top-1 | dataset top-1 (errors) | mean err (dataset) |
|---|---|---|---|
| B0 softmax seed 0 | 0.9792 | 0.9920 (115) | 1.6 min |
| B0 softmax seed 1 | 0.9847 | 0.9926 (106) | 1.4 min |
| **B0 circular seed 0** | **0.2083** | **0.2128 (11335)** | 13.5 min |
| **B0 circular seed 1** | **0.2833** | **0.2744 (10448)** | 8.2 min |
| B3 softmax seed 0 | **0.9951** | **0.9975 (36)** | 0.5 min |
| B3 softmax seed 1 | 0.9896 | 0.9964 (52) | 0.7 min |

- **The cyclic sin/cos regression head is far worse — flat softmax wins
  decisively.** Both circular seeds collapse to ~21–28% exact-bucket top-1
  (vs ~99% for softmax). Mean error is ~8–13 min, well above chance (~180),
  so the head learns the *approximate* hand position but cannot resolve the
  5-minute bucket. Almost certainly because **`RandomRotation(0.03)` (±10.8°)
  augments the image but not the absolute-angle target** — every training
  example carries up to ±21 min of label noise. The softmax head is immune
  (the model reads dial numerals and stays class-invariant to small
  rotation — the documented rotation-invariance result); an absolute-angle
  regression target is not. Not pursued further unless someone wants to
  retry it with rotation aug disabled for `--head circular` (and/or a
  von Mises / `1 − cos` angular loss instead of MSE-on-(sin,cos), whose
  gradient vanishes near the target).
- **Seed 0 reproduces the archived baseline.** B3 softmax seed 0 hit 0.9975
  dataset / 0.9951 test — matching `clock_model_unfrozen_aug_80ep.keras`
  (0.9977 / 0.9958) within noise. So the recipe *does* reproduce with a
  fixed seed; the backbone-ablation misses were run-to-run variance, now
  bounded at ~0.001 dataset / ~0.005 test between seeds 0 and 1.
- **B0 vs B3 gap holds up.** B0 softmax (0.9920–0.9926 dataset) trails B3
  softmax (0.9964–0.9975) by ~0.4–0.5 pt across seeds — a bit wider than the
  ablation's 0.12 pt, and now outside pure defect noise. B0 stays the
  "cheap deployment" option, not a free swap.
- **Verdict: keep flat 144-way softmax; keep EfficientNetB3 as default.**
  Nothing here justifies swapping the default checkpoint.

### Step 2 scaffolding — cyclic regression head + seed control

- `train.py --head circular` adds a 2-unit `(sin, cos)` output
  (`Dense(2) → UnitNormalization`) trained with MSE against
  `clockmodel.CIRCULAR_TARGETS` (each class's dial angle), plus a
  `MeanMinutesError` metric. Checkpoint/EarlyStopping monitor
  `val_min_err` (min) instead of `val_accuracy`. `--head softmax` (default)
  is unchanged.
- `train.py --seed N` → `keras.utils.set_random_seed` for reproducible
  weights + augmentation + shuffling.
- `clockmodel.output_to_class_idx()` decodes either head to class indices
  (144-wide → argmax; 2-wide → nearest angle). `evaluate.py` and
  `backbone_ablation.py` now route through it and are head-agnostic;
  `evaluate.py` skips top-5 / confidence for the regression head.
- `scripts/train_head_ablation.slurm` (job 478751): B0 softmax ×2 seeds,
  B0 circular ×2 seeds, B3 softmax ×2 seeds — answers "does the cyclic head
  beat flat softmax?" and bounds the run-to-run variance in one job.
  120-epoch budget, patience 12.

Resolved: single-angle regression underperforms badly (job 478751 above);
flat softmax kept. A two-hand-angle decomposition was not tried — the more
likely fix is rotation-aug / loss-function, see the Step 2 results.
- ResNet50V2 path is left as-is (documented failure); fix only if a
  ResNet-class control is ever actually wanted.

## 2026-08-24 — characterizing the rotation-brittleness cliff

Follow-up to both the original rotation-brittleness finding (an ad-hoc,
uncommitted check: 0 deg -> 100%, 3 deg -> ~69%, 6 deg -> 0%) and the
real-photo collapse below, which raised the question of how much of that
failure is rotation-shaped. Two things were unknown: the cliff's actual
shape (only three points had ever been measured, on the *released*
checkpoint only), and whether training-time `RandomRotation(0.03)`
augmentation (used for the current default checkpoint) changes its shape or
just shifts it.

### Added
- `scripts/characterize_rotation_brittleness.py` -- sweeps rotation angle
  (-90 to +90 degrees, finer resolution near 0) over 144 test images (one
  per class), for both the default and released checkpoints, using
  `scipy.ndimage.rotate` (edge-extended corners, not black -- avoids
  confounding "rotation" with "sudden black wedge in frame"). Also renders
  Grad-CAM at increasing angles for one example.
- `docs/images/rotation_cliff.png` -- accuracy vs. angle, both checkpoints.
- `docs/images/rotation_gradcam.png` -- Grad-CAM at 0/3/6/10/20/45 degrees.
- Added `scipy` to `requirements.txt` (already installed in the conda env).

### Findings
- **Corrects an earlier error**: `RandomRotation(0.03)` was previously
  described as "~5.4 degrees" (AGENTS.md, README.md). It's actually
  **+/-10.8 degrees** -- Keras's `factor` is a fraction of a full turn used
  as *both* the lower and upper bound, so `0.03` means the layer samples
  from `[-0.03, 0.03] * 360 deg`, not half that. The measured plateau below
  confirms this figure is right.
- **The default checkpoint has a genuine flat plateau, not just a softer
  peak.** Accuracy stays at ~99% essentially unchanged from -10 deg to
  +10 deg -- matching the +/-10.8 degree training augmentation range almost
  exactly -- then collapses sharply: 93.8% at +15 deg, 41.7% at +20 deg,
  **0% from +30 deg through +60 deg** (a genuine dead zone, not a gradual
  tail), before jumping back to 99.3% at exactly +90 deg. Symmetric in the
  negative direction.
- **The released checkpoint (no rotation augmentation at all) has no
  plateau** -- a much narrower, smoothly-decaying peak: 95.8% at 2 deg
  already down to 83.3%/80.6% at 3 deg, ~0% by 15 deg, same dead zone, same
  recovery at 90 deg. Confirms the earlier three-point estimate (3 deg ->
  ~69%) was in the right ballpark, roughly the average of the +3/-3 values
  measured here (80.6%/83.3%) at a different sample.
- **Training-time rotation augmentation doesn't teach genuine rotation
  invariance -- it widens the memorized-safe range to match what it saw in
  training, verbatim.** The plateau's edges align with the augmentation
  range almost to the degree. This is augmentation working exactly as
  literally specified, not as a proxy for a more general robustness.
- **Confident wrongness, not uncertainty, dominates the dead zone.** Mean
  top-1 confidence stays high (0.44-0.99) throughout the 15-89 degree dead
  zone for both checkpoints, even at 0% accuracy -- the model doesn't
  "know" it's off-distribution, it just reads the wrong time with the same
  confidence it reads the right one. Notably *higher* confidence than the
  real-photo test below (mean 0.205) -- suggesting the real-photo failure
  isn't pure rotation-confusion; something about real photos (lighting,
  perspective, dial style, or the *combination*) pushes the model into a
  less-confident regime that isolated synthetic rotation alone doesn't
  reproduce.
- **Grad-CAM answers "does attention diffuse or drift?" -- neither. It stays
  locked on the (rotated) hands even as the prediction goes wrong.** At
  +20 deg and +45 deg (both fully in the dead zone, both wrong and both
  ~0.78-0.79 confident), attention is exactly as sharp and hand-focused as
  at 0 deg. This is a different failure signature than the blank-dial
  dataset defect (diffuse attention, near-random 0.06-0.10 confidence, see
  the 33-error diagnosis) -- the rotation failure is a confident
  *misreading* of a correctly-located hand position, not a loss of
  localization. Consistent with 90/180/270-degree exact invariance: those
  are the only out-of-distribution orientations with zero interpolation
  artifacts and exact pixel-grid correspondence between rotated numerals and
  rotated hands; every other angle introduces interpolation the model
  never learned to read correctly outside its narrow trained/memorized
  range.
- **Practical implication for the real-photo work above:** since
  augmentation only ever teaches the literal trained range, pushing the
  real-photo gap further with more rotation augmentation would likely need
  a much wider augmentation range (well past +/-10.8 degrees) to have any
  chance of covering real off-axis photography -- not a small tweak.

## 2026-08-24 (very newest) — real (non-synthetic) clock photos: the model collapses

Follow-up to the rotation-brittleness finding (AGENTS.md: 3° costs ~30
points, 6° destroys accuracy entirely). Every image the model has ever been
measured on is synthetic, upright, centered, and set to an exact 5-minute
mark. Real photos violate all of that at once (off-axis angle, imprecise
hand positions, varied lighting/framing) -- does the model generalize, or was
99.99% "real" accuracy on the synthetic set an artifact of how narrow that
set is?

### Added
- `scripts/evaluate_real_photos.py` -- evaluates the default checkpoint
  against a directory of real clock photos labelled by filename
  (`<hash>_<hour>_<minute>.jpg`), reporting both exact-class accuracy
  (true time rounded to the nearest 5-minute class, matching the model's
  actual 144-way task) and circular minutes-off error against the
  *unrounded* true time. Also writes a qualitative 12-image
  best-6/worst-6 grid.
- `real_data/` (gitignored) -- `kongaskristjan/real-clocks` downloaded via
  `kaggle datasets download -d kongaskristjan/real-clocks -p real_data
  --unzip`. See `DATASET.md`'s new "Real-photo test set" section for
  provenance/license/re-download steps.
- `docs/images/real_photo_predictions.png` -- the qualitative grid,
  committed as a static figure.

### Findings
- **It collapses, as predicted.** 92 real photos: **5.4% top-1 accuracy**
  (vs. 99.58% on the synthetic test split), 15.2% top-5, mean circular error
  **175.7 minutes** (median 170), only 6.5% of predictions within 5 minutes
  of the true time, mean confidence on its own (usually wrong) top-1 guess
  just 0.205 -- vs. typically >0.9 on synthetic images, correct or not. This
  is not "somewhat worse," it's a different regime: essentially random
  performance on a 144-way task (chance level ~0.7%) that would need the
  model to have learned nothing at all to do much worse.
- **Errors are broadly spread, not one clean systematic offset.** Binned by
  circular minutes-off: 6 within 5 min, 3 within 5-15, 3 within 15-30, 2
  within 30-60, 14 within 60-120, 16 within 120-165, 14 within 165-195
  (roughly a 6-hour/180-degree "opposite side of the dial" error -- notably
  present but only 15.2% of cases, not dominant), 18 within 195-300, 16
  within 300-360. Ruled out the appealing hypothesis that this is "just" the
  ±3h15m dataset-defect pattern recurring, or a single systematic
  hand-reading flip -- the spread across every error magnitude looks like
  genuine model confusion on out-of-distribution inputs, not one fixable bug.
- **Qualitatively**, the 6 best predictions (0-2 min off, one 3 min) are
  visually close to the synthetic training distribution: clean, roughly
  frontal, well-lit clock faces. The 6 worst (300-360 min off, i.e. roughly
  opposite-side-of-dial reads) include odd framings, ornate/photographic
  dials, and reflective surfaces -- consistent with the rotation/off-axis
  brittleness hypothesis, though this sample is too small (92 images, no
  angle/lighting labels) to isolate which specific factor (angle vs.
  lighting vs. dial style vs. imprecise hand positions) drives the failure.
- **Conclusion:** this confirms the concern that motivated the test --
  99.99%-on-synthetic does not transfer to real photos at all, and this is a
  materially different, harder problem than anything tuned on the current
  dataset. Treat as a separate investigation (different data collection,
  probably different architecture/training strategy, possibly explicit
  rotation/perspective augmentation on a real or realistic training set) --
  not a case for more epochs or augmentation on the synthetic set, which
  this result suggests wouldn't transfer anyway.
- Not folded into any other accuracy figure in this repo -- `real_data/` is
  kept strictly separate from `data/` (the training/eval set) and this
  checkpoint was not retrained on it.

## 2026-08-24 (newest) — automated tests + CI

### Added
- `tests/test_clockmodel.py` (pytest) — two tiers. Pure-logic tests (label
  formatting, prediction decoding, the Keras 2.8→3 `DepthwiseConv2D` compat
  patch) always run. Data/model-backed tests need `data/` and/or the default
  checkpoint and skip automatically otherwise:
  `test_class_names_order_is_alphabetical_over_underscore_labels` and
  `test_class_names_does_not_match_csv_class_index_order` lock down
  `clockmodel.class_names()`'s ordering against both known-wrong
  alternatives (the class-ordering trap in `AGENTS.md` #1);
  `test_default_checkpoint_test_accuracy_regression` asserts test-split
  top-1 accuracy stays >= 0.99 against the default checkpoint.
- `.github/workflows/ci.yml` — runs the pytest suite on every push/PR to
  `main`. `data/` and `models/` are both gitignored (symlink to shared
  storage / local training output) and not present on the runner, so CI
  only ever exercises the pure-logic tier; the data/model-backed tests are
  for local verification per `AGENTS.md`'s "verify, don't assume" guidance.
- `pytest==9.0.2` added to `requirements.txt`.
- `README.md` gained a "Testing" section; the corresponding "Automated
  tests" / "CI" future-work items are removed now that both exist.

### Verified
- `python -m pytest tests/ -v`: 11 passed locally (data and default
  checkpoint both present).

## 2026-08-24 — README figures + expanded future work

### Added
- `scripts/generate_readme_figures.py` — generates the 4 PNGs now embedded in
  `README.md`, run once against the default checkpoint and committed as
  static files under `docs/images/`: `sample_predictions.png` (grid of test
  images with true/predicted labels), `gradcam_comparison.png` (Grad-CAM on a
  correct prediction vs. both dataset defects), `dataset_defects.png` (normal
  render vs. blank-dial vs. ±3h15m-shift defect, side by side), and
  `checkpoint_accuracy.png` (bar chart of the three checkpoints' full-dataset
  accuracy from the `AGENTS.md` comparison table). Re-run and recommit if the
  default checkpoint changes.
- Expanded `README.md`'s "Future work" section: real (non-synthetic) photo
  testing, a public model-card/writeup, confidence-based dataset-defect
  auto-flagging, characterizing the rotation-brittleness cliff, confidence
  calibration, and model size/deployment — on top of the existing automated
  tests / CI items.

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
