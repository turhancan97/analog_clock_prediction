# Dataset

## Source

**TIME — Image Dataset-Classification** by Gerry Piosenka (`gpiosenka`), on Kaggle.

- Page: https://www.kaggle.com/datasets/gpiosenka/time-image-datasetclassification
- Slug: `gpiosenka/time-image-datasetclassification`
- Subtitle: *"144 time classes of the form hour-minute"*
- **License: CC0 — Public Domain** (no attribution required, though it's polite)
- Kaggle-reported size: 402,067,147 bytes

Synthetically generated clock faces, 224×224 RGB, rendered with fixed-length
rectangular hands in assorted colours over a variety of dial designs. 144
classes = 12 hours × 12 five-minute positions, labelled `hour-minute`
(`1-30`, `11-45`).

### How this was identified

The dataset ships no metadata and the JPEGs carry no EXIF. It was matched on the
`clocks.csv` header — `class index,filepaths,labels,data set` — which is the
signature of Gerry Piosenka's Kaggle datasets, all of which use that exact
schema. The `hour-minute` label format, the 144-class count, the 224×224 RGB
images, and the 80/10/10 per-class split all corroborate.

The bundled `time-99.68.h5` also follows that author's convention of shipping a
trained checkpoint named `<topic>-<accuracy>.h5`, and its architecture
(EfficientNetB3 → GlobalMaxPool → BatchNorm → Dense(256) → Dropout → Dense,
Adamax @ 1e-3) is his standard notebook recipe. **Confirmed 2026-08-23**: an
authenticated `kaggle datasets download -f time-99.68.h5` on this dataset slug
returns the file directly (`2022-08-18` creation date, same day as
`clocks.csv`), and it is byte-identical to the local copy — 135,837,376 bytes,
md5 `6bb35a8914cd9d329c036edfcd575879`. It ships with the dataset, not added
later.

## Where it lives right now

`data/` in this repo is a **symlink**, not a directory:

```
data -> /shared/sets/datasets/vision/analog-clock/data
```

It is on shared storage. Nothing below needs running unless that path goes away
or you're setting up on a new machine. Note the shared copy totals 418,698,419
bytes against Kaggle's reported 402,067,147 — about 4% larger, most likely a
dataset version difference. Treat exact byte equality as a non-goal.

## Downloading a fresh copy

Needs a Kaggle account and an API token: kaggle.com → your profile → Settings →
API → *Create New Token*, which downloads `kaggle.json`.

```bash
pip install kaggle
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json          # kaggle refuses to run without this

# into a directory of your choosing
kaggle datasets download -d gpiosenka/time-image-datasetclassification -p /path/to/dest --unzip
```

Then point the repo at it:

```bash
cd /home/kargin/Projects/personal/analog_clock_prediction
rm -f data && ln -s /path/to/dest data
```

Or just download the ZIP from the dataset page and unzip it yourself — the CLI
is a convenience, not a requirement.

## Verifying a copy

Expected layout:

```
data/
  train/  144 dirs × 80 images = 11,520
  valid/  144 dirs × 10 images =  1,440
  test/   144 dirs × 10 images =  1,440
  clocks.csv          14,401 lines (header + 14,400)
  time-99.68.h5       135,837,376 bytes
```

```bash
cd /home/kargin/Projects/personal/analog_clock_prediction
for s in train valid test; do
  echo "$s: $(ls data/$s | wc -l) classes, $(find -L data/$s -name '*.jpg' | wc -l) images"
done
wc -l < data/clocks.csv
```

Use `find -L` — plain `find data` silently returns nothing through the symlink.

The real end-to-end check is the pretrained model, which should reproduce its
published accuracy on an intact copy:

```bash
python evaluate.py --split test     # expect ~0.9938 top-1
```

## Caveats

- **Synthetic and uniform.** Every clock is upright, centred, and set to an
  exact 5-minute mark. The pretrained model is correspondingly brittle — a 6°
  rotation drops it to 0% (see `CHANGELOG.md`). Accuracy here will not transfer
  to photographs of real clocks.
- **`clocks.csv`'s `class index` column does not match the model's class
  order.** See `AGENTS.md` before using it for anything.
- Datasets on Kaggle can be revised in place. If counts drift from the numbers
  above, you likely have a newer version than the one these results came from.
- **19 of 144 classes have a broken render at file index `36`** — a bare dial
  with no hands drawn at all (visually confirmed; not corrupt JPEGs, just
  blank renders), plus one further mislabeled/mis-rendered image
  (`valid/8-50/36.jpg`, hands drawn near 12:05, not 8:50). All 20 land on
  index `36` specifically, spread across otherwise-unrelated classes — a
  dataset-generation bug, not a model weakness. They account for the majority
  (20 of 33) of the current best model's dataset-wide errors; see
  `CHANGELOG.md` (2026-08-23, "33-error diagnosis"). Any accuracy figure
  computed on this dataset is capped below 100% by these regardless of model
  quality.

### If you later want real photographs

Related Kaggle datasets, none of them verified here:

- `kongaskristjan/real-clocks` — time-labelled photos of real clocks
- `vctorsuarezvara/real-images-of-analogclocks` — 102 hand-labelled real images
- `shivajbd/analog-clocks` — 50K synthetic, labels as hour/minute columns
