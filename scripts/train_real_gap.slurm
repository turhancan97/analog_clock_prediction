#!/bin/bash
# Portable resource request. Site-specific flags (partition, QoS, node
# exclusions) are NOT baked in -- pass them to sbatch, which overrides:
#
#   sbatch -p <your partition> --qos=<your qos> scripts/<this file>
#
# The runs in CHANGELOG.md used: -p dgxa100 --qos=big --exclude=<bad nodes>
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --job-name=clock_real_gap
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=12:00:00

# Run from wherever sbatch was invoked (submit from the repo root), else
# fall back to this script's parent directory.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

# Point PY at your interpreter, or export PY=<your python> before submitting.
PY="${PY:-python3}"
PYROOT="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
export TF_CPP_MIN_LOG_LEVEL=3
# See AGENTS.md / CHANGELOG.md (2026-08-23 GPU setup).
export LD_LIBRARY_PATH=$(for d in "$PYROOT"/nvidia/*/lib; do echo -n "$d:"; done)
export PATH="$PYROOT/nvidia/cuda_nvcc/bin:$PATH"

# README future work #1: close the synthetic->real gap. Two moves, stacked:
#   B  realism augmentation  (--realism-aug: perspective / shear / photometric
#      / blur on top of the ±54° rotation) -- scales to all 144 classes,
#      touches no real data.
#   A  real-data fine-tune   (--real-mix: 138 real training photos from
#      real_data/real_manifest.csv, split=='train', oversampled into the
#      synthetic stream; --init-weights from the current default).
# The 57-photo real TEST split is never trained on -- reported by train.py at
# the end of every run and by evaluate_real_photos.py --split test.
#
# Baseline to beat (current default, clock_model_rot54_s0.keras):
#   real test  top-1 ~0.07   |   real train top-1 ~0.014   |   synthetic 0.9972

DEF=models/clock_model_rot54_s0.keras

run () {  # name  extra-args...
    NAME="$1"; shift
    OUT="models/clock_${NAME}.keras"
    echo "======== ${NAME} -> $OUT ========"
    $PY scripts/train.py --backbone efficientnetb3 --head softmax --seed 0 \
        --epochs 120 --patience 15 --out "$OUT" "$@"
    echo "---- evaluate.py (synthetic test) ----"
    $PY scripts/evaluate.py --split test --model "$OUT"
    echo "---- evaluate_real_photos.py --split test ----"
    $PY scripts/evaluate_real_photos.py --split test --model "$OUT" \
        --figure "docs/images/real_photo_predictions_${NAME}.png"
}

# B alone: realism aug, trained fresh.
run realism  --realism-aug

# B + A: realism aug + real-photo mix, fine-tuned from the current default.
run realism_realmix  --realism-aug --real-mix --real-oversample 25 \
    --init-weights "$DEF" --learning-rate 3e-4

# B + A from scratch (no init) -- controls for "fine-tune vs train-together".
run realism_realmix_scratch  --realism-aug --real-mix --real-oversample 25

echo "All runs done."
