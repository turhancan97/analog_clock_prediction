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
#SBATCH --job-name=clock_b0_confirm
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

# EfficientNetB0 confirmation run (backbone ablation step 1 follow-up). The
# ablation sweep's B0 landed at 0.9949 dataset-wide but early-stopped at
# epoch 32 / best 27 with patience=5 -- possibly undertrained vs the 0.9977
# clock_model_unfrozen_aug_80ep.keras default (which itself best'd at epoch
# 20 but on B3). Give it a longer budget and more patience; everything else
# matches the winning recipe. If this matches ~0.9977, swap DEFAULT_MODEL in
# src/clockmodel.py to this checkpoint and regenerate docs/images/.
OUT=models/clock_efficientnetb0_confirm.keras
$PY scripts/train.py --backbone efficientnetb0 --epochs 150 --patience 12 --out "$OUT"

echo "======== ablation summary ========"
$PY scripts/backbone_ablation.py "$OUT" --no-latency

echo "======== per-class error vs current default (compare_full_dataset) ========"
# compare_full_dataset.py hardcodes its two checkpoints; run evaluate.py for
# the split-level view against both instead.
$PY scripts/evaluate.py --split test --model "$OUT"
$PY scripts/evaluate.py --split test --model models/clock_model_unfrozen_aug_80ep.keras
