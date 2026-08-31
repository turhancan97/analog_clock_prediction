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
#SBATCH --job-name=clock_backbone_ablation
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=24:00:00

# Run from wherever sbatch was invoked (submit from the repo root), else
# fall back to this script's parent directory.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

# Point PY at your interpreter, or export PY=<your python> before submitting.
PY="${PY:-python3}"
PYROOT="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
export TF_CPP_MIN_LOG_LEVEL=3
# See AGENTS.md / CHANGELOG.md (2026-08-23 GPU setup): without both of these
# TF silently falls back to CPU or the cuDNN autotuner crashes.
export LD_LIBRARY_PATH=$(for d in "$PYROOT"/nvidia/*/lib; do echo -n "$d:"; done)
export PATH="$PYROOT/nvidia/cuda_nvcc/bin:$PATH"

# Backbone ablation (step 1 of the architecture investigation). Same recipe as
# the winning clock_model_unfrozen_aug_80ep.keras run -- 80-epoch budget,
# EarlyStopping(patience=5), same augmentation -- only the backbone changes.
# efficientnetb3 is retrained here too as the apples-to-apples control (the
# released default was trained on different tooling). simplecnn has no
# ImageNet weights so it may want the full 80 epochs.
for BB in efficientnetb3 efficientnetb0 mobilenetv3small resnet50v2 simplecnn; do
    OUT=models/clock_${BB}.keras
    echo "======== training $BB -> $OUT ========"
    $PY scripts/train.py --epochs 80 --backbone "$BB" --out "$OUT"
    echo "======== ablation summary: $BB ========"
    $PY scripts/backbone_ablation.py "$OUT" --no-latency
done

echo "All backbones done. Run scripts/backbone_ablation.py <ckpt> in the"
echo "interactive CPU shell for the latency column (GPU-node CPUs differ)."
