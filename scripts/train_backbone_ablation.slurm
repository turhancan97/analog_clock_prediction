#!/bin/bash
#SBATCH -p dgxa100
#SBATCH --gpus=1
#SBATCH --qos=big
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=c22,c11,c15
#SBATCH --job-name=clock_backbone_ablation
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=24:00:00

cd /home/kargin/Projects/personal/analog_clock_prediction

PY=/shared/results/common/kargin/tck_miniconda3/bin/python3
PYROOT=/shared/results/common/kargin/tck_miniconda3/lib/python3.12/site-packages
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
