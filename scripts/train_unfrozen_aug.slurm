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
#SBATCH --job-name=clock_train
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=06:00:00

# Run from wherever sbatch was invoked (submit from the repo root), else
# fall back to this script's parent directory.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

# Point PY at your interpreter, or export PY=<your python> before submitting.
PY="${PY:-python3}"
PYROOT="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
export TF_CPP_MIN_LOG_LEVEL=3
# Without this, TF can't dlopen the pip-installed CUDA/cuDNN libs and
# silently falls back to CPU-only training.
export LD_LIBRARY_PATH=$(for d in "$PYROOT"/nvidia/*/lib; do echo -n "$d:"; done)
# ptxas (from nvidia-cuda-nvcc-cu12) is needed by cuDNN's conv autotuner to
# JIT-compile kernels; without it every algorithm fails with "Autotuner
# could not compile any configs".
export PATH="$PYROOT/nvidia/cuda_nvcc/bin:$PATH"

$PY scripts/train.py --epochs 20 --out models/clock_model_unfrozen_aug.keras

$PY scripts/evaluate.py --split test --model models/clock_model_unfrozen_aug.keras
