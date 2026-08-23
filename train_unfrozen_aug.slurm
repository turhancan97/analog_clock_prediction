#!/bin/bash
#SBATCH -p dgxa100
#SBATCH --gpus=1
#SBATCH --qos=big
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=c22,c11,c15
#SBATCH --job-name=clock_train
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=06:00:00

cd /home/kargin/Projects/personal/analog_clock_prediction

PY=/shared/results/common/kargin/tck_miniconda3/bin/python3
PYROOT=/shared/results/common/kargin/tck_miniconda3/lib/python3.12/site-packages
export TF_CPP_MIN_LOG_LEVEL=3
# Without this, TF can't dlopen the pip-installed CUDA/cuDNN libs and
# silently falls back to CPU-only training.
export LD_LIBRARY_PATH=$(for d in "$PYROOT"/nvidia/*/lib; do echo -n "$d:"; done)
# ptxas (from nvidia-cuda-nvcc-cu12) is needed by cuDNN's conv autotuner to
# JIT-compile kernels; without it every algorithm fails with "Autotuner
# could not compile any configs".
export PATH="$PYROOT/nvidia/cuda_nvcc/bin:$PATH"

$PY train.py --epochs 20 --out clock_model_unfrozen_aug.keras

$PY evaluate.py --split test --model clock_model_unfrozen_aug.keras
