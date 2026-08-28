#!/bin/bash
#SBATCH -p dgxa100
#SBATCH --gpus=1
#SBATCH --qos=big
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=c22,c11,c15
#SBATCH --job-name=clock_rotation_range
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=12:00:00

cd /home/kargin/Projects/personal/analog_clock_prediction

PY=/shared/results/common/kargin/tck_miniconda3/bin/python3
PYROOT=/shared/results/common/kargin/tck_miniconda3/lib/python3.12/site-packages
export TF_CPP_MIN_LOG_LEVEL=3
# See AGENTS.md / CHANGELOG.md (2026-08-23 GPU setup).
export LD_LIBRARY_PATH=$(for d in "$PYROOT"/nvidia/*/lib; do echo -n "$d:"; done)
export PATH="$PYROOT/nvidia/cuda_nvcc/bin:$PATH"

# README future work: "Widen the rotation-augmentation range."
# The brittleness cliff's edges track the literal training aug range
# (CHANGELOG 2026-08-24). Train B3 softmax (seed 0, the recipe that
# reproduces the 0.9977 baseline) at four RandomRotation factors and see
# whether the robust plateau widens with the training range, and what it
# costs upright accuracy / training time.
#
#   factor 0.03 -> +/-10.8 deg  (current default, control)
#   factor 0.06 -> +/-21.6 deg
#   factor 0.10 -> +/-36.0 deg
#   factor 0.15 -> +/-54.0 deg

CKPTS=()
run () {  # factor
    OUT="models/clock_rot_f${1}_s0.keras"
    CKPTS+=("$OUT")
    echo "======== rotation-factor $1 -> $OUT ========"
    $PY scripts/train.py --backbone efficientnetb3 --head softmax --seed 0 \
        --rotation-factor "$1" --epochs 120 --patience 12 --out "$OUT"
    $PY scripts/backbone_ablation.py "$OUT" --no-latency
    $PY scripts/evaluate.py --split test --model "$OUT"
}

run 0.03
run 0.06
run 0.10
run 0.15

echo "======== rotation-range sweep across all four ========"
$PY scripts/rotation_range_sweep.py "${CKPTS[@]}"

echo "All runs done."
