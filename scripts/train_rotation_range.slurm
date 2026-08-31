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
#SBATCH --job-name=clock_rotation_range
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
