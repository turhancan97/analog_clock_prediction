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
#SBATCH --job-name=clock_head_ablation
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

# Step 2 of the architecture investigation + a seed-variance check.
#
# Question 1 (the "better fit"): does a cyclic (sin, cos) regression head beat
# the flat 144-way softmax? It respects that 11:55 and 12:00 are adjacent;
# softmax does not. Compared head-to-head on efficientnetb0 (fast to iterate),
# two seeds each.
# Question 2 (housekeeping): bound the run-to-run variance that made the
# backbone ablation's retrained B3 (0.9889 test) miss the archived 0.9977
# default -- two seeded B3 softmax runs.
#
# Recipe otherwise matches the winning run: 120-epoch budget, patience 12.

run () {  # backbone head seed
    OUT="models/clock_${1}_${2}_s${3}.keras"
    echo "======== $1 / $2 / seed $3 -> $OUT ========"
    $PY scripts/train.py --backbone "$1" --head "$2" --seed "$3" \
        --epochs 120 --patience 12 --out "$OUT"
    $PY scripts/backbone_ablation.py "$OUT" --no-latency
}

run efficientnetb0 softmax  0
run efficientnetb0 softmax  1
run efficientnetb0 circular 0
run efficientnetb0 circular 1
run efficientnetb3 softmax  0
run efficientnetb3 softmax  1

echo "All runs done."
