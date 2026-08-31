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
#SBATCH --job-name=clock_demo_model
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=4:00:00

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

# Compact model for the browser inference demo (docs/demo.html). The B3
# default is ~135 MB and cannot be embedded in an Artifact (16 MB cap, no
# runtime fetch); this ~0.64 M-param from-scratch CNN quantises to <1 MB.
#
# NOTE: the demo ships models/clock_simplecnn.keras (from the backbone
# ablation, +/-10.8 deg rotation aug, 0.963 test). A from-scratch simplecnn
# does NOT converge under the +/-54 deg default aug (stuck ~0.54 val at
# epoch 37 -- job 479504, cancelled); the ImageNet-pretrained backbones
# handle +/-54 deg fine but simplecnn cannot learn it from scratch.
# --rotation-factor 0.06 (+/-21.6 deg) is a reproducible middle ground.

OUT=models/clock_demo_simplecnn.keras
$PY scripts/train.py --backbone simplecnn --seed 0 --rotation-factor 0.06 \
    --epochs 150 --patience 15 --out "$OUT"

$PY scripts/evaluate.py --split test --model "$OUT"
$PY scripts/backbone_ablation.py "$OUT" --no-latency

echo "done."
