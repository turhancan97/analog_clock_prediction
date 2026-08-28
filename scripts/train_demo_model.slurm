#!/bin/bash
#SBATCH -p dgxa100
#SBATCH --gpus=1
#SBATCH --qos=big
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=c22,c11,c15
#SBATCH --job-name=clock_demo_model
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=4:00:00

cd /home/kargin/Projects/personal/analog_clock_prediction

PY=/shared/results/common/kargin/tck_miniconda3/bin/python3
PYROOT=/shared/results/common/kargin/tck_miniconda3/lib/python3.12/site-packages
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
