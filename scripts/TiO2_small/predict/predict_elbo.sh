#!/bin/bash
set -euo pipefail

export PYTHONPATH=/home/g15farris/bin/bayesaenet:$PYTHONPATH
cd /home/g15farris/bin/bayesaenet
mkdir -p log/predict

# Usage examples:
#   MODEL_TYPE=lrt RUNS_DIR=bnn_aenet/logs/TiO2_small/hps/runs/lrt/2026-03-23_11-48-03 \
#   OUTPUT_DIR=bnn_aenet/logs/TiO2_small/pred/lrt_ln_test ./scripts/TiO2_small/predict/test.sh
#
#   MODEL_TYPE=rad RUNS_DIR=bnn_aenet/logs/TiO2_small/hps/runs/rad/<timestamp> \
#   OUTPUT_DIR=bnn_aenet/logs/TiO2_small/pred/rad_ln_test ./scripts/TiO2_small/predict/test.sh
MODEL_TYPE="${MODEL_TYPE:-lrt}"
RUNS_DIR="${RUNS_DIR:-bnn_aenet/logs/TiO2_small/hps/runs/lrt/2026-03-24_14-59-56}"
OUTPUT_DIR="${OUTPUT_DIR:-bnn_aenet/logs/TiO2_small/pred/${MODEL_TYPE}_elbo}"
SUBSETS="${SUBSETS:-test}"
MC_SAMPLES="${MC_SAMPLES:-20}"

mkdir -p "$OUTPUT_DIR"

# HPS runs keep Hydra files in sweep root, but predict_forces expects per-run .hydra.
ROOT_HYDRA_DIR="${RUNS_DIR}/.hydra"
if [ -f "${ROOT_HYDRA_DIR}/overrides.yaml" ]; then
  for run in "${RUNS_DIR}"/*; do
    [ -d "$run" ] || continue
    run_base="$(basename "$run")"
    [[ "$run_base" =~ ^[0-9]+$ ]] || continue
    mkdir -p "$run/.hydra"
    [ -f "$run/.hydra/overrides.yaml" ] || cp "${ROOT_HYDRA_DIR}/overrides.yaml" "$run/.hydra/overrides.yaml"
    [ -f "$run/.hydra/config.yaml" ] || cp "${ROOT_HYDRA_DIR}/config.yaml" "$run/.hydra/config.yaml"
  done
fi

python -m bnn_aenet.tasks.predict_forces \
  --model-type "$MODEL_TYPE" \
  --runs-dir "$RUNS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --data-dir data/TiO/train_forces.in \
  --use-run-config \
  --subsets "$SUBSETS" \
  --device cpu \
  --mc-samples "$MC_SAMPLES"

python -m bnn_aenet.tasks.plot \
   --pred-dir bnn_aenet/logs/TiO2_small/pred/lrt_elbo \
   --output-dir plots/TiO2_small/elbo \   
   --subsets test