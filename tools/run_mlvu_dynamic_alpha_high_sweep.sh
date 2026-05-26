#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export ALPHAS="${ALPHAS:-0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8}"
export MAX_DURATION="${MAX_DURATION:-3600}"
export NUM_CHUNKS="${NUM_CHUNKS:-4}"
export GPUS="${GPUS:-0 1 2 3}"
export SAMPLE_FPS="${SAMPLE_FPS:-1.0}"
export RUN_NAME="${RUN_NAME:-dynamic_mass_min16_max64_alpha0p4_0p8_step0p05_maxdur${MAX_DURATION}}"

bash tools/run_mlvu_dynamic_alpha_reuse_video.sh
