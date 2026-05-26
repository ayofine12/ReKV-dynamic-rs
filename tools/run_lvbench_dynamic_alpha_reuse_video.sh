#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL=${MODEL:-qwen2_5_vl_7b}
SAMPLE_FPS=${SAMPLE_FPS:-1.0}
MIN_RS=${MIN_RS:-16}
MAX_RS=${MAX_RS:-64}
# Qwen2.5-VL uses 64 visual tokens per ReKV block at frame_size=224.
N_LOCAL=${N_LOCAL:-4096}
QWEN_BLOCK_SIZE=${QWEN_BLOCK_SIZE:-64}
DEBUG=${DEBUG:-false}
SAVE_RETRIEVAL_LOGITS=${SAVE_RETRIEVAL_LOGITS:-true}
NORMALIZE=${NORMALIZE:-zscore_softmax}
GPU=${GPU:-0}
PYTHON=${PYTHON:-/root/mwnoh/anaconda3/envs/rekv/bin/python}
ANNO_PATH=${ANNO_PATH:-data/lvbench/full_mc.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ssd1/mwnoh/LVBench/results}
ALPHAS=${ALPHAS:-"0.2 0.25 0.3 0.35"}
RUN_NAME=${RUN_NAME:-dynamic_mass_min${MIN_RS}_max${MAX_RS}}

"${PYTHON}" tools/prepare_lvbench_data.py --output "${ANNO_PATH}"

min_n_local=$((MAX_RS * QWEN_BLOCK_SIZE))
RUN_N_LOCAL="${N_LOCAL}"
if (( RUN_N_LOCAL < min_n_local )); then
  RUN_N_LOCAL="${min_n_local}"
fi

save_root="${OUTPUT_ROOT}/${MODEL}/lvbench/${RUN_NAME}"
echo "[lvbench-dynamic-reuse] alphas=${ALPHAS}, min_rs=${MIN_RS}, max_rs=${MAX_RS}, gpu=${GPU}, n_local=${RUN_N_LOCAL}, save_root=${save_root}"
mkdir -p "${save_root}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" video_qa/rekv_offline_vqa.py \
  --model "${MODEL}" \
  --sample_fps "${SAMPLE_FPS}" \
  --n_local "${RUN_N_LOCAL}" \
  --retrieve_size "${MAX_RS}" \
  --dynamic_retrieve_alphas "${ALPHAS}" \
  --dynamic_retrieve_min_size "${MIN_RS}" \
  --dynamic_retrieve_max_size "${MAX_RS}" \
  --dynamic_retrieve_normalize "${NORMALIZE}" \
  --save_dir "${save_root}" \
  --anno_path "${ANNO_PATH}" \
  --debug "${DEBUG}" \
  --save_retrieval_logits "${SAVE_RETRIEVAL_LOGITS}" \
  --num_chunks 1 \
  --chunk_idx 0

for alpha in ${ALPHAS}; do
  alpha_label=${alpha//./p}
  run_dir="${save_root}/alpha${alpha_label}-${SAMPLE_FPS}"
  cp "${run_dir}/1_0.csv" "${run_dir}/results.csv"
  "${PYTHON}" video_qa/eval/eval_lvbench.py --save_dir "${run_dir}"
done

"${PYTHON}" tools/summarize_lvbench_dynamic_alpha.py \
  --base_dir "${save_root}" \
  --sample_fps "${SAMPLE_FPS}" \
  --max_rs "${MAX_RS}"
