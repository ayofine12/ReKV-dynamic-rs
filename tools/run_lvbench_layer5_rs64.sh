#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-qwen2_5_vl_7b}
SAMPLE_FPS=${SAMPLE_FPS:-1.0}
BASE_RS=${BASE_RS:-16}
HIGH_RS=${HIGH_RS:-64}
LAYER_IDX=${LAYER_IDX:-5}
# Qwen2.5-VL uses 64 visual tokens per ReKV block at frame_size=224.
N_LOCAL=${N_LOCAL:-4096}
QWEN_BLOCK_SIZE=${QWEN_BLOCK_SIZE:-64}
DEBUG=${DEBUG:-false}
SAVE_RETRIEVAL_LOGITS=${SAVE_RETRIEVAL_LOGITS:-false}
GPU=${GPU:-0}
PYTHON=${PYTHON:-/root/mwnoh/anaconda3/envs/rekv/bin/python}
ANNO_PATH=${ANNO_PATH:-data/lvbench/full_mc.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ssd1/mwnoh/LVBench/results}
RUN_NAME=${RUN_NAME:-layer5_only_base${BASE_RS}_high${HIGH_RS}}

"${PYTHON}" tools/prepare_lvbench_data.py --output "${ANNO_PATH}"

min_n_local=$((HIGH_RS * QWEN_BLOCK_SIZE))
RUN_N_LOCAL="${N_LOCAL}"
if (( RUN_N_LOCAL < min_n_local )); then
  RUN_N_LOCAL="${min_n_local}"
fi

save_dir="${OUTPUT_ROOT}/${MODEL}/lvbench/${RUN_NAME}/${LAYER_IDX}-${SAMPLE_FPS}"
layer_spec="${LAYER_IDX}:${HIGH_RS}"

echo "[lvbench-layer5] base_rs=${BASE_RS}, override=${layer_spec}, gpu=${GPU}, n_local=${RUN_N_LOCAL}, save_dir=${save_dir}"
mkdir -p "${save_dir}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" video_qa/rekv_offline_vqa.py \
  --model "${MODEL}" \
  --sample_fps "${SAMPLE_FPS}" \
  --n_local "${RUN_N_LOCAL}" \
  --retrieve_size "${BASE_RS}" \
  --layer_retrieve_sizes "${layer_spec}" \
  --save_dir "${save_dir}" \
  --anno_path "${ANNO_PATH}" \
  --debug "${DEBUG}" \
  --save_retrieval_logits "${SAVE_RETRIEVAL_LOGITS}" \
  --num_chunks 1 \
  --chunk_idx 0

cp "${save_dir}/1_0.csv" "${save_dir}/results.csv"
"${PYTHON}" video_qa/eval/eval_lvbench.py --save_dir "${save_dir}"

"${PYTHON}" tools/summarize_lvbench_layer_ablation.py \
  --base_dir "${OUTPUT_ROOT}/${MODEL}/lvbench" \
  --ablation_subdir "${RUN_NAME}" \
  --sample_fps "${SAMPLE_FPS}" \
  --base_rs "${BASE_RS}" \
  --high_rs "${HIGH_RS}"
