#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-qwen2_5_vl_7b}
SAMPLE_FPS=${SAMPLE_FPS:-1.0}
# Qwen2.5-VL uses 64 visual tokens per ReKV block at frame_size=224.
# 64 local-window blocks => 64 * 64 = 4096 tokens.
N_LOCAL=${N_LOCAL:-4096}
QWEN_BLOCK_SIZE=${QWEN_BLOCK_SIZE:-64}
DEBUG=${DEBUG:-false}
SAVE_RETRIEVAL_LOGITS=${SAVE_RETRIEVAL_LOGITS:-true}
PYTHON=${PYTHON:-/root/mwnoh/anaconda3/envs/rekv/bin/python}
ANNO_PATH=${ANNO_PATH:-data/lvbench/full_mc.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ssd1/mwnoh/LVBench/results}

"${PYTHON}" tools/prepare_lvbench_data.py --output "${ANNO_PATH}"

run_one() {
  local rs="$1"
  local gpu="$2"
  local save_dir="${OUTPUT_ROOT}/${MODEL}/lvbench/${rs}-${SAMPLE_FPS}"
  local run_n_local="${N_LOCAL}"
  local min_n_local=$((rs * QWEN_BLOCK_SIZE))

  if (( run_n_local < min_n_local )); then
    run_n_local="${min_n_local}"
  fi

  echo "[lvbench] rs=${rs} -> GPU ${gpu}, n_local=${run_n_local}, save_dir=${save_dir}"
  mkdir -p "${save_dir}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" video_qa/rekv_offline_vqa.py \
    --model "${MODEL}" \
    --sample_fps "${SAMPLE_FPS}" \
    --n_local "${run_n_local}" \
    --retrieve_size "${rs}" \
    --save_dir "${save_dir}" \
    --anno_path "${ANNO_PATH}" \
    --debug "${DEBUG}" \
    --save_retrieval_logits "${SAVE_RETRIEVAL_LOGITS}" \
    --num_chunks 1 \
    --chunk_idx 0

  cp "${save_dir}/1_0.csv" "${save_dir}/results.csv"
  "${PYTHON}" video_qa/eval/eval_lvbench.py --save_dir "${save_dir}"
}

run_one 16 0 &
pid_rs16=$!
run_one 64 1 &
pid_rs64=$!

wait "${pid_rs16}"
wait "${pid_rs64}"

"${PYTHON}" tools/compare_lvbench_rs.py --base_dir "${OUTPUT_ROOT}/${MODEL}/lvbench" --sample_fps "${SAMPLE_FPS}"
