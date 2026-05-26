#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL=${MODEL:-qwen2_5_vl_7b}
SAMPLE_FPS=${SAMPLE_FPS:-1.0}
RETRIEVE_SIZES=${RETRIEVE_SIZES:-"24 32 40 48 56"}
# Qwen2.5-VL uses 64 visual tokens per ReKV block at frame_size=224.
N_LOCAL=${N_LOCAL:-4096}
QWEN_BLOCK_SIZE=${QWEN_BLOCK_SIZE:-64}
DEBUG=${DEBUG:-false}
SAVE_RETRIEVAL_LOGITS=${SAVE_RETRIEVAL_LOGITS:-true}
NUM_CHUNKS=${NUM_CHUNKS:-2}
GPUS=${GPUS:-"0 1"}
PYTHON=${PYTHON:-/root/mwnoh/anaconda3/envs/rekv/bin/python}
ANNO_PATH=${ANNO_PATH:-data/lvbench/full_mc.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ssd1/mwnoh/LVBench/results}

read -r -a retrieve_sizes <<< "${RETRIEVE_SIZES}"
if (( ${#retrieve_sizes[@]} == 0 )); then
  echo "RETRIEVE_SIZES must not be empty." >&2
  exit 1
fi

MAX_RS=0
rs_label_parts=()
for rs in "${retrieve_sizes[@]}"; do
  if (( rs <= 0 )); then
    echo "retrieve size must be positive: ${rs}" >&2
    exit 1
  fi
  if (( rs > MAX_RS )); then
    MAX_RS="${rs}"
  fi
  rs_label_parts+=("rs${rs}")
done
rs_label="$(IFS=_; echo "${rs_label_parts[*]}")"
RUN_NAME=${RUN_NAME:-"fixed_${rs_label}_reuse"}

"${PYTHON}" tools/prepare_lvbench_data.py --output "${ANNO_PATH}"

min_n_local=$((MAX_RS * QWEN_BLOCK_SIZE))
RUN_N_LOCAL="${N_LOCAL}"
if (( RUN_N_LOCAL < min_n_local )); then
  RUN_N_LOCAL="${min_n_local}"
fi

save_root="${OUTPUT_ROOT}/${MODEL}/lvbench/${RUN_NAME}"
read -r -a gpu_list <<< "${GPUS}"
if (( ${#gpu_list[@]} < NUM_CHUNKS )); then
  echo "GPUS must contain at least NUM_CHUNKS entries: GPUS='${GPUS}', NUM_CHUNKS=${NUM_CHUNKS}" >&2
  exit 1
fi

echo "[lvbench-fixed-reuse] retrieve_sizes=${RETRIEVE_SIZES}, max_rs=${MAX_RS}, num_chunks=${NUM_CHUNKS}, gpus=${GPUS}, n_local=${RUN_N_LOCAL}, save_root=${save_root}"
mkdir -p "${save_root}"

pids=()
for chunk_idx in $(seq 0 $((NUM_CHUNKS - 1))); do
  gpu="${gpu_list[chunk_idx]}"
  echo "[lvbench-fixed-reuse] launch chunk ${chunk_idx}/${NUM_CHUNKS} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" video_qa/rekv_offline_vqa.py \
    --model "${MODEL}" \
    --sample_fps "${SAMPLE_FPS}" \
    --n_local "${RUN_N_LOCAL}" \
    --retrieve_size "${MAX_RS}" \
    --retrieve_sizes "${RETRIEVE_SIZES}" \
    --save_dir "${save_root}" \
    --anno_path "${ANNO_PATH}" \
    --debug "${DEBUG}" \
    --save_retrieval_logits "${SAVE_RETRIEVAL_LOGITS}" \
    --num_chunks "${NUM_CHUNKS}" \
    --chunk_idx "${chunk_idx}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if (( failed )); then
  echo "[lvbench-fixed-reuse] at least one chunk failed" >&2
  exit 1
fi

for rs in "${retrieve_sizes[@]}"; do
  run_dir="${save_root}/rs${rs}-${SAMPLE_FPS}"
  results_path="${run_dir}/results.csv"
  : > "${results_path}"
  for chunk_idx in $(seq 0 $((NUM_CHUNKS - 1))); do
    chunk_csv="${run_dir}/${NUM_CHUNKS}_${chunk_idx}.csv"
    if [[ ! -s "${chunk_csv}" ]]; then
      echo "missing or empty chunk output: ${chunk_csv}" >&2
      exit 1
    fi
    if (( chunk_idx == 0 )); then
      head -n 1 "${chunk_csv}" > "${results_path}"
    fi
    tail -n +2 "${chunk_csv}" >> "${results_path}"
  done
  "${PYTHON}" video_qa/eval/eval_lvbench.py --save_dir "${run_dir}"
done
