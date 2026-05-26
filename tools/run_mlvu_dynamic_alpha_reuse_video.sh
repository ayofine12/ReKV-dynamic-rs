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
NUM_CHUNKS=${NUM_CHUNKS:-4}
GPUS=${GPUS:-"0 1 2 3"}
PYTHON=${PYTHON:-/root/mwnoh/anaconda3/envs/rekv/bin/python}
MLVU_ROOT=${MLVU_ROOT:-/mnt/ssd1/mwnoh/MLVU/MLVU}
ANNO_PATH=${ANNO_PATH:-data/mlvu/full_mc.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ssd1/mwnoh/MLVU/results}
ALPHAS=${ALPHAS:-"0.2 0.25 0.3 0.35"}
# MLVU contains several 2h+ videos and one ~9h surveillance video. The default
# skips videos longer than 1 hour. Set MAX_DURATION= to include all videos.
MAX_DURATION=${MAX_DURATION:-3600}
if [[ -z "${RUN_NAME:-}" ]]; then
  RUN_NAME="dynamic_mass_min${MIN_RS}_max${MAX_RS}"
  if [[ -n "${MAX_DURATION}" ]]; then
    RUN_NAME="${RUN_NAME}_maxdur${MAX_DURATION}"
  fi
fi

prepare_args=(tools/prepare_mlvu_data.py --input-root "${MLVU_ROOT}" --output "${ANNO_PATH}")
if [[ -n "${MAX_DURATION}" ]]; then
  prepare_args+=(--max-duration "${MAX_DURATION}")
fi
prepare_args+=(--balance-chunks "${NUM_CHUNKS}")
"${PYTHON}" "${prepare_args[@]}"

min_n_local=$((MAX_RS * QWEN_BLOCK_SIZE))
RUN_N_LOCAL="${N_LOCAL}"
if (( RUN_N_LOCAL < min_n_local )); then
  RUN_N_LOCAL="${min_n_local}"
fi

save_root="${OUTPUT_ROOT}/${MODEL}/mlvu/${RUN_NAME}"
read -r -a gpu_list <<< "${GPUS}"
if (( ${#gpu_list[@]} < NUM_CHUNKS )); then
  echo "GPUS must contain at least NUM_CHUNKS entries: GPUS='${GPUS}', NUM_CHUNKS=${NUM_CHUNKS}" >&2
  exit 1
fi

echo "[mlvu-dynamic-reuse] alphas=${ALPHAS}, min_rs=${MIN_RS}, max_rs=${MAX_RS}, num_chunks=${NUM_CHUNKS}, gpus=${GPUS}, n_local=${RUN_N_LOCAL}, save_root=${save_root}"
mkdir -p "${save_root}"

pids=()
for chunk_idx in $(seq 0 $((NUM_CHUNKS - 1))); do
  gpu="${gpu_list[chunk_idx]}"
  echo "[mlvu-dynamic-reuse] launch chunk ${chunk_idx}/${NUM_CHUNKS} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" video_qa/rekv_offline_vqa.py \
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
  echo "[mlvu-dynamic-reuse] at least one chunk failed" >&2
  exit 1
fi

for alpha in ${ALPHAS}; do
  alpha_label=${alpha//./p}
  run_dir="${save_root}/alpha${alpha_label}-${SAMPLE_FPS}"
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
  "${PYTHON}" video_qa/eval/eval_multiple_choice.py --save_dir "${run_dir}"
done
