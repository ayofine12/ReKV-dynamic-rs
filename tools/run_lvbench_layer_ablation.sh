#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-qwen2_5_vl_7b}
SAMPLE_FPS=${SAMPLE_FPS:-1.0}
BASE_RS=${BASE_RS:-16}
HIGH_RS=${HIGH_RS:-64}
# Qwen2.5-VL uses 64 visual tokens per ReKV block at frame_size=224.
N_LOCAL=${N_LOCAL:-4096}
QWEN_BLOCK_SIZE=${QWEN_BLOCK_SIZE:-64}
DEBUG=${DEBUG:-false}
SAVE_RETRIEVAL_LOGITS=${SAVE_RETRIEVAL_LOGITS:-false}
PYTHON=${PYTHON:-/root/mwnoh/anaconda3/envs/rekv/bin/python}
ANNO_PATH=${ANNO_PATH:-data/lvbench/full_mc.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ssd1/mwnoh/LVBench/results}
GPUS=${GPUS:-"0 1"}
ABLATIONS=${ABLATIONS:-"layer4=4:${HIGH_RS} layer5=5:${HIGH_RS} layer9=9:${HIGH_RS} layer11=11:${HIGH_RS} layer17=17:${HIGH_RS} layer21=21:${HIGH_RS} layers4-5=4-5:${HIGH_RS} layers9-11=9-11:${HIGH_RS}"}

"${PYTHON}" tools/prepare_lvbench_data.py --output "${ANNO_PATH}"

min_n_local=$((HIGH_RS * QWEN_BLOCK_SIZE))
RUN_N_LOCAL="${N_LOCAL}"
if (( RUN_N_LOCAL < min_n_local )); then
  RUN_N_LOCAL="${min_n_local}"
fi

read -r -a gpu_list <<< "${GPUS}"
if (( ${#gpu_list[@]} == 0 )); then
  echo "No GPU ids provided in GPUS." >&2
  exit 1
fi

run_one() {
  local label="$1"
  local spec="$2"
  local gpu="$3"
  local save_dir="${OUTPUT_ROOT}/${MODEL}/lvbench/layer_ablation_base${BASE_RS}_high${HIGH_RS}/${label}-${SAMPLE_FPS}"

  echo "[lvbench-ablation] ${label}: base_rs=${BASE_RS}, override=${spec}, gpu=${gpu}, n_local=${RUN_N_LOCAL}, save_dir=${save_dir}"
  mkdir -p "${save_dir}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" video_qa/rekv_offline_vqa.py \
    --model "${MODEL}" \
    --sample_fps "${SAMPLE_FPS}" \
    --n_local "${RUN_N_LOCAL}" \
    --retrieve_size "${BASE_RS}" \
    --layer_retrieve_sizes "${spec}" \
    --save_dir "${save_dir}" \
    --anno_path "${ANNO_PATH}" \
    --debug "${DEBUG}" \
    --save_retrieval_logits "${SAVE_RETRIEVAL_LOGITS}" \
    --num_chunks 1 \
    --chunk_idx 0

  cp "${save_dir}/1_0.csv" "${save_dir}/results.csv"
  "${PYTHON}" video_qa/eval/eval_lvbench.py --save_dir "${save_dir}"
}

read -r -a ablation_list <<< "${ABLATIONS}"
pids=()
slot=0

for entry in "${ablation_list[@]}"; do
  label="${entry%%=*}"
  spec="${entry#*=}"
  gpu="${gpu_list[$((slot % ${#gpu_list[@]}))]}"
  run_one "${label}" "${spec}" "${gpu}" &
  pids+=("$!")
  slot=$((slot + 1))

  if (( ${#pids[@]} >= ${#gpu_list[@]} )); then
    wait "${pids[@]}"
    pids=()
  fi
done

if (( ${#pids[@]} > 0 )); then
  wait "${pids[@]}"
fi

"${PYTHON}" tools/summarize_lvbench_layer_ablation.py \
  --base_dir "${OUTPUT_ROOT}/${MODEL}/lvbench" \
  --ablation_subdir "layer_ablation_base${BASE_RS}_high${HIGH_RS}" \
  --sample_fps "${SAMPLE_FPS}" \
  --base_rs "${BASE_RS}" \
  --high_rs "${HIGH_RS}"
