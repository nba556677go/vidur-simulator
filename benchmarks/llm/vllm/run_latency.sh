#!/usr/bin/env bash
set -euo pipefail

# OUTPUT directory
OUTPUT_DIR=latency
mkdir -p "${OUTPUT_DIR}"


#DEBUG flags
export NCCL_DEBUG=INFO 
export NCCL_P2P_DISABLE=1

# === Configurable parameters ===
MODEL="Qwen/Qwen3-8B"
MODEL="Qwen/Qwen3-0.6B"
#MODEL="openai-community/gpt2-xl"
DTYPE="float16"
INPUT_LEN=4000
OUTPUT_LEN=50
BATCH_SIZE=8
MAX_NUM_BATCHED_TOKENS=8192
N=5
TP=1
DP=2
PP=1
BACKEND="ray"
IMPL="transformers"
ITER_WARMUP=2
ITER=5

# derive a filename base from the above
# Strip forward slashes from model name for filename
MODEL_FNAME="$(echo ${MODEL} | tr -d '/')" 
FNAME="latency_model${MODEL_FNAME}_bs${BATCH_SIZE}_dp${DP}_tp${TP}_n${N}_${DTYPE}_warmup${ITER_WARMUP}_backend${BACKEND}"
mkdir -p "${OUTPUT_DIR}/${MODEL_FNAME}"
LOGFILE="${OUTPUT_DIR}/${MODEL_FNAME}/${FNAME}.log"
JSONFILE="${OUTPUT_DIR}/${MODEL_FNAME}/${FNAME}.json"

# === Run & tee logs ===
vllm bench latency \
  --model "${MODEL}" \
  --dtype "${DTYPE}" \
  --input-len ${INPUT_LEN} \
  --output-len ${OUTPUT_LEN} \
  --batch-size ${BATCH_SIZE} \
  --n ${N} \
  --tensor-parallel-size ${TP} \
  --data-parallel-size ${DP} \
  --pipeline-parallel-size ${PP} \
  --distributed-executor-backend ${BACKEND} \
  --no-enable-chunked-prefill \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
  --model-impl ${IMPL} \
  --num-iters-warmup ${ITER_WARMUP} \
  --num-iters ${ITER} \
  --output-json "${JSONFILE}" \
  2>&1 | tee "${LOGFILE}"
