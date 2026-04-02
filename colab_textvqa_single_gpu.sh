set -x

export PYTHONPATH="$(pwd):${PYTHONPATH}"

SMALL_CHECKPOINT=${SMALL_CHECKPOINT:?Set SMALL_CHECKPOINT to your small model path}
LARGE_CHECKPOINT=${LARGE_CHECKPOINT:?Set LARGE_CHECKPOINT to your large model path}
DATASETS=${DATASETS:-textvqa_val}
OUT_DIR=${OUT_DIR:-/content/drive/MyDrive/sgl-results}
SMALL_ATTENTION_LAYER_RANGE=${SMALL_ATTENTION_LAYER_RANGE:-all}
LARGE_MODEL_PRUNE_LAYER=${LARGE_MODEL_PRUNE_LAYER:-0.4}
LARGE_MODEL_PRUNE_RATIO=${LARGE_MODEL_PRUNE_RATIO:-0.4}
LOAD_FLAGS=${LOAD_FLAGS:-}

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=1 \
    --master_port=29500 \
    eval/vqa/evaluate_vqa.py \
    --small_checkpoint "${SMALL_CHECKPOINT}" \
    --large_checkpoint "${LARGE_CHECKPOINT}" \
    --datasets "${DATASETS}" \
    --dynamic \
    --out-dir "${OUT_DIR}" \
    --small_model_attention_layer_range "${SMALL_ATTENTION_LAYER_RANGE}" \
    --large_model_prune_layer "${LARGE_MODEL_PRUNE_LAYER}" \
    --large_model_prune_ratio "${LARGE_MODEL_PRUNE_RATIO}" \
    ${LOAD_FLAGS}
