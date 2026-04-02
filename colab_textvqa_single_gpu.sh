set -x

export PYTHONPATH="$(pwd):${PYTHONPATH}"

SMALL_CHECKPOINT=${SMALL_CHECKPOINT:?Set SMALL_CHECKPOINT to your small model path}
LARGE_CHECKPOINT=${LARGE_CHECKPOINT:?Set LARGE_CHECKPOINT to your large model path}
DATASETS=${DATASETS:-textvqa_val}
OUT_DIR=${OUT_DIR:-/content/drive/MyDrive/sgl-results}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-/content/model_ckpts}
SMALL_ATTENTION_LAYER_RANGE=${SMALL_ATTENTION_LAYER_RANGE:-all}
LARGE_MODEL_PRUNE_LAYER=${LARGE_MODEL_PRUNE_LAYER:-0.4}
LARGE_MODEL_PRUNE_RATIO=${LARGE_MODEL_PRUNE_RATIO:-0.4}
LOAD_FLAGS=${LOAD_FLAGS:-}

resolve_model_dir() {
    local input_path="$1"

    if [ -f "${input_path}/config.json" ]; then
        echo "${input_path}"
        return 0
    fi

    if [ -d "${input_path}/snapshots" ]; then
        local snapshot_dir
        snapshot_dir=$(find "${input_path}/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n 1)
        if [ -n "${snapshot_dir}" ] && [ -f "${snapshot_dir}/config.json" ]; then
            echo "${snapshot_dir}"
            return 0
        fi
    fi

    local parent_dir base_name cache_match snapshot_dir
    parent_dir=$(dirname "${input_path}")
    base_name=$(basename "${input_path}")
    cache_match=$(find "${parent_dir}" -mindepth 1 -maxdepth 1 -type d -name "models--*--${base_name}" | head -n 1)
    if [ -n "${cache_match}" ]; then
        if [ -f "${cache_match}/config.json" ]; then
            echo "${cache_match}"
            return 0
        fi
        snapshot_dir=$(find "${cache_match}/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n 1)
        if [ -n "${snapshot_dir}" ] && [ -f "${snapshot_dir}/config.json" ]; then
            echo "${snapshot_dir}"
            return 0
        fi
    fi

    echo "Could not resolve a model directory from: ${input_path}" >&2
    echo "Look for a folder containing config.json or a Hugging Face cache folder with snapshots/." >&2
    return 1
}

download_model_dir() {
    local model_spec="$1"
    python - "$model_spec" "$MODEL_CACHE_DIR" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

model_spec = sys.argv[1]
cache_dir = sys.argv[2]

if "/" in model_spec:
    repo_id = model_spec
elif model_spec.startswith("InternVL"):
    repo_id = f"OpenGVLab/{model_spec}"
else:
    raise SystemExit(
        f"Cannot infer a Hugging Face repo id from '{model_spec}'. "
        "Use a local path, a full repo id like OpenGVLab/InternVL2-2B, or a bare InternVL model name."
    )

local_dir = os.path.join(cache_dir, repo_id.split("/")[-1])
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(local_dir)
PY
}

resolve_or_download_model_dir() {
    local model_spec="$1"
    local resolved_dir

    if resolved_dir=$(resolve_model_dir "${model_spec}" 2>/dev/null); then
        echo "${resolved_dir}"
        return 0
    fi

    if [[ "${model_spec}" = /* ]] || [[ "${model_spec}" = ./* ]] || [[ "${model_spec}" = ../* ]]; then
        echo "Could not resolve local model directory: ${model_spec}" >&2
        return 1
    fi

    mkdir -p "${MODEL_CACHE_DIR}"
    resolved_dir=$(download_model_dir "${model_spec}") || return 1
    resolve_model_dir "${resolved_dir}"
}

SMALL_CHECKPOINT=$(resolve_or_download_model_dir "${SMALL_CHECKPOINT}")
LARGE_CHECKPOINT=$(resolve_or_download_model_dir "${LARGE_CHECKPOINT}")

echo "Resolved SMALL_CHECKPOINT=${SMALL_CHECKPOINT}"
echo "Resolved LARGE_CHECKPOINT=${LARGE_CHECKPOINT}"

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
