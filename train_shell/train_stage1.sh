#!/usr/bin/env bash
set -euo pipefail

# Stage 1: initialize from DINOv3 and train for 150,000 minibatch steps.
# Edit these paths or supply them as environment variables.
IMAGE_LIST="${IMAGE_LIST:-path to the pretraining image list}"
OUTPUT_DIR="${OUTPUT_DIR:-path to the shared EchoDino training output directory}"
DINOV3_CHECKPOINT="${DINOV3_CHECKPOINT:-path to the pretrained DINOv3 ViT-L/16 model checkpoint}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$MASTER_PORT" \
    "$REPO_ROOT/dinov3/train/train.py" \
    --config-file "$REPO_ROOT/dinov3/configs/train/echodino_stage1_pretrain.yaml" \
    --output-dir "$OUTPUT_DIR" \
    "train.dataset_path=echodino:root=${IMAGE_LIST}:extra=unused" \
    "student.resume_from_plain_chkpt=${DINOV3_CHECKPOINT}"
