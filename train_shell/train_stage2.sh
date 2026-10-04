#!/usr/bin/env bash
set -euo pipefail

# Stage 2: resume stage 1 and continue with Gram anchoring to 300,000 minibatch steps.
# Use the same IMAGE_LIST and OUTPUT_DIR as stage 1. Keep automatic resume enabled.
IMAGE_LIST="${IMAGE_LIST:-path to the pretraining image list}"
OUTPUT_DIR="${OUTPUT_DIR:-path to the shared EchoDino training output directory}"
GRAM_TEACHER_CHECKPOINT="${GRAM_TEACHER_CHECKPOINT:-${OUTPUT_DIR}/eval/training_149999/teacher_checkpoint.pth}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# The trainer resumes from the latest numeric checkpoint directory in OUTPUT_DIR/ckpt.
checkpoint_found=false
for checkpoint_dir in "$OUTPUT_DIR"/ckpt/*; do
    if [[ -d "$checkpoint_dir" && "${checkpoint_dir##*/}" =~ ^[0-9]+$ ]]; then
        checkpoint_found=true
        break
    fi
done
if [[ "$checkpoint_found" != true ]]; then
    printf 'No training checkpoint found in %s/ckpt. Set OUTPUT_DIR to the stage-1 output directory.\n' "$OUTPUT_DIR" >&2
    exit 1
fi
if [[ ! -f "$GRAM_TEACHER_CHECKPOINT" ]]; then
    printf 'Gram teacher checkpoint not found: %s\n' "$GRAM_TEACHER_CHECKPOINT" >&2
    exit 1
fi

torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$MASTER_PORT" \
    "$REPO_ROOT/dinov3/train/train.py" \
    --config-file "$REPO_ROOT/dinov3/configs/train/echodino_stage2_gram_continue.yaml" \
    --output-dir "$OUTPUT_DIR" \
    "train.dataset_path=echodino:root=${IMAGE_LIST}:extra=unused" \
    "gram.ckpt=${GRAM_TEACHER_CHECKPOINT}"
