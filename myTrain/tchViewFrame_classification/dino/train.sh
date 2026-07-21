#!/bin/bash


#############################################################################
## For EchoDino
#############################################################################


unset CKPT_DICT
declare -A CKPT_DICT

CKPT_DICT["my299999_210"]="/rdf/projects/echoDinoCkpt/3_1_continue/training_299999/teacher_checkpoint.pth"

for LR in 1e-3; do
  for CKPT_NAME in "${!CKPT_DICT[@]}"; do
    CKPT_PATH="${CKPT_DICT[$CKPT_NAME]}"
    for ENC_STATUS in "false"; do
      for MODEL_TYPE in "dino_tchViewFrame_cls"; do
        if [ "$ENC_STATUS" == "true" ]; then
            BATCH_SIZE=16
        else
            BATCH_SIZE=128
        fi

        EXP_NAME="${MODEL_TYPE}_${CKPT_NAME}_TchViewFrameCLS_Encoder${ENC_STATUS}_LR${LR}"
        OUTPUT_DIR="/data/sc159/EchoDino/output/TCHViewFrameCLS/${MODEL_TYPE}/Ckpt_${CKPT_NAME}/Encoder_${ENC_STATUS}/LR_${LR}"

        CMD="PYTHONPATH=. python -m accelerate.commands.launch \
            --num_processes=2 \
            --main_process_port=29501 \
            /data/sc159/EchoDino/myTrain/stage2_tchViewFrame_classification/dino/train.py \
            --model_type ${MODEL_TYPE} \
            --ckpt_path ${CKPT_PATH} \
            --epochs 30 \
            --warmup_epochs 3 \
            --batch_size ${BATCH_SIZE} \
            --lr ${LR} \
            --wandb_project_name DINO_TchViewFrameCLS \
            --experiment_name ${EXP_NAME} \
            --output_dir ${OUTPUT_DIR}"

        if [ "$ENC_STATUS" == "true" ]; then
            CMD="$CMD --train_encoder"
        fi

        eval $CMD

        EVAL_CMD="PYTHONPATH=. python \
        /data/sc159/EchoDino/myTrain/stage2_tchViewFrame_classification/dino/eval.py \
        --model_type ${MODEL_TYPE} \
        --whole_model_path ${OUTPUT_DIR}/best_model.pth \
        --split test"

        eval $EVAL_CMD

        sleep 20
      done
    done
  done
done

