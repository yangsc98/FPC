#!/bin/bash
for SEED in 178 213 611
do
CUDA_VISIBLE_DEVICES=0 python -u mainRE.py \
    --train \
    --dataset_name tacred \
    --template EntTyp \
    --folder_name exp_$SEED \
    --downloaded_model \
    --use_gradient_checkpoint \
    --model_train mlmkg \
    --mlm_ratio 0.4 \
    --num_guide_epochs 5 \
    --seed $SEED
done
