#!/bin/bash

torchrun --nnodes=1 --nproc_per_node=8 train.py \
       --dataset DCI \
       --i2t_data_path /path/to/i2t_preference_dataset \
       --t2i_data_path /path/to/t2i_preference_dataset \
       --batch_size 128 \
       --accumulation_steps 2 \
       --grad_clip 0 \
       --weight_decay 0 \
       --epochs 2 \
       --distributed True \
       --clear_visualizer \
       --warmup 0 \
       --valid_per_epoch 4 \
       --max_length 128 \
       --lr 2e-5 \
       --lr-decay-style cosine \
       --fix_rate 0.7 \
       --data_ratio 1 \
       --threshold_similar 0.005 \
       --threshold_negative 0.7 