#!/bin/bash

# mkdir ./log1
# save_path=../log/10_epoch/bucketing_optimized
save_path=./log_2-layer-10,25/SAGE/h_1024
mkdir $save_path

hidden=1024
for nb in 4 5 6 7 8 9 10 11 12 16 32
do
    echo "---start  $nb batches"
    python bucky_time_2.py \
        --dataset ogbn-arxiv \
        --selection-method arxiv_25_backpack_bucketing \
        --num-batch $nb \
        --mem-constraint 18.1 \
        --num-layers 2 \
        --fan-out 10,25 \
        --num-hidden $hidden \
        --num-runs 1 \
        --num-epoch 20 \
        --aggre lstm \
        --log-indent 3 \
        --lr 1e-3 \
    > "${save_path}/nb_${nb}.log"
done

