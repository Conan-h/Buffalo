#!/bin/bash

# mkdir ./log1
# save_path=../log/10_epoch/bucketing_optimized
save_path=./log_3-layer-10,25,30/SAGE/h_2048
mkdir $save_path
dataset=cora
hidden=2048
layer=3
fanout='10,25,30'

# for nb in 1 2 
for nb in 3 4 5 6
do
    echo "---start  $nb batches"
    python bucky_time_2.py \
        --dataset $dataset \
        --selection-method cora_30_backpack_bucketing \
        --num-batch $nb \
        --mem-constraint 7.5 \
        --num-layers $layer \
        --fan-out $fanout \
        --model SAGE \
        --num-hidden $hidden \
        --num-runs 1 \
        --num-epoch 20 \
        --aggre lstm \
        --log-indent 3 \
        --lr 1e-3 \
    > "${save_path}/nb_${nb}.log"
done

