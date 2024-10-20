#!/bin/bash


save_path=./log_1-layer_10/SAGE/h_602
mkdir $save_path

hidden=602
# for nb in 3 4 5 6 7 8
for nb in 4 
do
    echo "---start  $nb batches"
    python reddit_backpack_bucketing.py \
        --dataset reddit \
        --selection-method reddit_10_backpack_bucketing \
        --num-batch $nb \
        --mem-constraint 18.1 \
        --num-layers 1 \
        --fan-out 10 \
        --num-hidden $hidden \
        --num-runs 1 \
        --num-epoch 20 \
        --aggre lstm \
        --log-indent 3 \
        --lr 1e-4 \
    > "${save_path}/nb_${nb}.log"
done

