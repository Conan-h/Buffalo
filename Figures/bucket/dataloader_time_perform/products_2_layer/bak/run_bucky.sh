#!/bin/bash

# mkdir ./log1
# save_path=../log/10_epoch/bucketing_optimized
save_path=./log_2-layer-10,25/SAGE/h_128
mkdir $save_path

hidden=128
for nb in 12 13 14 15 16 17 18 19 20 24 32
# for nb in 12 
do
    echo "---start products_25_time.py REG $nb batches"
    python time_products_SAMPLER.py \
        --dataset ogbn-products \
        --selection-method 25_backpack_products_bucketing \
        --num-batch $nb \
        --mem-constraint 18.1 \
        --num-layers 2 \
        --fan-out 10,25 \
        --num-hidden $hidden \
        --num-runs 1 \
        --num-epoch 20 \
        --aggre lstm \
        --log-indent 3 \
        --lr 1e-2 \
    > "${save_path}/nb_${nb}.log"
done

