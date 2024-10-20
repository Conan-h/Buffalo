#!/bin/bash

# mkdir ./log1
# save_path=../log/10_epoch/bucketing_optimized
save_path=./range_random_metis/SAGE/h_128
mkdir $save_path

hidden=128
folder=range
for nb in 14 15 16 17 18 19 20 22 24 32
do
    echo "---start  $nb batches"
    python bucky_time_2.py \
        --dataset ogbn-products \
        --selection-method range_bucketing \
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
    > "${save_path}/${folder}/nb_${nb}.log"
done

folder=random
for nb in  17 18 19 20 22 
do
    echo "---start  $nb batches"
    python bucky_time_2.py \
        --dataset ogbn-products \
        --selection-method random_bucketing \
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
    > "${save_path}/${folder}/nb_${nb}.log"
done

# folder=metis
# for nb in 12 13 14 15 16 17 18 19 20 22 24 32
# do
#     echo "---start metis  $nb batches"
#     python bucky_time_2.py \
#         --dataset ogbn-products \
#         --selection-method metis_bucketing \
#         --num-batch $nb \
#         --mem-constraint 18.1 \
#         --num-layers 2 \
#         --fan-out 10,25 \
#         --num-hidden $hidden \
#         --num-runs 1 \
#         --num-epoch 20 \
#         --aggre lstm \
#         --log-indent 3 \
#         --lr 1e-2 \
#     > "${save_path}/${folder}/nb_${nb}.log"
# done