#!/bin/bash

# mkdir ./log1
# save_path=../log/10_epoch/bucketing_optimized
save_path=./metis/log_2-layer-10,25/SAGE/h_256
mkdir $save_path

dataset=ogbn-papers100M
name=papers100M
hidden=256
np=20
for nb in 8 9 10 15 24  
do
    echo "---start  $nb batches"
    python3 bucky_time.py \
        --dataset $dataset \
        --selection-method metis_bucketing \
        --num-batch $nb \
        --mem-constraint 72 \
        --num-layers 2 \
        --fan-out 10,25 \
        --num-hidden $hidden \
        --num-runs 1 \
        --num-epoch $np \
        --aggre lstm \
        --log-indent 3 \
        --lr 1e-2 \
    > "${save_path}/nb_${nb}.log"
done

