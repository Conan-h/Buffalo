#!/bin/bash

hidden=128
save_path=./data_parepare_log

for nb in  12 16 18
do
    echo "---start  hidden ${hidden},  nb ${nb} batches"
    python3 distributed_buffalo_data_prepare.py  \
        --dataset ogbn-products \
        --num-batch ${nb} \
        --num-layers 2 \
        --fan-out 10,25 \
        --num-hidden ${hidden} \
        --num-runs 1 \
        --num-epoch 1 \
        --aggre lstm \
        --log-indent 3 \
        --lr 1e-2 \
        > ${save_path}/nb_${nb}_hidden_${hidden}.log
done


