#!/bin/bash
save_path=./betty_log/
hidden=128
for md in REG
do

    save_path=./betty_log/${md}
    # for nb in 16 24 32 17 18 19 20 21 22
    # for nb in  7 8 9 10 11 12 13 14 15 16 24 32 17 18 19 20
    for nb in  17 18 19 20
    do
        echo "---start  hidden ${hidden},  nb ${nb} batches"
        python Betty_e2e_time.py  \
            --dataset ogbn-products \
            --selection-method $md \
            --num-batch ${nb} \
            --num-layers 2 \
            --fan-out 10,25 \
            --num-hidden ${hidden} \
            --num-runs 1 \
            --num-epoch 20 \
            --aggre lstm \
            --log-indent 3 \
            --lr 1e-2 \
            > ${save_path}/nb_${nb}_e2e_${hidden}.log
    done
done

