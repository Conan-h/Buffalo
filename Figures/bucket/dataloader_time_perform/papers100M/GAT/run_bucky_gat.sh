#!/bin/bash

np=20

hidden=256
# dataset=cora
# dataset=pubmed
# dataset=ogbn-arxiv
# name=arxiv
# dataset=reddit
# name=reddit
dataset=ogbn-papers100M
name=papers100M
# method=${name}_25_backpack_bucketing
method=range_bucketing
save_path=./bucket_log/${hidden}/${np}_epochs/
# mkdir $save_path

lr=1e-3
# # Cora---------------------
# nb=1
# capacity=0.38
# nb=2
# capacity=0.19

# # pubmed---------------------
# nb=1
# capacity=0.32
# nb=2
# capacity=0.16

# # arxiv---------------------
# nb=20
# capacity=18

# # Reddit---------------------
# nb=180
# capacity=18


# ogbn-papers100M------------
nb=18
capacity=70

python3 bucky_GAT.py \
    --dataset $dataset \
    --selection-method $method \
    --num-batch $nb \
    --mem-constraint $capacity\
    --num-layers 2 \
    --fan-out 10,25 \
    --num-hidden $hidden \
    --num-runs 1 \
    --num-epoch $np \
    --aggre lstm \
    --log-indent 0 \
    --lr $lr \
    > ${save_path}/train_loss_nb_${nb}_bucketing_h_${hidden}_method_${method}.log
