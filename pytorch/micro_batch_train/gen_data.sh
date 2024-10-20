#!/bin/bash

File=gen_data.py

# data=ogbn-papers100M
# data=ogbn-products
# data=cora
# data=pubmed
# data=reddit
# data=karate
# data=karate_iso

num_epoch=10
# fan_out=10
data=ogbn-arxiv

# mkdir ~/dataset/fan_out_10
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data


# 
fan_out=10,25
mkdir ~/dataset/fan_out_10,25
python3 $File --fan-out=$fan_out --num-layers=2 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data
#--------------------------------------------------------------------------------------------------------
# data=ogbn-products
# num_epoch=10
# fan_out=10
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data
# fan_out=15
# mkdir ~/dataset/fan_out_15
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data
# fan_out=20
# mkdir ~/dataset/fan_out_20
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data
# fan_out=800
# mkdir ~/dataset/fan_out_800
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data


# fan_out=10,25
# python $File --fan-out=$fan_out --num-layers=2 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data
# fan_out=10,50
# python $File --fan-out=$fan_out --num-layers=2 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data
# fan_out=10,100
# python $File --fan-out=$fan_out --num-layers=2 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data


# fan_out=10,25,30
# mkdir ~/dataset/fan_out_10,25,30
# python $File --fan-out=$fan_out --num-layers=3 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data


# fan_out=10,25,30,40
# mkdir ~/dataset/fan_out_10,25,30,40
# python $File --fan-out=$fan_out --num-layers=4 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data


# fan_out=10,25,30,40,50
# mkdir ~/dataset/fan_out_10,25,30,40,50
# python $File --fan-out=$fan_out --num-layers=5 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data
# data=karate
# fan_out=2
# mkdir ~/dataset/fan_out_2
# python $File --fan-out=$fan_out --num-layers=1 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data

# data=karate_iso
# fan_out=2,4
# mkdir ~/dataset/fan_out_2,4
# python3 $File --fan-out=$fan_out --num-layers=2 --num-epochs=$num_epoch --num-hidden=1 --dataset=$data

