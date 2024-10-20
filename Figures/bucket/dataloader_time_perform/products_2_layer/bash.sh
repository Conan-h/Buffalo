export MASTER_ADDR=localhost  # Address of the master node (can be the IP if distributed over multiple machines)  
export MASTER_PORT=12346       # Port used for communication  
export WORLD_SIZE=2            # Total number of processes (GPUs)  
export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_LAUNCH_BLOCKING=1

# Run each process with corresponding RANK  
for rank in $(seq 0 1); do  
    CUDA_VISIBLE_DEVICES=$rank RANK=$rank MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE python3 distributed_buffalo.py &  
done  

wait 


# export MASTER_ADDR=localhost  # Address of the master node (can be the IP if distributed over multiple machines)  
# export MASTER_PORT=12346       # Port used for communication  
# export WORLD_SIZE=2            # Total number of processes (GPUs)  
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_LAUNCH_BLOCKING=2

# # Run each process with corresponding RANK  
# for rank in $(seq 0 ); do  
#     CUDA_VISIBLE_DEVICES=$rank RANK=$rank MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE python3 distributed_buffalo.py &  
# done  

# wait 