#!/bin/bash
File=AA_loss_log.py
# it's the log version of multi_layer_bucket_loss.py
Data=cora

model=sage
seed=1236 
setseed=True
GPUmem=True
load_full_batch=True
lr=0.01
dropout=0.5
pMethodList=(random_bucketing)
run=1
epoch=1
logIndent=0

layersList=(2)
fan_out_list=(10,25)

hiddenList=(128 )
AggreList=(mean )
num_batch=(1 2 3)

# mkdir ./log1
# mkdir ./log1/bucketing
save_path=./log1/bucketing
# mkdir $save_path

for Aggre in ${AggreList[@]}
do      
	
	for pMethod in ${pMethodList[@]}
	do      
		
			for layers in ${layersList[@]}
			do      
				for hidden in ${hiddenList[@]}
				do
					for fan_out in ${fan_out_list[@]}
					do
						
						for nb in ${num_batch[@]}
						do
							
							
							echo 'number of batches equals '${nb}
							python $File \
							--dataset $Data \
							--aggre $Aggre \
							--seed $seed \
							--setseed $setseed \
							--GPUmem $GPUmem \
							--selection-method $pMethod \
							--num-batch $nb \
							--lr $lr \
							--num-runs $run \
							--num-epochs $epoch \
							--num-layers $layers \
							--num-hidden $hidden \
							--dropout $dropout \
							--fan-out $fan_out \
							--log-indent $logIndent \
							--load-full-batch True \
							> ${save_path}/${layers}_layer_aggre_${Aggre}_batch_${nb}.log

							
						done
					done
				done
			done
		
	done
done
