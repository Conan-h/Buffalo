from collections import Counter
				

def info_collection(b_block_dataloader):
	data_dict = []
	print('redundancy ratio #input/#seeds/degree')
	redundant_ratio = []
	for step, (input_nodes, seeds, blocks) in enumerate(b_block_dataloader):
		print(len(input_nodes)/len(seeds)/(step+1))
		redundant_ratio.append(len(input_nodes)/len(seeds)/(step+1))

	for step, (input_nodes, seeds, blocks) in enumerate(b_block_dataloader):
		layer = 0
		dict_list =[]
		for b in blocks:
			print('layer ', layer)
			graph_in = dict(Counter(b.in_degrees().tolist()))
			graph_in = dict(sorted(graph_in.items()))

			print(graph_in)
			dict_list.append(graph_in)

			layer = layer +1
		print()
		data_dict.append(dict_list)
	print('data_dict')
	print(data_dict)
	return data_dict, redundant_ratio

def estimate_mem_2_layer(data_dict, in_feat, hidden_size, redundant_ratio):	
	
	estimated_mem_list = []
	for deg, data in enumerate(data_dict):
		estimated_mem = 0
		for i in range (len(data)):
			sum_b = 0
			for idx, (key, val) in enumerate(data[i].items()):
				sum_b = sum_b + key*val
				if idx ==0: # the input layer, in_feat 100
					estimated_mem  +=  sum_b*in_feat*18*4/1024/1024/1024
				if idx ==1: # the output layer
					estimated_mem  +=  sum_b*hidden_size*18*4/1024/1024/1024	
		estimated_mem_list.append(estimated_mem)

	modified_estimated_mem_list = []
	for deg in range(len(redundant_ratio)):
		modified_estimated_mem_list.append(estimated_mem_list[deg]*redundant_ratio[deg]) 
		# redundant_ratio[i] is a variable depends on graph characteristic
		print(' MM estimated memory/GB degree '+str(deg)+': '+str(estimated_mem_list[deg]) + " * " +str(redundant_ratio[deg]) ) 
	

	return modified_estimated_mem_list, estimated_mem_list

def MEM_EST(in_feat, hidden, b_block_dataloader):
	data_dict, redundant_ratio = info_collection(b_block_dataloader)
	modified_res, res = estimate_mem_2_layer(data_dict, in_feat, hidden, redundant_ratio)
	return modified_res
				