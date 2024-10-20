import torch
import dgl
import numpy as np
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/utils/')
from utils import get_in_degree_bucketing
import numpy
import time
import pickle
import io
from math import ceil
from math import floor
from math import ceil
from itertools import islice
from statistics import mean
from multiprocessing import Manager, Pool
from multiprocessing import Process, Value, Array
import multiprocessing as mp
from functools import partial


from gat_bucket_partitioner import GAT_Bucket_Partitioner


from my_utils import gen_batch_output_list
from memory_usage import see_memory_usage

from sortedcontainers import SortedList, SortedSet, SortedDict
from multiprocessing import Process, Queue
from collections import Counter, OrderedDict
import copy
from typing import Union, Collection
from my_utils import torch_is_in_1d
sys.path.insert(0, './pybind_mp')
import remove_values
import pdb

sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing/pybind_remove_duplicates')
import remove_duplicates

# sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing/global_2_local')
# import find_indices
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing/gen_tails')
import gen_tails
# sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing/src_gen')
# import src_gen
# sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing/gen_src_tail')
# import gen_src_tails

class OrderedCounter(Counter, OrderedDict):
	'Counter that remembers the order elements are first encountered'

	def __repr__(self):
		return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

	def __reduce__(self):
		return self.__class__, (OrderedDict(self),)



def get_global_graph_edges_ids_block(raw_graph, block):

	edges=block.edges(order='eid', form='all')
	edge_src_local = edges[0]
	edge_dst_local = edges[1]
	# edge_eid_local = edges[2]
	induced_src = block.srcdata[dgl.NID]
	induced_dst = block.dstdata[dgl.NID]
	induced_eid = block.edata[dgl.EID]

	raw_src, raw_dst=induced_src[edge_src_local], induced_dst[edge_dst_local]
	# raw_src, raw_dst=induced_src[edge_src_local], induced_src[edge_dst_local]

	# in homo graph: raw_graph
	global_graph_eids_raw = raw_graph.edge_ids(raw_src, raw_dst)
	# https://docs.dgl.ai/generated/dgl.DGLGraph.edge_ids.html?highlight=graph%20edge_ids#dgl.DGLGraph.edge_ids
	# https://docs.dgl.ai/en/0.4.x/generated/dgl.DGLGraph.edge_ids.html#dgl.DGLGraph.edge_ids

	return global_graph_eids_raw, (raw_src, raw_dst)

def generate_one_block(raw_graph, global_srcnid, global_dstnid, global_eids):
	'''

	Parameters
	----------
	G    global graph                     DGLGraph
	eids  cur_batch_subgraph_global eid   tensor int64

	Returns
	-------

	'''
	_graph = dgl.edge_subgraph(raw_graph, global_eids, store_ids=True)
	edge_dst_list = _graph.edges(order='eid')[1].tolist()
	# dst_local_nid_list=list(OrderedCounter(edge_dst_list).keys())
	dst_local_nid_list = remove_duplicates.remove_duplicates(edge_dst_list) # speedup version

	new_block = dgl.to_block(_graph, dst_nodes=torch.tensor(dst_local_nid_list, dtype=torch.long))
	new_block.srcdata[dgl.NID] = global_srcnid
	new_block.dstdata[dgl.NID] = global_dstnid
	new_block.edata['_ID']=_graph.edata['_ID']

	return new_block



def check_connections_block(g_batched_nodes_list, current_layer_block):
    
	print('check connections block*********************************')

	induced_src = current_layer_block.srcdata[dgl.NID]
	eids_global = current_layer_block.edata['_ID']

	src_nid_list = induced_src.tolist()

	print('')
	global_batched_nids_list = [nid.tolist() for nid in g_batched_nodes_list]
	timess = time.time()
	# global_batched_nids_list = [nid.tolist() for nid in batched_nodes_list]
	# output_nid_list = find_indices.find_indices(src_nid_list, global_batched_nids_list)
	dict_nid_2_local = dict(zip(src_nid_list, range(len(src_nid_list)))) # speedup global to local
	local_output_nid_list =[]
	for step, output_nid in enumerate(g_batched_nodes_list):
		# in current layer subgraph, only has src and dst nodes,
		# and src nodes includes dst nodes, src nodes equals dst nodes.
		if torch.is_tensor(output_nid): output_nid = output_nid.tolist()
		local_output_nid = list(map(dict_nid_2_local.get, output_nid))
		local_output_nid_list.append(local_output_nid)
	# print('bucketing dataloader: connection check: local output nid list ', local_output_nid_list)
	print('the find indices time spent ', time.time()-timess)
	
	# in current layer subgraph, only has src and dst nodes,
	# and src nodes includes dst nodes, src nodes equals dst nodes.
	print()
	
	
	time1= time.time()
	# dgl graph.in_edges() sequential
	local_in_edges_tensor_list=[]
	for step, local_output_nid in enumerate(local_output_nid_list):
		local_in_edges_tensor = current_layer_block.in_edges(local_output_nid, form='all')
		local_in_edges_res = [id.tolist() for id in local_in_edges_tensor]
		local_in_edges_tensor_list.append(local_in_edges_res)
	time2=time.time()
	print('in edges time spent ', time2-time1)

	time31=time.time()
	
	eids_list = []
	src_long_list = []

	for local_in_edges_tensor, global_output_nid in (zip(local_in_edges_tensor_list, global_batched_nids_list)):
		mini_batch_src_local= local_in_edges_tensor[0] # local (𝑈,𝑉,𝐸𝐼𝐷);
		# mini_batch_src_local = list(dict.fromkeys(mini_batch_src_local))
		mini_batch_src_local = remove_duplicates.remove_duplicates(mini_batch_src_local)
		mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.


		eid_local_list = local_in_edges_tensor[2] # local (𝑈,𝑉,𝐸𝐼𝐷); 
		global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
		eids_list.append(global_eid_tensor)
		src_long_list.append(mini_batch_src_global)
		# print('bucketing dataloader: connection check:  mini_batch_src_local ', mini_batch_src_local)
		# print('bucketing dataloader: connection check:  mini_batch_src_global ', mini_batch_src_global)
	time32 = time.time()
	print('local to global src and eids time spent ', time32-time31)
	
	time33 = time.time()
	tails_list = gen_tails.gen_tails(src_long_list, global_batched_nids_list)
	time34 = time.time()
	print('time gen tails ', time34-time33)
	res =[]
	for global_output_nid, r_,eid  in zip(global_batched_nids_list,tails_list,eids_list):
		src_nid = torch.tensor(global_output_nid + r_, dtype=torch.long)
		output_nid = torch.tensor(global_output_nid, dtype=torch.long)

		res.append((src_nid, output_nid, eid))
	# parallel-------------------------------------------------end
	print("res  length", len(res))
	return res


# def check_connections_block(batched_nodes_list, current_layer_block):
# 	str_=''
# 	res=[]
# 	print('check_connections_block*********************************')

# 	induced_src = current_layer_block.srcdata[dgl.NID]
# 	eids_global = current_layer_block.edata['_ID']

# 	t1=time.time()
# 	src_nid_list = induced_src.tolist()
# 	# print('src_nid_list ', src_nid_list)
# 	# the order of srcdata in current block is not increased as the original graph. For example,
# 	# src_nid_list  [1049, 432, 741, 554, ... 1683, 1857, 1183, ... 1676]
# 	# dst_nid_list  [1049, 432, 741, 554, ... 1683]
	
# 	dict_nid_2_local = dict(zip(src_nid_list, range(len(src_nid_list)))) # speedup 
# 	str_+= 'time for parepare 1: '+str(time.time()-t1)+'\n'


# 	for step, output_nid in enumerate(batched_nodes_list):
# 		# in current layer subgraph, only has src and dst nodes,
# 		# and src nodes includes dst nodes, src nodes equals dst nodes.
# 		tt=time.time()
# 		output_nid = output_nid.tolist()
# 		local_output_nid = list(map(dict_nid_2_local.get, output_nid))
# 		# print('connection check : local_output_nid ', local_output_nid)
# 		print('connection check : local_output_nid ', len(local_output_nid))
# 		str_+= 'local_output_nid generation: '+ str(time.time()-tt)+'\n'
# 		tt1=time.time()

# 		local_in_edges_tensor = current_layer_block.in_edges(local_output_nid, form='all')
# 		# print('local_in_edges_tensor', local_in_edges_tensor)
# 		str_+= 'local_in_edges_tensor generation: '+str(time.time()-tt1)+'\n'
		
# 		# return (𝑈,𝑉,𝐸𝐼𝐷)
# 		# get local srcnid and dstnid from subgraph
# 		mini_batch_src_local= list(local_in_edges_tensor)[0] # local (𝑈,𝑉,𝐸𝐼𝐷);
# 		str_+= "\n&&&&&&&&&&&&&&& before remove duplicate length: "+ str(len(mini_batch_src_local))+'\n'
# 		ttpp=time.time()
# 		# print('mini_batch_src_local', mini_batch_src_local)
# 		mini_batch_src_local = list(OrderedDict.fromkeys(mini_batch_src_local.tolist()))
# 		# print('mini_batch_src_local', mini_batch_src_local)
# 		str_+= 'remove duplicated spend time : '+ str(time.time()-ttpp)+'\n\n'
# 		str_+= "&&&&&&&&&&&&&&& after remove duplicate length: "+ str(len(mini_batch_src_local)) +'\n\n'
		
# 		tt2=time.time()
# 		# mini_batch_src_local = torch.tensor(mini_batch_src_local, dtype=torch.long)
# 		mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.
# 		str_+= 'mini_batch_src_global generation: '+str( time.time()-tt2) +'\n'
		

# 		mini_batch_dst_local= list(local_in_edges_tensor)[1]
		
# 		if set(mini_batch_dst_local.tolist()) != set(local_output_nid):
# 			print('local dst not match')
# 		eid_local_list = list(local_in_edges_tensor)[2] # local (𝑈,𝑉,𝐸𝐼𝐷); 
# 		global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
# 	# 	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  bottleneck  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 		# str_+= "\n&&&&&&&&&&&&&&& before remove duplicate length: "+ str(len(mini_batch_src_global))+'\n'
# 		# # ttp=time.time()
# 		# print('mini_batch_src_global', mini_batch_src_global)
# 		# mini_batch_src_global = list(OrderedDict.fromkeys(mini_batch_src_global))
# 		# print('mini_batch_src_global', mini_batch_src_global)
# 		# str_+= "&&&&&&&&&&&&&&& after remove duplicate length: "+ str(len(mini_batch_src_global)) +'\n\n'
# 		ttp1=time.time()

# 		c=OrderedCounter(mini_batch_src_global)
# 		list(map(c.__delitem__, filter(c.__contains__,output_nid)))
# 		r_=list(c.keys())
# 		# 	# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$   bottleneck  $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 		# str_+= 'remove duplicate: '+ str(ttp1-ttp)+'\n'
# 		# str_+= 'r_  generation: '+ str(time.time()-ttp1)+'\n\n'
	
# 		src_nid = torch.tensor(output_nid + r_, dtype=torch.long)
# 		output_nid = torch.tensor(output_nid, dtype=torch.long)

# 		res.append((src_nid, output_nid, global_eid_tensor))
# 	# print(str_)
# 	return res






def generate_blocks_for_one_layer_block(raw_graph, layer_block, global_batched_output_nid_list):

	blocks = []
	check_connection_time = []
	block_generation_time = []

	t1= time.time()
	batches_temp_res_list = check_connections_block(global_batched_output_nid_list, layer_block)
	t2 = time.time()
	check_connection_time.append(t2-t1)

	src_list=[]
	dst_list=[]


	for step, (srcnid, dstnid, current_block_global_eid) in enumerate(batches_temp_res_list):
		t_ = time.time()
		cur_block = generate_one_block(raw_graph, srcnid, dstnid, current_block_global_eid) # block -------
		t__=time.time()
		block_generation_time.append(t__-t_)

		blocks.append(cur_block)
		src_list.append(srcnid)
		dst_list.append(dstnid)

	connection_time = sum(check_connection_time)
	block_gen_time = sum(block_generation_time)
	print()
	print('block_gen_time in "generate_blocks_for_one_layer_block" ', block_gen_time)
	print()
	return blocks, src_list, dst_list, (connection_time, block_gen_time)



def gen_grouped_dst_list(prev_layer_blocks):
	post_dst=[]
	for block in prev_layer_blocks:
		src_nids = block.srcdata['_ID']
		post_dst.append(src_nids)
	return post_dst # return next layer's dst nids(equals prev layer src nids)


def generate_dataloader_block(raw_graph, full_block_dataloader, args):

	if args.num_batch == 1:
		return full_block_dataloader,[1], [0, 0, 0]
	if 'bucketing' in args.selection_method:
		return	generate_dataloader_bucket_block(raw_graph, full_block_dataloader, args)



# def	generate_dataloader_bucket_block(raw_graph, full_block_dataloader, args):
# 	data_loader=[]
# 	dst_nids = []
# 	blocks_list=[]
# 	connect_checking_time_list=[]
# 	block_gen_time_total=0
# 	num_batch = 0
# 	for _,(src_full, dst_full, full_blocks) in enumerate(full_block_dataloader):

# 		# dst_nids = dst_full
		
# 		for layer_id, layer_block in enumerate(reversed(full_blocks)):
			
# 			if layer_id == 0:

# 				bucket_partitioner = Bucket_Partitioner(layer_block, args, full_block_dataloader)
# 				batched_output_nid_list,weights_list,batch_list_generation_time, p_len_list = bucket_partitioner.init_partition()

# 				num_batch=len(batched_output_nid_list)
# 				print('layer ',layer_id )
# 				print(' the number of batches: ', num_batch)
				
# 				# block 0 : (src_0, dst_0); block 1 : (src_1, dst_1);.......
# 				blocks, src_list, dst_list, time_1 = generate_blocks_for_one_layer_block(raw_graph, layer_block,  batched_output_nid_list)

# 				prev_layer_blocks=blocks
# 				blocks_list.append(blocks)
# 				final_dst_list=dst_list
# 				if layer_id==args.num_layers-1:
# 					final_src_list=src_list
# 			else:
# 				tmm=time.time()
# 				grouped_output_nid_list=gen_grouped_dst_list(prev_layer_blocks)
				
# 				num_batch=len(grouped_output_nid_list)
# 				print('layer ',layer_id )
# 				print('num of batch ',num_batch )
# 				blocks, src_list, dst_list, time_1 = generate_blocks_for_one_layer_block(raw_graph, layer_block, grouped_output_nid_list)

# 				if layer_id==args.num_layers-1: # if current block is the final block, the src list will be the final src
# 					final_src_list=src_list
# 				else:
# 					prev_layer_blocks=blocks

# 				blocks_list.append(blocks)

# 			connection_time, block_gen_time = time_1
# 			connect_checking_time_list.append(connection_time)
# 			block_gen_time_total += block_gen_time

# 		batch_blocks_gen_mean_time = block_gen_time_total/num_batch
# 	tt1 = time.time()
# 	for batch_id in range(num_batch):
# 		cur_blocks=[]
# 		for i in range(args.num_layers-1,-1,-1):
# 			cur_blocks.append(blocks_list[i][batch_id])

# 		dst = final_dst_list[batch_id]
# 		src = final_src_list[batch_id]
# 		data_loader.append((src, dst, cur_blocks))
# 	tt2 = time.time()
# 	print('block collection to dataloader spend ', tt2-tt1)
# 	args.num_batch=num_batch
# 	return data_loader, weights_list, [sum(connect_checking_time_list), block_gen_time_total, batch_blocks_gen_mean_time]




def	generate_dataloader_bucket_block(raw_graph, full_block_dataloader, args):
	data_loader=[]

	blocks_list=[]
	connect_checking_time_list=[]
	block_gen_time_total=0
	
	for _,(src_full, dst_full, full_blocks) in enumerate(full_block_dataloader):

		for layer_id, layer_block in enumerate(reversed(full_blocks)):
			
			if layer_id == 0:

				bucket_partitioner = GAT_Bucket_Partitioner(layer_block, args, full_block_dataloader)
				global_batched_output_nid_list,weights_list,batch_list_generation_time, p_len_list = bucket_partitioner.init_partition()
				# global output nids list ([tensor,tensor])
				args.num_batch=len(global_batched_output_nid_list)
				# print('bucketing dataloader: layer ',layer_id )
				# print('bucketing dataloader: the number of batches: ', args.num_batch)
				# print('bucketing dataloader: global_batched_output_nid_list ', global_batched_output_nid_list)
				
				# block 0 : (src_0, dst_0); block 1 : (src_1, dst_1);.......
				blocks, src_list, dst_list, time_1 = generate_blocks_for_one_layer_block(raw_graph, layer_block,  global_batched_output_nid_list)
				# print('bucketing dataloader: global src_list ', src_list)
				prev_layer_blocks=blocks
				blocks_list.append(blocks)
				final_dst_list=dst_list
				if layer_id==args.num_layers-1:
					final_src_list=src_list
			else:
				
				grouped_output_nid_list=gen_grouped_dst_list(prev_layer_blocks)
				
				print('-'*40)
				# print('bucketing dataloader: layer ',layer_id )
				# print('bucketing dataloader: num of batch ',args.num_batch )
				blocks, src_list, _, time_1 = generate_blocks_for_one_layer_block(raw_graph, layer_block, grouped_output_nid_list)
				# print('bucketing dataloader: src_list ', src_list)
				if layer_id==args.num_layers-1: # if current block is the final block, the src list will be the final src
					final_src_list=src_list
				else:
					prev_layer_blocks=blocks
				blocks_list.append(blocks)

			connection_time, block_gen_time = time_1
			connect_checking_time_list.append(connection_time)
			block_gen_time_total += block_gen_time

		batch_blocks_gen_mean_time = block_gen_time_total/args.num_batch
	tt1 = time.time()
	blocks_list = blocks_list[::-1]
	for batch_id in range(args.num_batch):
		cur_blocks = [blocks[batch_id] for blocks in blocks_list]

		dst = final_dst_list[batch_id]
		src = final_src_list[batch_id]
		data_loader.append((src, dst, cur_blocks))
	tt2 = time.time()
	print('block collection to dataloader spend ', tt2-tt1)
	
	return data_loader, weights_list, [sum(connect_checking_time_list), block_gen_time_total, batch_blocks_gen_mean_time]




def swap(split_list):
	index1 = 0
	index2 = len(split_list)-1
	temp = split_list[index1]
	split_list[index1] = split_list[index2]
	split_list[index2] = temp
	return split_list



def split_tensor(tensor, num_parts):
	N = tensor.size(0)
	split_size = N // num_parts
	if N % num_parts != 0:
		split_size += 1

	# Split the tensor into two parts
	split_tensors = torch.split(tensor, split_size)

	# Convert the split tensors into a list
	split_list = list(split_tensors)
 
	split_list = swap(split_list)
	
	weight_list = [len(part) / N for part in split_tensors]
	return split_list, weight_list

def dataloader_gen_range(full_batch_dataloader,g,processed_fan_out, num_batch):
	block_dataloader = []
	blocks_list=[]
	weights_list=[]
	
	for step , (src, dst, full_blocks) in enumerate(full_batch_dataloader):
		
		dst_list, weights_list = split_tensor(dst, num_batch) #######
			
		final_dst_list = dst_list
		pre_dst_list=[]
		for layer , full_block in enumerate(reversed(full_blocks)):
			layer_block_list=[]
			
			layer_graph = dgl.edge_subgraph(g, full_block.edata['_ID'],relabel_nodes=False,store_ids=True)
			src_len = len(full_block.srcdata['_ID'])
			layer_graph.ndata['_ID']=torch.tensor([-1]*len(layer_graph.nodes()))
			layer_graph.ndata['_ID'][:src_len] = full_block.srcdata['_ID']

			if layer == 0:
				print('the output layer ')
				for i,dst_new in enumerate(dst_list) :
					sg1 = dgl.sampling.sample_neighbors_range(layer_graph, dst_new, processed_fan_out[-1])
					block = dgl.to_block(sg1,dst_new, include_dst_in_src= True)
					pre_dst_list.append(block.srcdata[dgl.NID]) 
					layer_block_list.append(block)
			elif layer == 1:
				print('input layer')
				src_list=[]
				for i,dst_new in enumerate(pre_dst_list):
					sg1 = dgl.sampling.sample_neighbors_range(layer_graph, dst_new, processed_fan_out[0])
					block = dgl.to_block(sg1,dst_new, include_dst_in_src= True)
					layer_block_list.append(block)
					src_list.append(block.srcdata[dgl.NID]) 
				final_src_list = src_list

			blocks_list.append(layer_block_list)

		blocks_list = blocks_list[::-1]
		for batch_id in range(num_batch):
			cur_blocks = [blocks[batch_id] for blocks in blocks_list]
			dst = final_dst_list[batch_id]
			src = final_src_list[batch_id]
			block_dataloader.append((src, dst, cur_blocks))
	return block_dataloader, weights_list

def dataloader_gen_bucketing(full_batch_dataloader, g, processed_fan_out, args):
	num_batch = args.num_batch
	block_dataloader = []
	blocks_list=[]
	weights_list=[]
	
	for step , (src, dst, full_blocks) in enumerate(full_batch_dataloader):
		
		# dst_list, weights_list = split_tensor(dst, num_batch) #######
		
		final_src_list = []
		final_dst_list = []
		pre_dst_list=[]
		dst_list=[]
		weights_list=[]

		for layer , full_block in enumerate(reversed(full_blocks)):
			layer_block_list=[]
			layer_graph = dgl.edge_subgraph(g, full_block.edata['_ID'],relabel_nodes=False,store_ids=True)
			src_len = len(full_block.srcdata['_ID'])
			layer_graph.ndata['_ID']=torch.tensor([-1]*len(layer_graph.nodes()))
			layer_graph.ndata['_ID'][:src_len] = full_block.srcdata['_ID']
			if layer == 0:
				print('the output layer ')
				# bucket_partitioner = Bucket_Partitioner(full_block, args, full_batch_dataloader)
				# dst_list, weights_list,_= bucket_partitioner.buckets_partition()
				bucket_partitioner = GAT_Bucket_Partitioner(full_block, args, full_batch_dataloader)
				dst_list,weights_list,batch_list_generation_time, p_len_list = bucket_partitioner.init_partition()
				final_dst_list = dst_list
				src_list = []
				for i,dst_new in enumerate(dst_list) :
					sg1 = dgl.sampling.sample_neighbors_range(layer_graph, dst_new, processed_fan_out[-1])
					block = dgl.to_block(sg1,dst_new, include_dst_in_src= True)
					pre_dst_list.append(block.srcdata[dgl.NID]) 
					layer_block_list.append(block)
					src_list.append(block.srcdata[dgl.NID])
				if layer == args.num_layers-1:
					final_src_list = src_list
			elif layer == 1:
				print('input layer')
				src_list=[]
				for i,dst_new in enumerate(pre_dst_list):
					sg1 = dgl.sampling.sample_neighbors_range(layer_graph, dst_new, processed_fan_out[0])
					block = dgl.to_block(sg1,dst_new, include_dst_in_src= True)
					layer_block_list.append(block)
					src_list.append(block.srcdata[dgl.NID]) 
				pre_dst_list = src_list
				if layer == args.num_layers-1:
					final_src_list = src_list
				

			blocks_list.append(layer_block_list)

		blocks_list = blocks_list[::-1]
		for batch_id in range(num_batch):
			cur_blocks = [blocks[batch_id] for blocks in blocks_list]
			dst = final_dst_list[batch_id]
			src = final_src_list[batch_id]
			block_dataloader.append((src, dst, cur_blocks))
	return block_dataloader, weights_list

def dataloader_gen_bucketing_time(full_batch_dataloader, g, processed_fan_out, args):
	num_batch = args.num_batch
	block_dataloader = []
	blocks_list=[]
	weights_list=[]
	check_connection_time=0
	block_gen_time=0
	backpack_schedule_time = 0
	for step , (src, dst, full_blocks) in enumerate(full_batch_dataloader):
		
		# dst_list, weights_list = split_tensor(dst, num_batch) #######
		
		final_src_list = []
		final_dst_list = []
		pre_dst_list=[]
		dst_list=[]
		weights_list=[]
		
		for layer , (full_block, fanout) in enumerate(zip(reversed(full_blocks),reversed(processed_fan_out))):
			g_gen_start_time= time.time()
			layer_graph = dgl.edge_subgraph(g, full_block.edata['_ID'],relabel_nodes=False,store_ids=True)
			g_gen_end_time= time.time()
			check_connection_time += (g_gen_end_time-g_gen_start_time)

			src_len = len(full_block.srcdata['_ID'])
			layer_graph.ndata['_ID']=torch.tensor([-1]*len(layer_graph.nodes()))
			layer_graph.ndata['_ID'][:src_len] = full_block.srcdata['_ID']

			layer_block_list=[]
			src_list = []
			if layer == 0:
				print('the output layer ')
				schedule_start_time= time.time()
				# bucket_partitioner = Bucket_Partitioner(full_block, args, full_batch_dataloader)
				# dst_list, weights_list,_= bucket_partitioner.buckets_partition()
				bucket_partitioner = GAT_Bucket_Partitioner(full_block, args, full_batch_dataloader)
				dst_list,weights_list,batch_list_generation_time, p_len_list = bucket_partitioner.init_partition()
				schedule_end_time = time.time()
				backpack_schedule_time = schedule_end_time-schedule_start_time
				final_dst_list = dst_list

				for i,dst_new in enumerate(dst_list) :
					check_start_time = time.time()
					sg1 = dgl.sampling.sample_neighbors_range(layer_graph, dst_new, fanout)
					check_end_time = time.time()
					check_connection_time += (check_end_time-check_start_time)

					block = dgl.to_block(sg1,dst_new, include_dst_in_src= True)
					block_gen_end_time = time.time()
					block_gen_time += (block_gen_end_time-check_end_time)

					pre_dst_list.append(block.srcdata[dgl.NID]) 
					layer_block_list.append(block)
					src_list.append(block.srcdata[dgl.NID])

			else: # layer >= 1:
				for i,dst_new in enumerate(pre_dst_list):
					check_start_time = time.time()
					sg1 = dgl.sampling.sample_neighbors_range(layer_graph, dst_new, fanout)
					check_end_time = time.time()
					check_connection_time += (check_end_time-check_start_time)

					block = dgl.to_block(sg1,dst_new, include_dst_in_src= True)
					block_gen_end_time = time.time()
					block_gen_time += (block_gen_end_time-check_end_time)

					layer_block_list.append(block)
					src_list.append(block.srcdata[dgl.NID]) 
				pre_dst_list = src_list

			if layer == args.num_layers-1:
				final_src_list = src_list
			
			blocks_list.append(layer_block_list)

		collect_start_time= time.time()
		blocks_list = blocks_list[::-1]
		for batch_id in range(num_batch):
			cur_blocks = [blocks[batch_id] for blocks in blocks_list]
			dst = final_dst_list[batch_id]
			src = final_src_list[batch_id]
			block_dataloader.append((src, dst, cur_blocks))
		collect_end_time= time.time()
		block_gen_time += (collect_end_time-collect_start_time)
	return block_dataloader, weights_list, backpack_schedule_time, check_connection_time, block_gen_time

