import torch
import dgl
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../utils/')
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


from my_utils import gen_batch_output_list
from memory_usage import see_memory_usage

from sortedcontainers import SortedList, SortedSet, SortedDict
from multiprocessing import Process, Queue
from collections import Counter, OrderedDict
import copy
from typing import Union, Collection
from my_utils import torch_is_in_1d

import pdb

class OrderedCounter(Counter, OrderedDict):
	'Counter that remembers the order elements are first encountered'

	def __repr__(self):
		return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

	def __reduce__(self):
		return self.__class__, (OrderedDict(self),)
#------------------------------------------------------------------------
# def unique_tensor_item(combined):
# 	uniques, counts = combined.unique(return_counts=True)
# 	return uniques.type(torch.long)




# def get_global_graph_edges_ids_block(raw_graph, block):

# 	edges=block.edges(order='eid', form='all')
# 	edge_src_local = edges[0]
# 	edge_dst_local = edges[1]
# 	# edge_eid_local = edges[2]
# 	induced_src = block.srcdata[dgl.NID]
# 	induced_dst = block.dstdata[dgl.NID]
# 	induced_eid = block.edata[dgl.EID]

# 	raw_src, raw_dst=induced_src[edge_src_local], induced_dst[edge_dst_local]
# 	# raw_src, raw_dst=induced_src[edge_src_local], induced_src[edge_dst_local]

# 	# in homo graph: raw_graph
# 	global_graph_eids_raw = raw_graph.edge_ids(raw_src, raw_dst)
# 	# https://docs.dgl.ai/generated/dgl.DGLGraph.edge_ids.html?highlight=graph%20edge_ids#dgl.DGLGraph.edge_ids
# 	# https://docs.dgl.ai/en/0.4.x/generated/dgl.DGLGraph.edge_ids.html#dgl.DGLGraph.edge_ids

# 	return global_graph_eids_raw, (raw_src, raw_dst)





# def unique_tensor(tensor):
# 	_, unique_indices = torch.unique(tensor, return_inverse=True)
# 	unique_indices, _ = torch.unique(unique_indices, return_inverse=True)

# 	# Use the unique indices to get the unique elements in the original order
# 	unique_elements_tensor = tensor[unique_indices]
# 	return unique_elements_tensor

def remove_duplicated_values(my_dict):
	new_dict = {}
	seen_values = set()
	for k, v in my_dict.items():
		if v not in seen_values:
			seen_values.add(v)
			new_dict[k] = v
	return new_dict


def check_connections_block(batched_nodes_list, current_layer_block):
	str_=''
	res=[]
	print('check connections block*********************************')
	time1 = time.time()
	induced_src = current_layer_block.srcdata[dgl.NID]
	# print(torch.nonzero(induced_src > 90941 ))
	# print(torch.nonzero(induced_src >= 4 ))

	induced_dst = current_layer_block.dstdata[dgl.NID]
	# print(current_layer_block.dstdata[dgl.NID])
	# print(torch.nonzero(induced_dst > 90941 ))
	# print(torch.nonzero(induced_dst >= 4 ))

	eids_global = current_layer_block.edata['_ID']
	time2 = time.time()
	src_nid_list = induced_src.tolist()
	# print('src_nid_list ', src_nid_list)
	# the order of srcdata in current block is not increased as the original graph. For example,
	# src_nid_list  [1049, 432, 741, 554, ... 1683, 1857, 1183, ... 1676]
	# dst_nid_list  [1049, 432, 741, 554, ... 1683]
	time3 = time.time()
	dict_nid_2_local = dict(zip(src_nid_list, range(len(src_nid_list)))) # speedup 
	time4 = time.time()

	g_2_l_time=[]
	in_edges_time = []
	Dict_time =[]
	local_2_global_time = []
	remove_repeate_time = []
	batch_time =[]
	for step, output_nid in enumerate(batched_nodes_list):
		# in current layer subgraph, only has src and dst nodes,
		# and src nodes includes dst nodes, src nodes equals dst nodes.
		if torch.is_tensor(output_nid): output_nid = output_nid.tolist()
		print('step start', step)
		time41 = time.time()
		local_output_nid = list(map(dict_nid_2_local.get, output_nid))
		time42 = time.time()
		g_2_l_time.append(time42-time41)
		# print('local_output_nid ', local_output_nid)
		# local_output_nid_tmp = torch.tensor(local_output_nid)
		# ind_tmp = torch.nonzero(local_output_nid_tmp > 90941 )
		# if len(ind_tmp)>1:
		# 	print((local_output_nid_tmp > 90941).nonzero(as_tuple=True)[0])
		
		local_in_edges_tensor = current_layer_block.in_edges(local_output_nid, form='all')
		time43 = time.time()
		in_edges_time.append(time43-time42)
		# return (𝑈,𝑉,𝐸𝐼𝐷)
		# get local srcnid and dstnid from subgraph
		mini_batch_src_local= list(local_in_edges_tensor)[0] # local (𝑈,𝑉,𝐸𝐼𝐷);
		time44 = time.time()
		# print('mini_batch_src_local', mini_batch_src_local)
		mini_batch_src_local = list(dict.fromkeys(mini_batch_src_local.tolist())) 
		# mini_batch_src_local = list(OrderedDict.fromkeys(mini_batch_src_local.tolist()))
		# mini_batch_src_local_idx_dict = dict(zip(range(len(mini_batch_src_local)), mini_batch_src_local.tolist()))
		# sorted_dict = dict(sorted(mini_batch_src_local_idx_dict.items(), key=lambda item: item[1]))
		# final_dict = remove_duplicated_values(sorted_dict)
		
		# sorted_dict_idx = {k: final_dict[k] for k in sorted(final_dict)}
		# mini_batch_src_local = list(sorted_dict_idx.values())
		time45 = time.time()
		Dict_time.append(time45-time44)
		# mini_batch_src_local = torch.tensor(mini_batch_src_local, dtype=torch.long)
		mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.
		time46 = time.time()

		mini_batch_dst_local= list(local_in_edges_tensor)[1]
		time47 = time.time()
		if set(mini_batch_dst_local.tolist()) != set(local_output_nid):
			print('local dst not match')
		eid_local_list = list(local_in_edges_tensor)[2] # local (𝑈,𝑉,𝐸𝐼𝐷); 
		global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
		
		time48 = time.time()
		local_2_global_time.append(time48-time45)
		c=OrderedCounter(mini_batch_src_global)
		list(map(c.__delitem__, filter(c.__contains__,output_nid)))
		r_=list(c.keys())
		time49 = time.time()

		remove_repeate_time.append(time49-time48)
		src_nid = torch.tensor(output_nid + r_, dtype=torch.long)
		output_nid = torch.tensor(output_nid, dtype=torch.long)

		res.append((src_nid, output_nid, global_eid_tensor))
		print('step end ', step)
		print()
		batch_time.append(time.time()-time41)
	time5 = time.time()
	print('all time spend ', time5-time4)
	print('each batch time ', batch_time)
	print(' global to local time ', sum(g_2_l_time))
	print('in edges time ', sum(in_edges_time))
	print('OrderedDict time ', sum(OrderedDict_time))
	print('local to global time ', sum(local_2_global_time))
	print('remove repeate  dst time ', sum(remove_repeate_time))
	print('dict_nid_2_local = dict(zip(src_nid_list, range(len(src_nid_list)))) ', time4-time3)
	# return
	return res



def generate_one_hop_neighbors(layer_block, batches_nid_list):

	check_connection_time = []

	t1= time.time()
	batches_temp_res_list = check_connections_block(batches_nid_list, layer_block)
	t2 = time.time()
	check_connection_time.append(t2-t1)

	src_list=[]
	dst_list=[]

	for step, (srcnid, dstnid, current_block_global_eid) in enumerate(batches_temp_res_list):

		src_list.append(srcnid)
		dst_list.append(dstnid)

	connection_time = sum(check_connection_time)
	

	return src_list, dst_list, (connection_time)




# def gen_grouped_dst_list(prev_layer_blocks):
# 	post_dst=[]
# 	for block in prev_layer_blocks:
# 		src_nids = block.srcdata['_ID']
# 		post_dst.append(src_nids)
# 	return post_dst # return next layer's dst nids(equals prev layer src nids)




def combine_list(list_of_lists):
    
	import itertools
	combined_list = list(itertools.chain(*list_of_lists))
	return combined_list

def cal_weights_list(batched_output_nid_list, len_dst_full):
    return [len(nids)/len_dst_full for nids in batched_output_nid_list]
    

def	generate_K_hop_neighbors(full_block_dataloader, args, batched_output_nid_list):
    # batched_output_nid_list can be the whole number of output nids
    # or it equals partial of the output nids
	
	dst_nids = []
	
	connect_checking_time_list=[]
	
	final_src_list =[]
	for _,(src_full, dst_full, full_blocks) in enumerate(full_block_dataloader):
    
		dst_nids = dst_full
		num_batch=len(batched_output_nid_list)
		print(' the number of batches: ', num_batch)
		temp = combine_list(batched_output_nid_list )
		print('the ratio of the output nids to be processed: ', len(temp)/len(dst_full))
		weights_list = cal_weights_list(batched_output_nid_list, len(dst_full))
		print('weights list of these nids: ', weights_list)

		for layer_id, layer_block in enumerate(reversed(full_blocks)):
			
			if layer_id == 0:
				src_list, dst_list, time_1 = generate_one_hop_neighbors( layer_block,  batched_output_nid_list)

				final_dst_list=dst_list
				if layer_id==args.num_layers-1:
					final_src_list=src_list
				prev_layer_src = src_list
			else:

				grouped_output_nid_list = prev_layer_src

				num_batch=len(grouped_output_nid_list)
				print('num of batch ',num_batch )
				src_list, dst_list, time_1 = generate_one_hop_neighbors( layer_block, grouped_output_nid_list)

				if layer_id==args.num_layers-1: # if current block is the final block, the src list will be the final src
					final_src_list=src_list
				prev_layer_src = src_list
				
			connection_time = time_1
			connect_checking_time_list.append(connection_time)

	
	return  final_src_list, weights_list, sum(connect_checking_time_list)
