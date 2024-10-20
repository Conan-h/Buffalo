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
from cpu_mem_usage import get_memory
from cpu_mem_usage import get_memory_GB
import pdb

from multiprocessing import Pool, Manager
from functools import partial



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

def tensot_eq(ta,tb):
	equals = torch.eq(ta, tb)
	all_equal = torch.all(equals)

	print(all_equal)
	return all_equal

# def process_node(induced_src, eids_global, shared_dict_nid_2_local, shared_current_layer_block,output_nid):
# 		# induced_src = current_layer_block.srcdata[dgl.NID]
# 		# eids_global = current_layer_block.edata['_ID']
# 		# dict_nid_2_local = dict(zip(induced_src.tolist(), range(len(induced_src)))) # speedup 
# 		print('start process_node -----------')
def process_node(induced_src, eids_global, dict_nid_2_local, current_layer_block, output_nid):
		# induced_src = current_layer_block.srcdata[dgl.NID]
		# eids_global = current_layer_block.edata['_ID']
		# dict_nid_2_local = dict(zip(induced_src.tolist(), range(len(induced_src)))) # speedup 
		print('start process_node -----------')
# 		if torch.is_tensor(output_nid): output_nid = output_nid.tolist()
		

		local_output_nid = list(map(dict_nid_2_local.get, output_nid))
		print('local_output_nid ', local_output_nid)
		local_output_nid = local_output_nid[:1]
		print('current_layer_block.device ', current_layer_block.device)
		print('current_layer_block', current_layer_block)
		local_in_edges_tensor = current_layer_block.in_edges(torch.tensor(local_output_nid), form='all')
		print("in_edges done")
		# return (𝑈,𝑉,𝐸𝐼𝐷)
# 		# get local srcnid and dstnid from subgraph
		mini_batch_src_local= list(local_in_edges_tensor)[0] # local (𝑈,𝑉,𝐸𝐼𝐷);

		print('mini_batch_src_local length', len(mini_batch_src_local))
		mini_batch_src_local = list(OrderedDict.fromkeys(mini_batch_src_local.tolist())) # to remove duplicated items while preserving the original order 
		print('mini_batch_src_local after dict', len(mini_batch_src_local))
		# mini_batch_src_local = torch.tensor(mini_batch_src_local, dtype=torch.long)
		mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.

		# mini_batch_dst_local= list(local_in_edges_tensor)[1]

		# if set(mini_batch_dst_local.tolist()) != set(local_output_nid):
		# 	print('local dst not match')
		eid_local_list = list(local_in_edges_tensor)[2] # local (𝑈,𝑉,𝐸𝐼𝐷); 
		global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
		

		c=OrderedCounter(mini_batch_src_global)
		list(map(c.__delitem__, filter(c.__contains__,output_nid)))
		r_=list(c.keys())

		src_nid = torch.tensor(output_nid + r_, dtype=torch.long)
		output_nid = torch.tensor(output_nid, dtype=torch.long)

		return(src_nid, output_nid, global_eid_tensor)

def process_data(shared_dict, shared_list, induced_src,eids_global, shared_dict_nid_2_local, shared_current_layer_block, items):
    print('process data start')
    # items is a list. Modify the function to work with a list.
    for item in items:
        shared_dict[item] = item * 2
        shared_list.append(item * 3)
    # return the results for this batch of items.
    return (dict(shared_dict), sorted(list(set(shared_list))))	

def check_connections_block(batched_nodes_list, current_layer_block):
	str_=''
	res=[]
	print('check_connections_block*********************************')
	
	# batched_nodes_list=[1,2,3,4,5,6,7,8,9,10,11,12]
	batched_nodes_list_2 = []
	if torch.is_tensor(batched_nodes_list[0]):
		for output_nid in batched_nodes_list:
			output_nid = output_nid.tolist()[:10]
			print('length of output nids ', len(output_nid))
			batched_nodes_list_2.append(output_nid)

	batched_nodes_list = batched_nodes_list_2


	induced_src = current_layer_block.srcdata[dgl.NID]
	induced_dst = current_layer_block.dstdata[dgl.NID]
	eids_global = current_layer_block.edata['_ID']
	
	src_nid_list = induced_src.tolist()
	dst_nid_list = induced_dst.tolist()
	# print('src_nid_list ', src_nid_list[:20])
	print('dst_nid_list ', dst_nid_list[:10])
	# the order of srcdata in current block is not increased as the original graph. For example,
	# src_nid_list  [1049, 432, 741, 554, ... 1683, 1857, 1183, ... 1676]
	# dst_nid_list  [1049, 432, 741, 554, ... 1683]
	
# 	# return
	dict_nid_2_local = dict(zip(src_nid_list, range(len(src_nid_list)))) # speedup 
	print('batched_nodes_list ', batched_nodes_list[2][:10])
	

	# with Manager() as manager:
	# 	induced_src.share_memory_()
	# 	shared_dict = manager.dict()
	# 	shared_list = manager.list()
	# 	items = [[j for j in range(i, i + 2)] for i in range(0, 10, 2)]  # A list of lists.

	# 	process_data_partial = partial(process_data, shared_dict, shared_list)

	# 	with Pool() as p:
	# 		results = p.map(process_data_partial, items)

    #     # Join the results from all processes.
	# 	joined_dict = {}
	# 	joined_list = []
	# 	for res in results:
	# 		joined_dict.update(res[0])  # join dictionaries
	# 		joined_list += res[1]  # join lists

	# 	print('joined_dict', joined_dict)
	# 	print('joined_list', sorted(list(set(joined_list))))
	# # return
 
 
 
	print('-='*40)
	res=[]
	with Manager() as manager:
		induced_src.share_memory_() # tensor to shared 
		eids_global.share_memory_()
		shared_dict_nid_2_local = manager.dict(dict_nid_2_local) # dict to shared 
		current_layer_block_list = [current_layer_block]*len(batched_nodes_list)
		args = [(induced_src, eids_global, shared_dict_nid_2_local, block, nid) for block, nid in zip(current_layer_block_list, batched_nodes_list)]
		with Pool() as p:
			res = p.starmap(process_node, args)
	
	with Manager() as manager:
		# shared_dict = manager.dict()
		# shared_list = manager.list()
		# shared_batched_nodes_list = manager.list(batched_nodes_list)
		induced_src.share_memory_() # tensor to shared 
		eids_global.share_memory_()
		shared_dict_nid_2_local = manager.dict(dict_nid_2_local) # dict to shared 
		# shared_current_layer_block = manager.Namespace()  # create a namespace to contain the shared object
		# shared_current_layer_block.value = current_layer_block
		get_memory_GB('before new_process_node = partial() ')
		current_layer_block_list = [current_layer_block]*len(batched_nodes_list)
		print('current_layer_block_list, ', current_layer_block_list)
		# print('before new_process_node = partial() ------------')
		# items = [[j for j in range(i, i + 2)] for i in range(0, 10, 2)]
		# print('items ', items)
		# print("batched_nodes_list ", len(batched_nodes_list))
		
		# for current_layer_block, output_nid in zip(current_layer_block_list,batched_nodes_list ):
		args = [(induced_src, eids_global, shared_dict_nid_2_local, block, output_nid) for block, output_nid in zip(current_layer_block_list, batched_nodes_list)]
		# new_process_node = partial(process_node, induced_src, eids_global, shared_dict_nid_2_local, current_layer_block)
		# print('after new_process_node = partial() ------------')
		# get_memory_GB('after  new_process_node = partial() ')
		get_memory_GB('before p.starmap ')
		with Pool() as p:
			res = p.starmap(process_node, args)
		print(len(res))
		get_memory_GB('after p.starmap ')

# 	return	
	print('multi res')
	print(res)
	return res

def remove_duplicated_values(my_dict):
	new_dict = {}
	seen_values = set()
	for k, v in my_dict.items():
		if v not in seen_values:
			seen_values.add(v)
			new_dict[k] = v
	return new_dict


def remove_duplicated_keep_order(mini_batch_src_local):
	max_value = max(mini_batch_src_local)
	seen = [False] * (max_value + 1)
	result = []

	for item in mini_batch_src_local:
		if not seen[item]:
			result.append(item)
			seen[item] = True
	return result




# def check_connections_block(batched_nodes_list, current_layer_block):
# 	str_=''
# 	res=[]
# 	print('check_connections_block*********************************')

# 	induced_src = current_layer_block.srcdata[dgl.NID]
# 	induced_dst = current_layer_block.dstdata[dgl.NID]
# 	eids_global = current_layer_block.edata['_ID']
	
# 	src_nid_list = induced_src.tolist()
# 	dst_nid_list = induced_dst.tolist()
# 	# print('src_nid_list ', src_nid_list[:20])
# 	print('dst_nid_list ', dst_nid_list[:10])
# 	# the order of srcdata in current block is not increased as the original graph. For example,
# 	# src_nid_list  [1049, 432, 741, 554, ... 1683, 1857, 1183, ... 1676]
# 	# dst_nid_list  [1049, 432, 741, 554, ... 1683]

# 	# return
# 	dict_nid_2_local = dict(zip(src_nid_list, range(len(src_nid_list)))) # speedup 
	
# 	for step, output_nid in enumerate(batched_nodes_list):
# 		# in current layer subgraph, only has src and dst nodes,
# 		# and src nodes includes dst nodes, src nodes equals dst nodes.
# 		if torch.is_tensor(output_nid): output_nid = output_nid.tolist()
# 		print('step start', step)

# 		local_output_nid = list(map(dict_nid_2_local.get, output_nid))
# 		print('local_output_nid ', local_output_nid[:10])

# 		local_in_edges_tensor = current_layer_block.in_edges(local_output_nid, form='all')
		
# 		# return (𝑈,𝑉,𝐸𝐼𝐷)
# 		# get local srcnid and dstnid from subgraph
# 		mini_batch_src_local= list(local_in_edges_tensor)[0] # local (𝑈,𝑉,𝐸𝐼𝐷);
# 		# mini_batch_src_local = mini_batch_src_local[:100]
# 		# mini_batch_src_local = torch.tensor([11,32,21,54,11,21,65])
# 		print('mini_batch_src_local length', len(mini_batch_src_local))
# 		# print('mini_batch_src_local tensor to list', mini_batch_src_local.tolist())
# 		time11 = time.time()
# 		mini_batch_src_local = list(OrderedDict.fromkeys(mini_batch_src_local.tolist())) # to remove duplicated items while preserving the original order 
# 		time12 = time.time()
# 		print('list(OrderedDict) time ', time12-time11)
# 		# mini_batch_src_local = remove_duplicated_keep_order(mini_batch_src_local.tolist())
# 		# time13 = time.time()
# 		# print('remove_duplicated_keep_order time ', time13-time11)
# 		# time1 = time.time()
# 		# mini_batch_src_local_idx_dict = dict(zip(range(len(mini_batch_src_local)), mini_batch_src_local.tolist()))
# 		# time2 = time.time()
# 		# sorted_dict = dict(sorted(mini_batch_src_local_idx_dict.items(), key=lambda item: item[1]))
# 		# time3 = time.time()
# 		# final_dict = remove_duplicated_values(sorted_dict)
# 		# time4 = time.time()
# 		# sorted_dict_idx = {k: final_dict[k] for k in sorted(final_dict)}
# 		# time5 = time.time()
# 		# mini_batch_src_local = list(sorted_dict_idx.values())
# 		# time6 = time.time()
# 		# print('index dict gen time ', time2-time1)
# 		# print('sorted nids time ', time3-time2)
# 		# print('remove duplicated time ', time4-time3)
# 		# print("sort by local index time ", time5-time4)
# 		# print("get mini_batch_src_local time ", time6-time5)
# 		# print('mini_batch_src_local ', mini_batch_src_local)
# 		print('mini_batch_src_local after dict', len(mini_batch_src_local))
		
# 		# mini_batch_src_local = torch.tensor(mini_batch_src_local, dtype=torch.long)
# 		mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.

# 		mini_batch_dst_local= list(local_in_edges_tensor)[1]

# 		if set(mini_batch_dst_local.tolist()) != set(local_output_nid):
# 			print('local dst not match')
# 		eid_local_list = list(local_in_edges_tensor)[2] # local (𝑈,𝑉,𝐸𝐼𝐷); 
# 		global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
		

# 		c=OrderedCounter(mini_batch_src_global)
# 		list(map(c.__delitem__, filter(c.__contains__,output_nid)))
# 		r_=list(c.keys())

# 		src_nid = torch.tensor(output_nid + r_, dtype=torch.long)
# 		output_nid = torch.tensor(output_nid, dtype=torch.long)

# 		res.append((src_nid, output_nid, global_eid_tensor))
		
# 	return
# 	return res



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
