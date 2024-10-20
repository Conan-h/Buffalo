import torch
import dgl
import numpy as np
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
import multiprocessing as mp
from functools import partial

from bucket_partitioner import Bucket_Partitioner
# from draw_graph import draw_dataloader_blocks_pyvis

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
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing/global_2_local')
import find_indices
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing/gen_tails')
import gen_tails
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing/src_gen')
import src_gen
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing/gen_src_tail')
import gen_src_tails

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






def unique_tensor(tensor):
	_, unique_indices = torch.unique(tensor, return_inverse=True)
	unique_indices, _ = torch.unique(unique_indices, return_inverse=True)

	# Use the unique indices to get the unique elements in the original order
	unique_elements_tensor = tensor[unique_indices]
	return unique_elements_tensor

def check_connections_block(batched_nodes_list, current_layer_block):
    
	print('check_connections_block*********************************')

	induced_src = current_layer_block.srcdata[dgl.NID]
	eids_global = current_layer_block.edata['_ID']

	src_nid_list = induced_src.tolist()

	print('')
	# global_batched_nids_list = [nid.tolist() for nid in batched_nodes_list]
	timess = time.time()
	global_batched_nids_list = [nid.tolist() for nid in batched_nodes_list]
	output_nid_list = find_indices.find_indices(src_nid_list, global_batched_nids_list)
	# dict_nid_2_local = dict(zip(src_nid_list, range(len(src_nid_list)))) # speedup
	# output_nid_list =[]
	# for step, output_nid in enumerate(batched_nodes_list):
	# 	# in current layer subgraph, only has src and dst nodes,
	# 	# and src nodes includes dst nodes, src nodes equals dst nodes.
	# 	if torch.is_tensor(output_nid): output_nid = output_nid.tolist()
	# 	local_output_nid = list(map(dict_nid_2_local.get, output_nid))
	# 	output_nid_list.append(local_output_nid)
	
	print('the find indices time spent ', time.time()-timess)
	
		# in current layer subgraph, only has src and dst nodes,
		# and src nodes includes dst nodes, src nodes equals dst nodes.
	print()
	
	# parallel-------------------------------------------------end
	time1= time.time()
	# dgl graph.in_edges() sequential
	local_in_edges_tensor_list=[]
	for step, local_output_nid in enumerate(output_nid_list):
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
		mini_batch_src_local = list(dict.fromkeys(mini_batch_src_local))
		# mini_batch_src_local = remove_duplicates.remove_duplicates(mini_batch_src_local)
		mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.

		eid_local_list = local_in_edges_tensor[2] # local (𝑈,𝑉,𝐸𝐼𝐷); 
		global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
		eids_list.append(global_eid_tensor)
		src_long_list.append(mini_batch_src_global)
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
# 	print('check_connections_block*********************************')

# 	induced_src = current_layer_block.srcdata[dgl.NID]
# 	induced_dst = current_layer_block.dstdata[dgl.NID]
# 	eids_global = current_layer_block.edata['_ID']

# 	src_nid_list = induced_src.tolist()
# 	print('')
# 	# global_batched_nids_list = [nid.tolist() for nid in batched_nodes_list]
# 	timess = time.time()
# 	global_batched_nids_list = [nid.tolist() for nid in batched_nodes_list]
# 	output_nid_list = find_indices.find_indices(src_nid_list, global_batched_nids_list)

# 	print('the find indices time spent ', time.time()-timess)

# 	print()
	

# 	time1= time.time()
# 	local_in_edges_tensor_list=[]
# 	for step, local_output_nid in enumerate(output_nid_list):
# 		local_in_edges_tensor = current_layer_block.in_edges(local_output_nid, form='all')
# 		local_in_edges_res = [id.tolist() for id in local_in_edges_tensor]
# 		local_in_edges_tensor_list.append(local_in_edges_res)
# 	time2=time.time()
# 	print('in edges time spent ', time2-time1)

# 	# time31=time.time()
	
# 	# induced_src_dict = dict(zip(range(len(src_nid_list)), src_nid_list))
# 	# eids_global_dict = dict(zip(range(len(eids_global)), eids_global.tolist()))
# 	# time310 =time.time()
# 	# print('src_gen start--=-=-=-=-=')
# 	# eids_list, src_long_list = src_gen.src_gen(local_in_edges_tensor_list, global_batched_nids_list, induced_src_dict, eids_global_dict)
# 	# print('src_gen end--=-=-=-=-=')
# 	# time311 =time.time()
# 	# eids_list = list({k: eids_list[k] for k in sorted(eids_list)}.values())
# 	# src_long_list = list({k: src_long_list[k] for k in sorted(src_long_list)}.values())
# 	# time32 = time.time()
# 	# print('local to global src and eids time spent ', time32-time31)
# 	# print('prepare time ', time310-time31)
# 	# print('src gen time ', time311-time310)
# 	# print('post sort  time ', time32-time311)	
	
# 	# time33 = time.time()
# 	# tails_list = gen_tails.gen_tails(src_long_list, global_batched_nids_list)
# 	# time34 = time.time()
# 	# print('time gen  tails ', time34-time33)

# 	time33 = time.time()
# 	local_in_edges_tensor_list= np.array(local_in_edges_tensor_list, dtype=object)
# 	global_batched_nids_list = np.array(global_batched_nids_list, dtype=object)
# 	induced_src_list = np.array(induced_src.tolist())
# 	eids_global_list = np.array(eids_global.tolist())
# 	eids_dict_list, tail_list= gen_src_tails.gen_src_tails(local_in_edges_tensor_list, global_batched_nids_list, induced_src_list, eids_global_list)
# 	time34 = time.time()
# 	print('time gen src  tails ', time34-time33)
# 	eids_dict_list = list({k: eids_dict_list[k] for k in sorted(eids_dict_list)}.values())
# 	tail_list = list({k: tail_list[k] for k in sorted(tail_list)}.values())
# 	# print('global_batched_nids_list ', global_batched_nids_list)
# 	# print('tail_list ', tail_list)

# 	# print('eids_dict_list',eids_dict_list)

# 	res =[]
# 	for global_output_nid, r_,eid  in zip(global_batched_nids_list,tail_list,eids_dict_list):
# 		src_nid = torch.tensor(global_output_nid + r_, dtype=torch.long)
# 		output_nid = torch.tensor(global_output_nid, dtype=torch.long)

# 		res.append((src_nid, output_nid, eid))
# 	# parallel-------------------------------------------------end
# 	print("res  length", len(res))
# 	return res

# def check_connections_block(batched_nodes_list, current_layer_block):
	
# 	print('check_connections_block*********************************')

# 	induced_src = current_layer_block.srcdata[dgl.NID]
# 	induced_dst = current_layer_block.dstdata[dgl.NID]
# 	eids_global = current_layer_block.edata['_ID']

# 	src_nid_list = induced_src.tolist()
# 	print('')
# 	# global_batched_nids_list = [nid.tolist() for nid in batched_nodes_list]
# 	timess = time.time()
# 	global_batched_nids_list = [nid.tolist() for nid in batched_nodes_list]
# 	output_nid_list = find_indices.find_indices(src_nid_list, global_batched_nids_list)
# 	# dict_nid_2_local = dict(zip(src_nid_list, range(len(src_nid_list)))) # speedup 
# 	# output_nid_list =[]
# 	# for step, output_nid in enumerate(batched_nodes_list):
# 	# 	# in current layer subgraph, only has src and dst nodes,
# 	# 	# and src nodes includes dst nodes, src nodes equals dst nodes.
# 	# 	if torch.is_tensor(output_nid): output_nid = output_nid.tolist()
# 	# 	local_output_nid = list(map(dict_nid_2_local.get, output_nid))
# 	# 	output_nid_list.append(local_output_nid)
# 	print('the find indices time spent ', time.time()-timess)

# 	# the order of srcdata in current block is not increased as the original graph. For example,
# 	# src_nid_list  [1049, 432, 741, 554, ... 1683, 1857, 1183, ... 1676]
# 	# dst_nid_list  [1049, 432, 741, 554, ... 1683]
# 	print()
	
# 	# parallel-------------------------------------------------end
# 	time1= time.time()
# 	# dgl graph.in_edges() sequential
# 	local_in_edges_tensor_list=[]
# 	for step, local_output_nid in enumerate(output_nid_list):
# 		local_in_edges_tensor = current_layer_block.in_edges(local_output_nid, form='all')
# 		local_in_edges_res = [id.tolist() for id in local_in_edges_tensor]
# 		local_in_edges_tensor_list.append(local_in_edges_res)
# 	time2=time.time()
# 	print('in edges time spent ', time2-time1)

# 	time31=time.time()
	
# 	# eids_list = {}
# 	# src_long_list = {}
	
# 	# for i, local_in_edges_tensor, global_output_nid in enumerate(zip(local_in_edges_tensor_list, global_batched_nids_list)):
# 	# 	mini_batch_src_local= local_in_edges_tensor[0] # local (𝑈,𝑉,𝐸𝐼𝐷);
# 	# 	mini_batch_src_local = list(dict.fromkeys(mini_batch_src_local))
# 	# 	mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.
# 	# 	# mini_batch_dst_local= local_in_edges_tensor[1]
# 	# 	# if len(set(mini_batch_dst_local)) != len(set(global_output_nid)):
# 	# 	# 	print('local dst length vs global dst length are not match')
# 	# 	eid_local_list = local_in_edges_tensor[2] # local (𝑈,𝑉,𝐸𝐼𝐷); 
# 	# 	global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
# 	# 	eids_list[i] = global_eid_tensor
# 	# 	src_long_list[i] = mini_batch_src_global
# 	# induced_src_dict = dict(zip(range(len(src_nid_list)), src_nid_list))
# 	# eids_global_dict = dict(zip(range(len(eids_global)), eids_global.tolist()))
# 	# time310 =time.time()
# 	# print('src_gen start--=-=-=-=-=')
# 	# eids_list, src_long_list = src_gen.src_gen(local_in_edges_tensor_list, global_batched_nids_list, induced_src_dict, eids_global_dict)
# 	# print('src_gen end--=-=-=-=-=')
# 	# time311 =time.time()

# 	time310 =time.time()
# 	print('src_gen start--=-=-=-=-=')
# 	eids_list, src_long_list = src_gen.src_gen(local_in_edges_tensor_list, global_batched_nids_list, induced_src.tolist(), eids_global.tolist())
# 	print('src_gen end--=-=-=-=-=')
# 	time311 =time.time()
# 	eids_list = list({k: eids_list[k] for k in sorted(eids_list)}.values())
# 	src_long_list = list({k: src_long_list[k] for k in sorted(src_long_list)}.values())
# 	time32 = time.time()
# 	print('local to global src and eids time spent ', time32-time31)
# 	# print('prepare time ', time310-time31)
# 	print('src gen time ', time311-time310)
# 	print('post sort  time ', time32-time311)	
# 	# original_dict = {'b': 1, 'a': 2, 'c': 3}
	

# 	time33 = time.time()
# 	tails_list = gen_tails.gen_tails(src_long_list, global_batched_nids_list)
# 	time34 = time.time()
# 	print('time gen tails ', time34-time33)
# 	res =[]
# 	for global_output_nid, r_,eid  in zip(global_batched_nids_list,tails_list,eids_list):
# 		src_nid = torch.tensor(global_output_nid + r_, dtype=torch.long)
# 		output_nid = torch.tensor(global_output_nid, dtype=torch.long)

# 		res.append((src_nid, output_nid, eid))
# 	# parallel-------------------------------------------------end
# 	print("res  length", len(res))
# 	return res
def check_connections_block_bak2(batched_nodes_list, current_layer_block):
	print('check_connections_block*********************************')

	induced_src = current_layer_block.srcdata[dgl.NID]
	induced_dst = current_layer_block.dstdata[dgl.NID]
	eids_global = current_layer_block.edata['_ID']

	src_nid_list = induced_src.tolist()
	print('')
	# global_batched_nids_list = [nid.tolist() for nid in batched_nodes_list]
	timess = time.time()
	global_batched_nids_list = [nid.tolist() for nid in batched_nodes_list]
	output_nid_list = find_indices.find_indices(src_nid_list, global_batched_nids_list)
	# dict_nid_2_local = dict(zip(src_nid_list, range(len(src_nid_list)))) # speedup 
	# output_nid_list =[]
	# for step, output_nid in enumerate(batched_nodes_list):
	# 	# in current layer subgraph, only has src and dst nodes,
	# 	# and src nodes includes dst nodes, src nodes equals dst nodes.
	# 	if torch.is_tensor(output_nid): output_nid = output_nid.tolist()
	# 	local_output_nid = list(map(dict_nid_2_local.get, output_nid))
	# 	output_nid_list.append(local_output_nid)
	print('the find indices time spent ', time.time()-timess)

	# the order of srcdata in current block is not increased as the original graph. For example,
	# src_nid_list  [1049, 432, 741, 554, ... 1683, 1857, 1183, ... 1676]
	# dst_nid_list  [1049, 432, 741, 554, ... 1683]
	print()
	
	# parallel-------------------------------------------------end
	time1= time.time()
	# dgl graph.in_edges() sequential
	local_in_edges_tensor_list=[]
	for step, local_output_nid in enumerate(output_nid_list):
		local_in_edges_tensor = current_layer_block.in_edges(local_output_nid, form='all')
		local_in_edges_res = [id.tolist() for id in local_in_edges_tensor]
		local_in_edges_tensor_list.append(local_in_edges_res)
	time2=time.time()
	print('in edges time spent ', time2-time1)

	time31=time.time()
	
	# eids_list = {}
	# src_long_list = {}
	
	# for i, local_in_edges_tensor, global_output_nid in enumerate(zip(local_in_edges_tensor_list, global_batched_nids_list)):
	# 	mini_batch_src_local= local_in_edges_tensor[0] # local (𝑈,𝑉,𝐸𝐼𝐷);
	# 	mini_batch_src_local = list(dict.fromkeys(mini_batch_src_local))
	# 	mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.
	# 	# mini_batch_dst_local= local_in_edges_tensor[1]
	# 	# if len(set(mini_batch_dst_local)) != len(set(global_output_nid)):
	# 	# 	print('local dst length vs global dst length are not match')
	# 	eid_local_list = local_in_edges_tensor[2] # local (𝑈,𝑉,𝐸𝐼𝐷); 
	# 	global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
	# 	eids_list[i] = global_eid_tensor
	# 	src_long_list[i] = mini_batch_src_global
	# induced_src_dict = dict(zip(range(len(src_nid_list)), src_nid_list))
	# eids_global_dict = dict(zip(range(len(eids_global)), eids_global.tolist()))
	# time310 =time.time()
	# print('src_gen start--=-=-=-=-=')
	# eids_list, src_long_list = src_gen.src_gen(local_in_edges_tensor_list, global_batched_nids_list, induced_src_dict, eids_global_dict)
	# print('src_gen end--=-=-=-=-=')
	# time311 =time.time()

	time310 =time.time()
	print('src_gen start--=-=-=-=-=')
	eids_list, src_long_list = src_gen.src_gen(local_in_edges_tensor_list, global_batched_nids_list, induced_src.tolist(), eids_global.tolist())
	print('src_gen end--=-=-=-=-=')
	time311 =time.time()
	eids_list = list({k: eids_list[k] for k in sorted(eids_list)}.values())
	src_long_list = list({k: src_long_list[k] for k in sorted(src_long_list)}.values())
	time32 = time.time()
	print('local to global src and eids time spent ', time32-time31)
	# print('prepare time ', time310-time31)
	print('src gen time ', time311-time310)
	print('post sort  time ', time32-time311)	
	# original_dict = {'b': 1, 'a': 2, 'c': 3}
	

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


def check_connections_block_bak(batched_nodes_list, current_layer_block):
	str_=''
	res=[]
	print('check_connections_block*********************************')

	induced_src = current_layer_block.srcdata[dgl.NID]
	induced_dst = current_layer_block.dstdata[dgl.NID]
	eids_global = current_layer_block.edata['_ID']

	src_nid_list = induced_src.tolist()

	# the order of srcdata in current block is not increased as the original graph. For example,
	# src_nid_list  [1049, 432, 741, 554, ... 1683, 1857, 1183, ... 1676]
	# dst_nid_list  [1049, 432, 741, 554, ... 1683]
	
	dict_nid_2_local = dict(zip(src_nid_list, range(len(src_nid_list)))) # speedup 

	for step, output_nid in enumerate(batched_nodes_list):
		# in current layer subgraph, only has src and dst nodes,
		# and src nodes includes dst nodes, src nodes equals dst nodes.
		if torch.is_tensor(output_nid): output_nid = output_nid.tolist()
		local_output_nid = list(map(dict_nid_2_local.get, output_nid))
		
		local_in_edges_tensor = current_layer_block.in_edges(local_output_nid, form='all')
		
		# return (𝑈,𝑉,𝐸𝐼𝐷)
		# get local srcnid and dstnid from subgraph
		mini_batch_src_local= list(local_in_edges_tensor)[0] # local (𝑈,𝑉,𝐸𝐼𝐷);
		
		# print('mini_batch_src_local', mini_batch_src_local)
		mini_batch_src_local = list(dict.fromkeys(mini_batch_src_local.tolist()))

		# mini_batch_src_local = torch.tensor(mini_batch_src_local, dtype=torch.long)
		mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.
	
		mini_batch_dst_local= list(local_in_edges_tensor)[1]
		
		if set(mini_batch_dst_local.tolist()) != set(local_output_nid):
			print('local dst not match')
		eid_local_list = list(local_in_edges_tensor)[2] # local (𝑈,𝑉,𝐸𝐼𝐷); 
		global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
		
		r_ = remove_values.remove_values(mini_batch_src_global, output_nid)
		src_nid = torch.tensor(output_nid + r_, dtype=torch.long)
		output_nid = torch.tensor(output_nid, dtype=torch.long)

		res.append((src_nid, output_nid, global_eid_tensor))
	print("res  length", len(res))
	return res

def generate_one_block(raw_graph, global_srcnid, global_dstnid, global_eids):
	'''

	Parameters
	----------
	G    global graph                     DGLGraph
	eids  cur_batch_subgraph_global eid   tensor int64

	Returns
	-------

	'''
	# raw_graph = raw_graph_.clone()
	_graph = dgl.edge_subgraph(raw_graph, global_eids, store_ids=True)
	edge_dst_list = _graph.edges(order='eid')[1].tolist()
	# dst_local_nid_list=list(OrderedCounter(edge_dst_list).keys())
	dst_local_nid_list = remove_duplicates.remove_duplicates(edge_dst_list)

	new_block = dgl.to_block(_graph, dst_nodes=torch.tensor(dst_local_nid_list, dtype=torch.long))
	new_block.srcdata[dgl.NID] = global_srcnid
	new_block.dstdata[dgl.NID] = global_dstnid
	new_block.edata['_ID']=_graph.edata['_ID']

	return new_block
	
    
# def generate_one_block(raw_graph_, global_srcnid, global_dstnid, global_eids):
# 	raw_graph = raw_graph_.clone()
# 	_graph = dgl.edge_subgraph(raw_graph, global_eids, store_ids=True)
# 	edge_dst_list = _graph.edges(order='eid')[1].tolist()
# 	dst_local_nid_list = remove_duplicates.remove_duplicates(edge_dst_list)
# 	new_block = dgl.to_block(_graph, dst_nodes=torch.tensor(dst_local_nid_list, dtype=torch.long))
# 	new_block.srcdata[dgl.NID] = global_srcnid
# 	new_block.dstdata[dgl.NID] = global_dstnid
# 	new_block.edata['_ID'] = _graph.edata['_ID']
# 	return new_block

# def generate(args, raw_graph):
#     step, (srcnid, dstnid, current_block_global_eid) = args
#     t_ = time.time()
#     cur_block = generate_one_block(raw_graph, srcnid, dstnid, current_block_global_eid)  # Graph is cloned and processed here
#     t__ = time.time()
#     return t__ - t_, cur_block, srcnid, dstnid

# def generate_blocks_for_one_layer_block(raw_graph, layer_block, batches_nid_list):
#     check_connection_time = []
#     t1 = time.time()
#     batches_temp_res_list = check_connections_block(batches_nid_list, layer_block)
#     t2 = time.time()
#     check_connection_time.append(t2-t1)

#     with mp.Pool(processes=4) as pool:
#         results = pool.map(partial(generate, raw_graph=raw_graph), enumerate(batches_temp_res_list))

#     block_generation_time, blocks, src_list, dst_list = zip(*results)

#     return blocks, src_list, dst_list, check_connection_time, block_generation_time









def generate_blocks_for_one_layer_block(raw_graph, layer_block, batches_nid_list):

	blocks = []
	check_connection_time = []
	block_generation_time = []

	t1= time.time()
	batches_temp_res_list = check_connections_block(batches_nid_list, layer_block)
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

				bucket_partitioner = Bucket_Partitioner(layer_block, args, full_block_dataloader)
				batched_output_nid_list,weights_list,batch_list_generation_time, p_len_list = bucket_partitioner.init_partition()

				args.num_batch=len(batched_output_nid_list)
				print('layer ',layer_id )
				print(' the number of batches: ', args.num_batch)
				
				# block 0 : (src_0, dst_0); block 1 : (src_1, dst_1);.......
				blocks, src_list, dst_list, time_1 = generate_blocks_for_one_layer_block(raw_graph, layer_block,  batched_output_nid_list)

				prev_layer_blocks=blocks
				blocks_list.append(blocks)
				final_dst_list=dst_list
				if layer_id==args.num_layers-1:
					final_src_list=src_list
			else:
				
				grouped_output_nid_list=gen_grouped_dst_list(prev_layer_blocks)
				
				
				print('layer ',layer_id )
				print('num of batch ',args.num_batch )
				blocks, src_list, _, time_1 = generate_blocks_for_one_layer_block(raw_graph, layer_block, grouped_output_nid_list)

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
