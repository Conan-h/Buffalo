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
	dst_local_nid_list=list(OrderedCounter(edge_dst_list).keys())

	new_block = dgl.to_block(_graph, dst_nodes=torch.tensor(dst_local_nid_list, dtype=torch.long))
	new_block.srcdata[dgl.NID] = global_srcnid
	new_block.dstdata[dgl.NID] = global_dstnid
	new_block.edata['_ID']=_graph.edata['_ID']

	return new_block







def check_connections_block(batched_nodes_list, current_layer_block):
	str_=''
	res=[]
	# print('check_connections_block*********************************')

	induced_src = current_layer_block.srcdata[dgl.NID]
	
	eids_global = current_layer_block.edata['_ID']

	src_nid_list = induced_src.tolist()
	# print('src_nid_list ', src_nid_list)
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
		mini_batch_src_local = list(OrderedDict.fromkeys(mini_batch_src_local.tolist()))
		
		# mini_batch_src_local = torch.tensor(mini_batch_src_local, dtype=torch.long)
		mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.
	

		mini_batch_dst_local= list(local_in_edges_tensor)[1]
		
		if set(mini_batch_dst_local.tolist()) != set(local_output_nid):
			print('local dst not match')
		eid_local_list = list(local_in_edges_tensor)[2] # local (𝑈,𝑉,𝐸𝐼𝐷); 
		global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
		

		c=OrderedCounter(mini_batch_src_global)
		list(map(c.__delitem__, filter(c.__contains__,output_nid)))
		r_=list(c.keys())
		
		src_nid = torch.tensor(output_nid + r_, dtype=torch.long)
		output_nid = torch.tensor(output_nid, dtype=torch.long)

		res.append((src_nid, output_nid, global_eid_tensor))

	return res




# def check_connections_block(batched_nodes_list, current_layer_block):
# 	str_=''
# 	res=[]
# 	# print('check_connections_block*********************************')


# 	induced_src = current_layer_block.srcdata[dgl.NID]
	
# 	eids_global = current_layer_block.edata['_ID']


	
# 	# src_nid_list = induced_src.tolist()
# 	# print('src_nid_list ', induced_src)
# 	# the order of srcdata in current block is not increased as the original graph. For example,
# 	# src_nid_list  [1049, 432, 741, 554, ... 1683, 1857, 1183, ... 1676]
# 	# dst_nid_list  [1049, 432, 741, 554, ... 1683]
	
# 	induced_src = induced_src.long()
# 	# print('batched_nodes_list ', batched_nodes_list)
# 	for step, output_nid in enumerate(batched_nodes_list):
		
# 		# print('global output_nid', output_nid)
# 		output_nid = output_nid.long()
		
		
# 		local_output_nid_long = torch_is_in_1d(induced_src, output_nid).long() # might change the order of the local nid 
# 		local_output_nid = torch.nonzero(local_output_nid_long, as_tuple=True)[0]
# 		# print('local_output_nid ', local_output_nid)
# 		output_nid = induced_src[local_output_nid] # to make the output nid match with the changed order
# 		local_in_edges_tensor = current_layer_block.in_edges(local_output_nid, form='all')
# 		# print('local_in_edges_tensor', local_in_edges_tensor) # is based on the changed local nid order
		
# 		# return (𝑈,𝑉,𝐸𝐼𝐷)
# 		# get local srcnid and dstnid from subgraph
# 		mini_batch_src_local= list(local_in_edges_tensor)[0] # local (𝑈,𝑉,𝐸𝐼𝐷);
# 		# print('mini_batch_src_local ', mini_batch_src_local)

# 		# print('mini_batch_src_local', mini_batch_src_local)
# 		mini_batch_src_local = list(OrderedDict.fromkeys(mini_batch_src_local.tolist()))

# 		# mini_batch_src_local = torch.tensor(mini_batch_src_local, dtype=torch.long)
# 		mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.


# 		mini_batch_dst_local= list(local_in_edges_tensor)[1]

# 		if set(mini_batch_dst_local.tolist()) != set(local_output_nid.tolist()):
# 			print('local dst not match')
# 		eid_local_list = list(local_in_edges_tensor)[2] # local (𝑈,𝑉,𝐸𝐼𝐷);
# 		global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
	
# 		c=OrderedCounter(mini_batch_src_global)
# 		list(map(c.__delitem__, filter(c.__contains__,output_nid.tolist())))
# 		r_=list(c.keys())
# 		r_ = torch.tensor(r_, dtype=torch.long)

# 		src_nid = torch.cat((output_nid , r_))
		
# 		res.append((src_nid, output_nid, global_eid_tensor))

# 	return res



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









# def gen_batched_output_list(dst_nids, args ):
# 	batch_size=0
# 	if args.num_batch != 0 :
# 		batch_size = ceil(len(dst_nids)/args.num_batch)
# 		args.batch_size = batch_size
# 	# print('number of batches is ', args.num_batch)
# 	# print('batch size is ', batch_size)
# 	partition_method = args.selection_method
# 	batches_nid_list=[]
# 	weights_list=[]
# 	if partition_method=='range':
# 		indices = [i for i in range(len(dst_nids))]
# 		map_output_list = list(numpy.array(dst_nids)[indices])
# 		batches_nid_list = [map_output_list[i:i + batch_size] for i in range(0, len(map_output_list), batch_size)]
# 		length = len(dst_nids)
# 		weights_list = [len(batch_nids)/length  for batch_nids in batches_nid_list]
# 	if partition_method=='random':
# 		indices = torch.randperm(len(dst_nids))
# 		map_output_list = dst_nids.view(-1)[indices].view(dst_nids.size())
# 		# map_output_list = list(numpy.array(dst_nids)[indices])
# 		batches_nid_list = [map_output_list[i:i + batch_size] for i in range(0, len(map_output_list), batch_size)]
# 		length = len(dst_nids)
# 		weights_list = [len(batch_nids)/length  for batch_nids in batches_nid_list]

# 	return batches_nid_list, weights_list


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




def	generate_dataloader_bucket_block(raw_graph, full_block_dataloader, args):
	data_loader=[]
	dst_nids = []
	blocks_list=[]
	connect_checking_time_list=[]
	block_gen_time_total=0
	for _,(src_full, dst_full, full_blocks) in enumerate(full_block_dataloader):

		dst_nids = dst_full
		

		for layer_id, layer_block in enumerate(reversed(full_blocks)):
			# print('layer_block.edata[dgl.NID]')
			# print(layer_block.edata['_ID'])
			# block_eidx_global, block_edges_nids_global = get_global_graph_edges_ids_block(raw_graph, layer_block)
			# layer_block.edata['_ID'] = block_eidx_global  # this only do in the first time
			
			if layer_id == 0:

				bucket_partitioner = Bucket_Partitioner(layer_block, args)
				batched_output_nid_list,weights_list,batch_list_generation_time, p_len_list = bucket_partitioner.init_partition()

				num_batch=len(batched_output_nid_list)
				print(' the number of batches: ', num_batch)
				# print('the batched output global nids ', batched_output_nid_list)


				#----------------------------------------------------------
				# select_time=time.time()-t1
				# print(str(args.selection_method)+' selection method  spend '+ str(select_time))
				# block 0 : (src_0, dst_0); block 1 : (src_1, dst_1);.......
				blocks, src_list, dst_list, time_1 = generate_blocks_for_one_layer_block(raw_graph, layer_block,  batched_output_nid_list)

				prev_layer_blocks=blocks
				blocks_list.append(blocks)
				final_dst_list=dst_list
				if layer_id==args.num_layers-1:
					final_src_list=src_list
			else:
				tmm=time.time()
				grouped_output_nid_list=gen_grouped_dst_list(prev_layer_blocks)
				# print('gen group dst list time: ', time.time()-tmm)
				num_batch=len(grouped_output_nid_list)
				# print('num of batch ',num_batch )
				blocks, src_list, dst_list, time_1 = generate_blocks_for_one_layer_block(raw_graph, layer_block, grouped_output_nid_list)

				if layer_id==args.num_layers-1: # if current block is the final block, the src list will be the final src
					final_src_list=src_list
				else:
					prev_layer_blocks=blocks

				blocks_list.append(blocks)

			connection_time, block_gen_time = time_1
			connect_checking_time_list.append(connection_time)
			block_gen_time_total += block_gen_time
		# connect_checking_time_res=sum(connect_checking_time_list)
		batch_blocks_gen_mean_time = block_gen_time_total/num_batch

	for batch_id in range(num_batch):
		cur_blocks=[]
		for i in range(args.num_layers-1,-1,-1):
			cur_blocks.append(blocks_list[i][batch_id])

		dst = final_dst_list[batch_id]
		src = final_src_list[batch_id]
		data_loader.append((src, dst, cur_blocks))
	args.num_batch=num_batch
	return data_loader, weights_list, [sum(connect_checking_time_list), block_gen_time_total, batch_blocks_gen_mean_time]

