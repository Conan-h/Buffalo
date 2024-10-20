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


from reg_partitioner import Graph_Partitioner

from my_utils import gen_batch_output_list
from memory_usage import see_memory_usage

from sortedcontainers import SortedList, SortedSet, SortedDict
from multiprocessing import Process, Queue
from collections import Counter, OrderedDict
import copy
from typing import Union, Collection
from my_utils import torch_is_in_1d

import pdb




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
		
		for layer , (full_block, fanout) in enumerate(zip(reversed(full_blocks),reversed(processed_fan_out))):
			
			layer_graph = dgl.edge_subgraph(g, full_block.edata['_ID'],relabel_nodes=False,store_ids=True)
			
			src_len = len(full_block.srcdata['_ID'])
			layer_graph.ndata['_ID']=torch.tensor([-1]*len(layer_graph.nodes()))
			layer_graph.ndata['_ID'][:src_len] = full_block.srcdata['_ID']

			layer_block_list=[]
			src_list = []
			if layer == 0:
				print('the output layer ')
				
				bucket_partitioner = Graph_Partitioner(full_block, args, full_batch_dataloader)
				dst_list,weights_list,batch_list_generation_time, p_len_list = bucket_partitioner.init_partition()
				
				final_dst_list = dst_list

				for i,dst_new in enumerate(dst_list) :
					
					sg1 = dgl.sampling.sample_neighbors_range(layer_graph, dst_new, fanout)
					block = dgl.to_block(sg1,dst_new, include_dst_in_src= True)
					
					pre_dst_list.append(block.srcdata[dgl.NID]) 
					layer_block_list.append(block)
					src_list.append(block.srcdata[dgl.NID])

			else: # layer >= 1:
				for i,dst_new in enumerate(pre_dst_list):
					
					sg1 = dgl.sampling.sample_neighbors_range(layer_graph, dst_new, fanout)
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
				
				bucket_partitioner = Graph_Partitioner(full_block, args)
				dst_list,weights_list,batch_list_generation_time, p_len_list = bucket_partitioner.init_graph_partition()
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

