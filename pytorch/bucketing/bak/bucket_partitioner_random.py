import numpy
import dgl
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../utils/')
from numpy.core.numeric import Infinity
import multiprocessing as mp
import torch
import time
from statistics import mean
from my_utils import *
import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import cupy as cp

from collections import Counter
from math import ceil
from cpu_mem_usage import get_memory
from my_utils import torch_is_in_1d

def asnumpy(input):
	return input.cpu().detach().numpy()

def equal (x,y):
	return x == y

def nonzero_1d(input):
	x = torch.nonzero(input, as_tuple=False).squeeze()
	return x if x.dim() == 1 else x.view(-1)

def gather_row(data, row_index):
	return torch.index_select(data, 0, row_index.long())

def zerocopy_from_numpy(np_array):
	return torch.as_tensor(np_array)

def my_sort_1d(val):  # add new function here, to replace torch.sort()
		idx_dict = dict(zip(range(len(val)),val.tolist())) #####
		sorted_res = dict(sorted(idx_dict.items(), key=lambda item: item[1])) ######
		sorted_val = torch.tensor(list(sorted_res.values())).to(val.device)  ######
		idx = torch.tensor(list(sorted_res.keys())).to(val.device) ######
		return sorted_val, idx

class Bucket_Partitioner:  # ----------------------*** split the output layer block ***---------------------
	def __init__(self, layer_block, args):
		# self.balanced_init_ratio=args.balanced_init_ratio
		self.dataset=args.dataset
		self.layer_block=layer_block # local graph with global nodes indices
		self.local=False
		self.output_nids=layer_block.dstdata['_ID'] # tensor type
		self.local_output_nids=[]
		self.local_src_nids=[]
		self.src_nids_tensor= layer_block.srcdata['_ID']
		self.src_nids_list= layer_block.srcdata['_ID'].tolist()
		self.full_src_len=len(layer_block.srcdata['_ID'])
		self.global_batched_seeds_list=[]
		self.local_batched_seeds_list=[]
		self.weights_list=[]
		# self.alpha=args.alpha
		# self.walkterm=args.walkterm
		self.num_batch=args.num_batch
		self.selection_method=args.selection_method
		self.batch_size=0
		self.ideal_partition_size=0

		# self.bit_dict={}
		self.side=0
		self.partition_nodes_list=[]
		self.partition_len_list=[]

		self.time_dict={}
		self.red_before=[]
		self.red_after=[]
		self.args=args
		
		self.in_degrees = self.layer_block.in_degrees()

	



	def _bucketing(self, val):
		# val : local index degrees 
		# sorted_val, idx = torch.sort(val)
		# print('degrees val')
		# print(val)
		sorted_val, idx = val.sort(stable=True)
		# print('sorted_val ', sorted_val)
		# print('idx ', idx)
		# sorted_val, idx = my_sort_1d(val) # keep the nodes in global order

		unique_val = asnumpy(torch.unique(sorted_val))
		bkt_idx = []
		for v in unique_val:
			bool_idx = (sorted_val == v)
			eqidx = torch.nonzero(bool_idx, as_tuple=False).squeeze().view(-1)
			# eqidx = nonzero_1d(equal(sorted_val, v))
			# bkt_idx.append(gather_row(idx, eqidx))
			local_nids = torch.index_select(idx, 0, eqidx.long())
			bkt_idx.append(local_nids)
			
		def bucketor(data):
			bkts = [gather_row(data, idx) for idx in bkt_idx]
			return bkts
		return unique_val, bucketor

	def get_in_degree_bucketing(self):
		
		degs = self.layer_block.in_degrees()
		# print('dst global nid ', self.layer_block.dstdata['_ID'])
		# print('corresponding in degs', degs)
		nodes = self.layer_block.dstnodes() # local dst nid
		
		# degree bucketing
		unique_degs, bucketor = self._bucketing(degs)
		bkt_nodes = []
		for deg, node_bkt in zip(unique_degs, bucketor(nodes)):
			if deg == 0:
				# skip reduce function for zero-degree nodes
				continue
			bkt_nodes.append(node_bkt) # local nid idx
			print('len(bkt) ', len(node_bkt))

		return bkt_nodes  # local nid idx
	


	def get_src(self, seeds):
		in_ids=list(self.layer_block.in_edges(seeds))[0].tolist()
		src= list(set(in_ids+seeds))
		return src



	def gen_batches_seeds_list(self, bkt_dst_nodes_list):

		if "bucketing" in self.selection_method :
			total_len = len(bkt_dst_nodes_list)
			tensor_lengths = [t.numel() for t in bkt_dst_nodes_list]

			if tensor_lengths[-1] > max(tensor_lengths[:-1]):
				fanout_dst_nids = bkt_dst_nodes_list[-1]
				group_nids_list = bkt_dst_nodes_list[:-1]
			else:
				fanout_dst_nids = torch.cat(bkt_dst_nodes_list[int(total_len/2):])
				group_nids_list = bkt_dst_nodes_list[:int(total_len/2)]
			
			if self.args.num_batch <= 1:
				print('no need to split fanout degree, full batch train ')
				self.local_batched_seeds_list = bkt_dst_nodes_list
				return
			
			if self.args.num_batch > 1:
				fanout_batch_size = ceil(len(fanout_dst_nids)/(self.args.num_batch-1))
				# args.batch_size = batch_size
		

			if 'random' in self.selection_method:
				# print('before  shuffle ', fanout_dst_nids)
				indices = torch.randperm(len(fanout_dst_nids))
				map_output_list = fanout_dst_nids.view(-1)[indices].view(fanout_dst_nids.size())
				# print('after shuffle ', map_output_list)
				batches_nid_list = [map_output_list[i:i + fanout_batch_size] for i in range(0, len(map_output_list), fanout_batch_size)]
			
    
			if 'range' in self.selection_method:   
				batches_nid_list = [map_output_list[i:i + fanout_batch_size] for i in range(0, len(map_output_list), fanout_batch_size)]
			
			
			if len(group_nids_list) == 1 :
				batches_nid_list.insert(0, group_nids_list[0])
			else:
				group_tensor = torch.cat(group_nids_list)
				group_tensor_increase, _ = torch.sort(group_tensor)
				batches_nid_list.insert(0, group_tensor_increase)
			
			length = len(self.output_nids)
			self.weights_list = [len(batch_nids)/length  for batch_nids in batches_nid_list]
			# print('batches_nid_list ', batches_nid_list)
			# print('weights_list ', self.weights_list)
			

			self.local_batched_seeds_list = batches_nid_list
		return


	def get_src_len(self,seeds):
		in_nids=self.layer_block.in_edges(seeds)[0]
		src =torch.unique(in_nids)
		return src.size()



	def get_partition_src_len_list(self):
		partition_src_len_list=[]
		for seeds_nids in self.local_batched_seeds_list:
			partition_src_len_list.append(self.get_src_len(seeds_nids))

		self.partition_src_len_list=partition_src_len_list
		self.partition_len_list=partition_src_len_list
		return 


	def buckets_partition(self):
		
		bkt_dst_nodes_list = self.get_in_degree_bucketing()
		t2 = time.time()

		self.gen_batches_seeds_list(bkt_dst_nodes_list)
		# print('total k batches seeds list generation spend ', time.time()-t2 )

		# self.get_partition_src_len_list()

		return 



	def global_to_local(self):
		
		sub_in_nids = self.src_nids_list
		# print('src global')
		# print(sub_in_nids)#----------------
		# global_nid_2_local = {sub_in_nids[i]: i for i in range(0, len(sub_in_nids))}
		global_nid_2_local = dict(zip(sub_in_nids,range(len(sub_in_nids))))
		self.local_output_nids = list(map(global_nid_2_local.get, self.output_nids.tolist()))
		# print('dst local')
		# print(self.local_output_nids)#----------------
		# self.local_src_nids = list(map(global_nid_2_local.get, self.src_nids_list))
		
		self.local=True
		
		return

	
	def local_to_global(self):
		# print('-'*40)
		# print('src global ', self.src_nids_tensor)

		for local_seed_nids in self.local_batched_seeds_list:
			# print('local nid ', local_seed_nids)
			
			# if 'range' in self.selection_method:
			# 	self.global_batched_seeds_list.append(gather_row(self.src_nids_tensor, local_seed_nids))
			if 'random' in self.selection_method:
				local_all = torch.tensor(list(range(len(self.src_nids_tensor))))
				
				eqidx = nonzero_1d(torch_is_in_1d(local_all, local_seed_nids))
				# print('eqidx ', eqidx)
				
				after_sort = gather_row(self.src_nids_tensor, eqidx)
					
				# print('after sort based on full batch order ', after_sort)
				self.global_batched_seeds_list.append(after_sort)
				
			print()

		self.local=False
		# print('-'*40)
		return


	def init_partition(self):
		ts = time.time()
		
		self.global_to_local() # global to local           
		
		t2=time.time()
		# Then, the graph_parition is run in block to graph local nids,it has no relationship with raw graph
		self.buckets_partition()  # generate  self.local_batched_seeds_list 

		# after that, we transfer the nids of batched output nodes from local to global.
		self.local_to_global() # local to global         self.global_batched_seeds_list
		t_total=time.time()-ts

		return self.global_batched_seeds_list, self.weights_list, t_total, self.partition_len_list