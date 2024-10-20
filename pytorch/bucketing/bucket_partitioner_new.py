from grouping_float import grouping_fanout_products, grouping_fanout_arxiv, grouping_cora, grouping_pre
from gen_K_hop_neighbors import generate_K_hop_neighbors
import pdb
from my_utils import torch_is_in_1d
from cpu_mem_usage import get_memory
from math import ceil
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import networkx as nx
from my_utils import *
from statistics import mean
import time
import torch
import multiprocessing as mp
from numpy.core.numeric import Infinity
import numpy
import dgl
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../utils/')
# sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing')
# sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/utils')
# import cupy as cp


def read_est_mem(filename):
	# Open the file in read mode

	with open(filename, 'r') as file:
		# Load the dictionary from the file
		data_str = file.read()
	# Evaluate the string as a python dictionary
		dict_data = eval(data_str)

	# Convert the dictionary values to a list
	list_data = list(dict_data.values())
	# Now list_data contains the dictionary values
	return list_data


def print_(list_):
	for ll in list_:
		print('length ', len(ll))
		print(ll)
		print()


def get_sum(list_idx, mem):
	# Extract the values from mem using a list comprehension
	values = [mem[idx] for idx in list_idx]
	# Compute the sum using the built-in function
	return sum(values)

# def get_sum(list_idx, mem):
#     res=0
#     # print(mem)
#     print(list_idx)
#     for idx in list_idx:
#         # print(idx)
#         temp = mem[idx]
#         res += temp
#     return res


def asnumpy(input):
	return input.cpu().detach().numpy()


def equal(x, y):
	return x == y


def nonzero_1d(input):
	x = torch.nonzero(input, as_tuple=False).squeeze()
	return x if x.dim() == 1 else x.view(-1)


def gather_row(data, row_index):
	return torch.index_select(data, 0, row_index.long())


def zerocopy_from_numpy(np_array):
	return torch.as_tensor(np_array)


def my_sort_1d(val):  # add new function here, to replace torch.sort()
	idx_dict = dict(zip(range(len(val)), val.tolist()))
	sorted_res = dict(sorted(idx_dict.items(), key=lambda item: item[1]))
	sorted_val = torch.tensor(list(sorted_res.values())).to(val.device)
	idx = torch.tensor(list(sorted_res.keys())).to(val.device)
	return sorted_val, idx


def split_list(input_list, k):
	avg = len(input_list) // k
	remainder = len(input_list) % k
	return [input_list[i * avg + min(i, remainder):(i + 1) * avg + min(i + 1, remainder)] for i in range(k)]


class Bucket_Partitioner:  # ----------------------*** split the output layer block ***---------------------
	def __init__(self, layer_block, args, full_batch_dataloader):
		# self.balanced_init_ratio=args.balanced_init_ratio
		self.memory_constraint = args.mem_constraint
		self.model = args.model
		self.dataset = args.dataset
		self.layer_block = layer_block  # local graph with global nodes indices
		self.local = False
		self.output_nids = layer_block.dstdata['_ID']  # tensor type
		self.local_output_nids = []
		self.local_src_nids = []
		self.src_nids_tensor = layer_block.srcdata['_ID']
		self.src_nids_list = layer_block.srcdata['_ID'].tolist()
		self.full_src_len = len(layer_block.srcdata['_ID'])
		self.global_batched_seeds_list = []
		self.local_batched_seeds_list = []
		self.weights_list = []
		self.hidden = args.num_hidden
		# self.alpha=args.alpha
		# self.walkterm=args.walkterm
		self.num_batch = args.num_batch
		self.selection_method = args.selection_method
		self.batch_size = 0
		self.ideal_partition_size = 0

		# self.bit_dict={}
		self.side = 0
		self.partition_nodes_list = []
		self.partition_len_list = []

		self.time_dict = {}
		self.red_before = []
		self.red_after = []
		self.args = args
		self.full_batch_dataloader = full_batch_dataloader

		self.in_degrees = self.layer_block.in_degrees()
		self.K = args.num_batch

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

	def get_weights(self, b_nids_list, length):
		weights_list = [len(batch_nids)/length for batch_nids in b_nids_list]
		return weights_list


	def get_in_degree_bucketing(self):
		print('self.num_batch (get_in_degree_bucketing)', self.num_batch)
		degs = self.layer_block.in_degrees()
		org_src = self.layer_block.srcdata['_ID']
		print('get_in_degree_bucketing src global nid ', self.layer_block.srcdata['_ID'])
		print('get_in_degree_bucketing dst global nid ', self.layer_block.dstdata['_ID'])
		print('get_in_degree_bucketing corresponding in degs', degs)
		# local dst nid (e.g. in full batch layer block)
		nodes = self.layer_block.dstnodes()
		total_output_nids = 0
		# degree bucketing
		unique_degs, bucketor = self._bucketing(degs)
		bkt_nodes = []
		for deg, node_bkt in zip(unique_degs, bucketor(nodes)):
			if deg == 0:
				# skip reduce function for zero-degree nodes
				continue
			global_node_bkt = org_src[node_bkt]  # local nid idx
			bkt_nodes.append(global_node_bkt)  # global nid idx
			# bkt_nodes.append(node_bkt) # local nid idx
			print('len(bkt) ', len(node_bkt))
			print('global bkt nids ', global_node_bkt)
			total_output_nids += len(node_bkt)
		print('total indegree bucketing result , ', total_output_nids)
		return bkt_nodes  # global nid idx

	def get_src(self, seeds):
		in_ids = list(self.layer_block.in_edges(seeds))[0].tolist()
		src = list(set(in_ids+seeds))
		return src

	def get_nids_by_degree_bucket_ID(self, bucket_lists, bkt_dst_nodes_list):
		print('bkt_dst_nodes_list length', len(bkt_dst_nodes_list))
		res = []
		print('bucket_lists ', bucket_lists)
		for cur_bucket_degrees in bucket_lists:
			print('cur_bucket_degrees', cur_bucket_degrees)
			temp = []
			for degree in cur_bucket_degrees:
				print('degree ', degree)
				print('bkt_dst_nodes_list[b] ', len(bkt_dst_nodes_list[degree]))
				temp.append(bkt_dst_nodes_list[degree])
			flattened_list = [
				element for sublist in temp for element in sublist]
			res.append(torch.tensor(flattened_list, dtype=torch.long))

		return res
	
	def _arxiv_1_layer_backpack_bucketing(self, bkt_dst_nodes_list):

		if '10_backpack_' in self.selection_method:
			time_split_start = time.time()
			print('len(bkt_dst_nodes_list) ', len(bkt_dst_nodes_list))
			fanout_dst_nids = bkt_dst_nodes_list[-1]
			fanout = len(bkt_dst_nodes_list)
			if self.args.num_batch >= 1:
				fanout_batch_size = ceil(len(fanout_dst_nids)/(self.K))
			indices = torch.arange(0, len(fanout_dst_nids)).long()
			map_output_list = fanout_dst_nids.view(
				-1)[indices].view(fanout_dst_nids.size())

			split_batches_nid_list = [map_output_list[i:i + fanout_batch_size]
										for i in range(0, len(map_output_list), fanout_batch_size)]
			adjust = 1000
			estimated_mem = [0.20663738250732422, 0.2413644790649414, 0.24947547912597656, 0.24201679229736328,
								0.2423257827758789, 0.22321128845214844, 0.2119159698486328, 0.20371055603027344, 0.19252681732177734]
			print('sum(estimated_mem)')
			print(sum(estimated_mem))
			print(len(estimated_mem))

			time_backpack_start = time.time()
			capacity_imp = self.memory_constraint-4
			print('self.K ', self.K)
			Groups_mem_list, G_BUCKET_ID_list = grouping_fanout_arxiv(
				adjust, estimated_mem, capacity_imp, fanout, self.K)
			print("G_BUCKET_ID_list", G_BUCKET_ID_list)
			print("Groups_mem_list ", Groups_mem_list)

			print("G_BUCKET_ID_list length", len(G_BUCKET_ID_list))
			g_bucket_nids_list = self.get_nids_by_degree_bucket_ID(
				G_BUCKET_ID_list, bkt_dst_nodes_list)

			time_backpack_end = time.time()
			print('backpack scheduling spend ',
					time_backpack_end-time_backpack_start)

			time_batch_gen_start = time.time()
			for j in range(len(g_bucket_nids_list)):
				tensor_group = torch.tensor(
					g_bucket_nids_list[j], dtype=torch.long)
				current_group_mem = get_sum(
					G_BUCKET_ID_list[j], estimated_mem)
				print("current group_mem ", current_group_mem)

				split_batches_nid_list[j] = torch.cat(
					(split_batches_nid_list[j], tensor_group))

			time_batch_gen_end = time.time()
			print('batches output list generation spend ',
					time_batch_gen_end-time_batch_gen_start)
			length = len(self.output_nids)
			self.weights_list = [
				len(batch_nids)/length for batch_nids in split_batches_nid_list]
			print('self.weights_list ', self.weights_list)

			self.local_batched_seeds_list = split_batches_nid_list

			return
		# else: #'25_backpack_' not in self.selection_method:
		# 	print('memory_constraint: ', self.memory_constraint)

		return
    
    
	def _arxiv_2_layer_backpack_bucketing(self, bkt_dst_nodes_list):
		fanout = len(bkt_dst_nodes_list)
		fanout_dst_nids = bkt_dst_nodes_list[-1]
		adjust = 1000
		global_split_batches_nid_list = list(torch.chunk(fanout_dst_nids, self.K))
		if '25_backpack_' in self.selection_method:
			if self.hidden == 512:
				# hidden = 512, 2-layers----------------------------------------
				# 1, fanout-1 sum = 30.68 GB
				# fanout degree bucket estimated(GB):  13.912
				estimated_mem = [1.198210731940998, 1.3826510845020086, 1.4798242805861725, 1.5131057876252738, 1.5469755167800183, 1.5329394980816933, 1.5077958816240011, 1.4923785888291725, 1.460850731217092, 1.4427281586605571, 1.3665918666594188, 1.3728889498582622,
									1.295971195170712, 1.2609649579666078, 1.2364575763492274, 1.1646228757256543, 1.1685366658374694, 1.1278081158988085, 1.1136677277641134, 1.0931365828546005, 1.0541397156769547, 0.9513897832433253, 0.9615408947013018, 0.9617162983703614]
				capacity_imp = self.memory_constraint-10.3  # nb 4 capacity = 7.8
				capacity_imp = self.memory_constraint-7.8  # nb 3 capacity = 10.3
				capacity_imp = self.memory_constraint-2.75  # nb 3 capacity = 15.35
			elif self.hidden == 256:  # 2-layers =========================================

				# new estimation start
				estimated_mem = [0.9289283752441406, 1.0921419664986132, 1.174592981690282, 1.2067483835407944, 1.2375349585000666, 1.2332499812783622, 1.219712377230607, 1.2118137788993009, 1.184205675680791, 1.17600772351804, 1.1165122126748235, 1.1218068344161014,
									1.0613692097616039, 1.0356467682433141, 1.0133642192630459, 0.9551042604703931, 0.9602214569977391, 0.928765299108964, 0.91671499803582, 0.9005825296175082, 0.8687226300373521, 0.7873555455142333, 0.8009047141131459, 0.7953034350585939]
				if self.model == 'SAGE':
					if self.num_batch == 2:
						capacity_imp = 12.7
					elif self.num_batch == 4:
						capacity_imp = 6.3  # nb 4
					elif self.num_batch == 8:
						capacity_imp = 3.2  # nb 8
					elif self.num_batch == 16:
						capacity_imp = 1.95  # nb 16
					elif self.num_batch == 20:
						capacity_imp = 1.75  # nb 20
					elif self.num_batch == 24:
						capacity_imp = 1.75  # nb 24
					elif self.num_batch == 32:
						capacity_imp = 1.75  # nb 32
				elif self.model == 'GAT':
					if self.num_batch == 16:
						capacity_imp = 1.935  # nb 16
					elif self.num_batch == 18:
						capacity_imp = 1.85  # nb 18
					elif self.num_batch == 20:
						capacity_imp = 1.77  # nb 20
					elif self.num_batch == 22:
						capacity_imp = 1.67  # nb 22
					elif self.num_batch == 24:
						capacity_imp = 1.57  # nb 24
					elif self.num_batch == 26:
						capacity_imp = 1.25  # nb 26
					elif self.num_batch == 32:
						capacity_imp = 1.25  # nb 32
					elif self.num_batch == 40:
						capacity_imp = 1.25  # nb 40
			elif self.hidden == 128:  # 2-layers =========================================
				# estimated_mem = [0.42060097957744036, 0.47311523799417204, 0.5099940648098965, 0.5260206879067616, 0.5394331193958898, 0.5433008773297489, 0.5375267645234505, 0.535240380926753, 0.5229354704483863, 0.5202699559667836, 0.494472521700576, 0.49753113977861085, 0.47079622652897474, 0.46104649837089745, 0.451559590774548, 0.4241338345118196, 0.4282073690309857, 0.4134191343331639, 0.4082059832911322, 0.4015348967842896, 0.38770616330547525, 0.35303355384683827, 0.35899451293571416, 0.35517937969207763]
				# sum of above 11.03(GB),  degree 25 mem estimated =  7.83 GB
				# capacity_imp = self.memory_constraint-10
				estimated_mem = [0.8136749267578125, 0.9482579950540875, 1.0206992380889157, 1.0538952284680039, 1.0818744809050456, 1.084757384115296, 1.0731104791436792, 1.069628515049197, 1.0480042225122452, 1.0424392025159752, 0.9898839248621336, 0.9953236727234502,
									0.9418539936965127, 0.919397501761373, 0.900892277961891, 0.8499588936003988, 0.8558874481191368, 0.8270597629788556, 0.8186162241878332, 0.8020723312370839, 0.7767946092237346, 0.7062005828906696, 0.7157837956733637, 0.7102104830932617]
				# sum of above 22.04627717461996 ,  degree 25 mem estimated =  7.83 GB
				if self.model == 'SAGE':
					if self.num_batch == 2:
						capacity_imp = 11.2
					elif self.num_batch == 3:
						capacity_imp = 7.4
					elif self.num_batch == 4:
						capacity_imp = 5.51
					elif self.num_batch == 5:
						capacity_imp = 4.45
					elif self.num_batch == 6:
						capacity_imp = 3.7
					elif self.num_batch == 7:
						capacity_imp = 3.3
					elif self.num_batch == 8:
						capacity_imp = 2.8
					elif self.num_batch == 10:
						capacity_imp = 2.45
					elif self.num_batch == 12:
						capacity_imp = 2.2
					elif self.num_batch == 14:
						capacity_imp = 1.845
					elif self.num_batch == 16:
						capacity_imp = 1.73
					elif self.num_batch == 18:
						capacity_imp = 1.65
				# fanout= 10,25, GAT model
				if self.model == 'GAT':
					estimated_mem = [0.8136749267578125, 0.9482579950540875, 1.0206992380889157, 1.0538952284680039, 1.0818744809050456, 1.084757384115296, 1.0731104791436792, 1.069628515049197, 1.0480042225122452, 1.0424392025159752, 0.9898839248621336, 0.9953236727234502,
										0.9418539936965127, 0.919397501761373, 0.900892277961891, 0.8499588936003988, 0.8558874481191368, 0.8270597629788556, 0.8186162241878332, 0.8020723312370839, 0.7767946092237346, 0.7062005828906696, 0.7157837956733637, 0.7102104830932617]
					# sum of above 22.04627717461996 ,  degree 25 mem estimated =  7.83 GB
					if self.num_batch == 3:
						capacity_imp = 7.3
					elif self.num_batch == 4:
						capacity_imp = 5.51
					elif self.num_batch == 5:
						capacity_imp = 4.45
					elif self.num_batch == 6:
						capacity_imp = 3.7
					elif self.num_batch == 7:
						capacity_imp = 3.3
					elif self.num_batch == 8:
						capacity_imp = 2.8
					elif self.num_batch == 10:
						capacity_imp = 2.45
					elif self.num_batch == 14:
						capacity_imp = 1.845
					elif self.num_batch == 16:
						capacity_imp = 1.73
					elif self.num_batch == 18:
						capacity_imp = 1.65

			elif self.hidden == 1024:  # 2-layers =========================================
				estimated_mem = [1.6722423737809742, 1.9625855842993116, 2.091162774761497, 2.1225820131933104, 2.164434594135045, 2.1270061419533235, 2.0853221102618043, 2.056020043092527, 2.003826841183247, 1.9838461929404219, 1.8742896464201482, 1.8757872602731587,
									1.7739355233489074, 1.7230819912868847, 1.6827613338597718, 1.5790509384345839, 1.5840132793947195, 1.5308580965452556, 1.511252469938908, 1.481189862667495, 1.429230435772136, 1.2836578043908993, 1.2942055601305504, 1.2939860083007813]
				# sum of above = 42.186328880365664
				# mem size of fanout degree bucket by formula (GB):  22.041518211364746
				capacity_imp = self.memory_constraint-11.8  # nb=7, capacity = 6.2GB
				capacity_imp = self.memory_constraint-12.4  # nb=8, capacity = 5.6GB
				capacity_imp = self.memory_constraint-12.9  # nb=9, capacity = 5.1GB
				capacity_imp = self.memory_constraint-13.1  # nb=10, capacity = 4.9GB
				capacity_imp = self.memory_constraint-13.7  # nb=11, capacity = 4.3GB
				capacity_imp = self.memory_constraint-14.3  # nb=12, capacity = 3.7GB
				capacity_imp = self.memory_constraint-14.7  # nb=16, capacity = 3.3GB
				capacity_imp = self.memory_constraint-15.8  # nb=32, capacity = 2.2GB
		elif '30_backpack_' in self.selection_method:
			# hidden = 256, 3-layers  ++++++++++++++++++++++
			# estimated_mem = [11.09133707869788, 7.949351660231331, 4.247930594170108, 3.767815723282392, 3.521911913357682, 3.3171698192934964, 3.1136675746243436, 2.9493490757618086, 2.783152518733855, 2.690315310282632, 2.4780320925231085, 2.4736405822131586, 2.424331795810457, 2.3423791134065306, 2.2508307132173453, 2.0888736283425056, 2.1393341709313427, 2.0144231711761864, 2.028492122175591, 1.8952586057017728, 1.9272310948984677, 1.7648288977543771, 1.8025470147872276, 1.7435488087790354, 1.624197941655698, 1.6781451757272114, 1.585553364875989, 1.5579015641599088, 1.508019266507371]
			# capacity_imp = self.memory_constraint - 7.2  # nb 8 capacity = 10.8 OOM
			# capacity_imp = self.memory_constraint - 8.5 # nb 9 capacity = 9.5
			# capacity_imp = self.memory_constraint-9.5 # nb 10 capacity = 8.5
			# capacity_imp = self.memory_constraint-10 # nb 11 capacity = 8
			# capacity_imp = self.memory_constraint-11 # nb 12 capacity = 7
			# capacity_imp = self.memory_constraint-10 # nb 11 capacity = 8
			# hidden = 128 , 3-layers -----------------------
			estimated_mem = [0.4535364771051893, 0.36959890964869774, 0.3526514615214476, 0.3520529200944431, 0.3502430736172785, 0.36746639428808736, 0.37070838534849293, 0.3745401657375872, 0.3829395052158705, 0.3897006004864236, 0.39194492014904353, 0.4089296170047096, 0.41463856040323055, 0.4250790632909854,
								0.4276191680715249, 0.433518633755968, 0.4386219996684242, 0.4403945473679008, 0.4539703137199875, 0.43765308297576344, 0.4485483741064351, 0.46742121625797345, 0.48099263700595957, 0.46918894709559855, 0.45248479346855597, 0.47846493770442255, 0.47077801986316126, 0.47288452210262105, 0.47387407820263094]
			# self.memory_constraint = 18.2
			capacity_imp = self.memory_constraint-10  # nb 4 capacity = 8.2
			capacity_imp = self.memory_constraint-12  # nb 3 capcity = 6.2
			capacity_imp = self.memory_constraint-11.3  # nb 2 capcity = 6.9
		elif '40_backpack_' in self.selection_method:
			if self.hidden == 256:  # 4-layers  ++++++++++++++++++++++
				estimated_mem = [11.09133707869788, 7.949351660231331, 4.247930594170108, 3.767815723282392, 3.521911913357682, 3.3171698192934964, 3.1136675746243436, 2.9493490757618086, 2.783152518733855, 2.690315310282632, 2.4780320925231085, 2.4736405822131586, 2.424331795810457, 2.3423791134065306,
									2.2508307132173453, 2.0888736283425056, 2.1393341709313427, 2.0144231711761864, 2.028492122175591, 1.8952586057017728, 1.9272310948984677, 1.7648288977543771, 1.8025470147872276, 1.7435488087790354, 1.624197941655698, 1.6781451757272114, 1.585553364875989, 1.5579015641599088, 1.508019266507371]
				capacity_imp = self.memory_constraint - 7.2  # nb 8 capcity = 10.8 OOM

			elif self.hidden == 128:  # 4-layers -----------------------
				# estimated_mem = [1.4202840539485668, 0.9531810572873257, 0.8501257401792143, 0.829695121515657, 0.8040267620971004, 0.8552266709134924, 0.87383665358779, 0.8877523187041695, 0.9202977174929393, 0.9447504233849566, 0.9844265929374212, 1.0289029187865644, 1.0804431370233614, 1.1354009080821856, 1.1611661075239867, 1.224069148253064, 1.240404931513701, 1.2636364692160853, 1.3306185914974074, 1.2931502650592053, 1.3629551133524638, 1.5343035032847725, 1.5850221538103173, 1.5260448500222705, 1.5164359292334326, 1.7007932353880122, 1.6631663204321356, 1.6820257959307852, 1.7593463747414162, 1.9869836014229492, 1.7687368601520908, 1.8374942623615267, 1.8503978645093515, 1.9892680420034075, 1.9863616244737083, 1.9066152871052424, 1.9559371199471611, 2.00413179896149, 1.8693271285505297]
				# self.memory_constraint = 18.2
				# estimated_mem = [1.065213040461425, 0.7148857929654943, 0.6375943051344107, 0.6222713411367428, 0.6030200715728252, 0.6414200031851193, 0.6553774901908423, 0.6658142390281272, 0.6902232881197046, 0.7085628175387175, 0.7383199447030658, 0.7716771890899233, 0.810332352767521, 0.8515506810616392, 0.8708745806429903, 0.918051861189798, 0.9303036986352756, 0.9477273519120639, 0.9979639436230555, 0.969862698794404, 1.0222163350143478, 1.1507276274635794, 1.1887666153577383, 1.1445336375167028, 1.1373269469250744, 1.275594926541009, 1.247374740324102, 1.261519346948089, 1.3195097810560623, 1.4902377010672119, 1.3265526451140681, 1.378120696771145, 1.3877983983820137, 1.4919510315025555, 1.489771218355281, 1.4299614653289316, 1.4669528399603706, 1.503098849221118, 1.4019953464128971]
				estimated_mem = [7.541331259903894, 5.061138357277836, 4.513941983252465, 4.405460822207029, 4.269168648303187, 4.541026571222084, 4.639840638519238, 4.713729125862847, 4.886536553059854, 5.016373929477647, 5.22704385630489, 5.463201338689721, 5.736866214283334, 6.0286773880470035, 6.165483756764532, 6.499482203113614, 6.586220875293987, 6.709574172828771, 7.0652314592782695,
									6.866284593234719, 7.236929805411312, 8.14674426522888, 8.416046834391066, 8.102893008967808, 8.051872190619996, 9.03076054188325, 8.830971612913995, 8.931110420871427, 9.341662166768582, 10.55035540578557, 9.39152315125004, 9.756606702804566, 9.825121404474434, 10.562485178779154, 10.547052873311724, 10.123620993479163, 10.385506831577846, 10.64140778209641, 9.925630771064759]
				# estimated_mem = [12.568885433173158, 8.435230595463059, 7.523236638754108, 7.342434703678381, 7.115281080505313, 7.568377618703472, 7.733067730865398, 7.8562152097714115, 8.144227588433091, 8.360623215796076, 8.711739760508152, 9.10533556448287, 9.561443690472224, 10.047795646745005, 10.27580626127422, 10.83247033852269, 10.977034792156646, 11.182623621381286, 11.775385765463781, 11.443807655391197, 12.06154967568552, 13.5779071087148, 14.026744723985109, 13.504821681613013, 13.41978698436666, 15.051267569805416, 14.718286021523324, 14.88518403478571, 15.569436944614303, 17.583925676309285, 15.652538585416732, 16.261011171340943, 16.37520234079072, 17.604141964631925, 17.578421455519543, 16.872701655798604, 17.309178052629743, 17.735679636827346, 16.5427179517746]
				capacity_imp = self.memory_constraint-7.5  # nb 50 capcity = 10.7
			
		print('self.K ', self.K)
		Groups_mem_list, G_BUCKET_ID_list = grouping_fanout_arxiv(
			adjust, estimated_mem, capacity_imp, fanout, self.K)
		print("G_BUCKET_ID_list", G_BUCKET_ID_list)
		print("Groups_mem_list ", Groups_mem_list)
		if len(G_BUCKET_ID_list) > self.K:
			print('------------errror-----------------')

		print("G_BUCKET_ID_list length", len(G_BUCKET_ID_list))
		print('bkt_dst_nodes_list length ', len(bkt_dst_nodes_list))
		g_bucket_nids_list = self.get_nids_by_degree_bucket_ID(
			G_BUCKET_ID_list, bkt_dst_nodes_list)

		print('len(g_bucket_nids_list) ', len(g_bucket_nids_list))
		
		for j in range(len(g_bucket_nids_list)):
			tensor_group = torch.tensor(
				g_bucket_nids_list[j], dtype=torch.long)
			current_group_mem = get_sum(
				G_BUCKET_ID_list[j], estimated_mem)
			print("current group_mem ", current_group_mem)

			global_split_batches_nid_list[j] = torch.cat(
				(global_split_batches_nid_list[j], tensor_group))  # splitting + grouping
			# local_split_batches_nid_list[j]= tensor_group      # grouping Only


		
		length = len(self.output_nids)
		weights_list = self.get_weights(bkt_dst_nodes_list, len(self.output_nids))
	

		return global_split_batches_nid_list, weights_list

	def fanout_bucketing(self, bkt_dst_nodes_list, output_nids):
		length = len(output_nids)
		
		# Directly calculate weights and assign the list
		weights_list = self.get_weights(bkt_dst_nodes_list, length)
		global_batched_seeds_list = bkt_dst_nodes_list

		return global_batched_seeds_list, weights_list

	def _50_backpack_bucketing(self, bkt_dst_nodes_list): # not correct yet, need to modify
		fanout_dst_nids = bkt_dst_nodes_list[-1]
		fanout = len(bkt_dst_nodes_list)
		if self.args.num_batch >= 1:
			fanout_batch_size = ceil(
						len(fanout_dst_nids)/(self.args.num_batch))
			indices = torch.arange(0, len(fanout_dst_nids)).long()
			map_output_list = fanout_dst_nids.view(
					-1)[indices].view(fanout_dst_nids.size())
			batches_nid_list = [map_output_list[i:i + fanout_batch_size]
									for i in range(0, len(map_output_list), fanout_batch_size)]
			src_list, weights_list, time_collection = generate_K_hop_neighbors(
					self.full_batch_dataloader, self.args, batches_nid_list)
			redundant_ratio = []
			for (i, input_nodes) in enumerate(src_list):
				print(len(input_nodes) /
							len(batches_nid_list[i])/fanout/0.411)
				redundant_ratio.append(
						len(input_nodes)/len(batches_nid_list[i])/fanout/0.411)
				print('the split redundant ratio ', )
				print(redundant_ratio)
				print(len(redundant_ratio))
				capacity = self.memory_constraint - \
					mean(redundant_ratio)*55.22/self.args.num_batch
				print('capacity: ', capacity)
				return
				adjust = 1000
				# estimated_mem = Estimate_MEM(bkt_dst_nodes_list)
				estimated_mem = [0.031600323026579925, 0.053446445057834434, 0.04691033726707499, 0.07212925883696267, 0.0954132446010461, 0.13250813817436047, 0.16562827234049787, 0.18126462923828512, 0.21130672298992675, 0.25300076929852366, 0.2809490893635299, 0.28129312471449885, 0.33190986587898375, 0.36230173630435075, 0.3834405979819673, 0.38852240658495635, 0.4104866247767621, 0.427057239492208, 0.45594087203866557, 0.4482479429953582, 0.494359802184077, 0.5455698065359045, 0.5838345744003708, 0.5952225418284881,
									0.6416539241286929, 0.6823511784373357, 0.666389745486164, 0.7496792492248849, 0.7371837931190246, 0.7577242599083827, 0.7889046908693763, 0.8683255342292655, 0.9311795745279405, 0.8477295250909833, 0.9436967117287708, 0.9945587138174034, 1.0309573992937635, 1.0749793136129961, 1.0747561831684673, 1.1274098691910925, 1.2304586825034851, 1.1488268197006972, 1.3300050600793791, 1.2305013597063668, 1.339544299635952, 1.363191539881995, 1.501307503974184, 1.4590092047286807, 1.473764838436366]
				Groups_mem_list, G_BUCKET_ID_list = grouping_fanout_1(
					adjust, estimated_mem, capacity=1.7)
				batches_nid_list = batches_nid_list + G_BUCKET_ID_list
				weights_list = weights_list
				global_batched_seeds_list = batches_nid_list
			return global_batched_seeds_list, weights_list

	def _products_backpack_bucketing(self, bkt_dst_nodes_list):
		fanout_dst_nids = bkt_dst_nodes_list[-1] # last bucket nodes nids
		fanout = len(bkt_dst_nodes_list)  # the total number of buckets
		print('type of fanout_dst_nids ', type(fanout_dst_nids))
		global_split_batches_nid_list = list(torch.chunk(fanout_dst_nids, self.K)) 
		# split the fanout N nodes into K partition
		adjust = 1000
		estimated_mem = []
		if "25_backpack_" in self.selection_method:  # 2-layers
			if self.hidden == 32:
				estimated_mem = [0.030038309142768727, 0.07084391270112667, 0.12535225476460024, 0.20481898381526728, 0.29499237421925145, 0.4037621224400672, 0.47553709239121866, 0.6368548657451465, 0.6928294528722763, 0.8294504749726038, 0.9602086547889414, 0.9185760502540227, 1.1564732894776686, 1.147224248034353, 1.2155314409701379, 1.3127071745620997, 1.494218664138062, 1.429131172109469, 1.61310518345536, 1.497551695710478, 1.7272323720153804, 1.980263964077, 1.9988796256383259, 1.988051485165607]
				if self.num_batch == 8:
					capacity_imp = 5
				if self.num_batch == 14:
					capacity_imp = 4
				elif self.num_batch == 16:
					capacity_imp = 3.6
				elif self.num_batch == 18:
					capacity_imp = 2
				elif self.num_batch == 19:
					capacity_imp = 2
				elif self.num_batch == 20:
					capacity_imp = 2
				elif self.num_batch == 22:
					capacity_imp = 2
				elif self.num_batch == 24:
					capacity_imp = 2
			if self.hidden == 16:
				estimated_mem = [0.030038309142768727, 0.07084391270112667, 0.12535225476460024, 0.20481898381526728, 0.29499237421925145, 0.4037621224400672, 0.47553709239121866, 0.6368548657451465, 0.6928294528722763, 0.8294504749726038, 0.9602086547889414, 0.9185760502540227, 1.1564732894776686, 1.147224248034353, 1.2155314409701379, 1.3127071745620997, 1.494218664138062, 1.429131172109469, 1.61310518345536, 1.497551695710478, 1.7272323720153804, 1.980263964077, 1.9988796256383259, 1.988051485165607]
				if self.num_batch == 8:
					capacity_imp = 5
				if self.num_batch == 14:
					capacity_imp = 4
				elif self.num_batch == 16:
					capacity_imp = 3.6
				elif self.num_batch == 18:
					capacity_imp = 2
				elif self.num_batch == 19:
					capacity_imp = 2
				elif self.num_batch == 20:
					capacity_imp = 2
				elif self.num_batch == 22:
					capacity_imp = 2
				elif self.num_batch == 24:
					capacity_imp = 2

			print('sum(estimated_mem)')
			print(sum(estimated_mem))
			print(len(estimated_mem))
							
			capacity_imp = max(estimated_mem)   # self.K = 17
							
			if max(estimated_mem) > capacity_imp:
				print('max degree bucket (1-fanout-1) >capacity')
				print('we can reschedule split K-->K+1 ')
				self.K = self.K + 1

		elif "10_backpack_" in self.selection_method:  # 1-layer
			if self.hidden == 256: 
				estimated_mem = [0.00116005539894104, 0.002963840961456299, 0.007523596286773682, 0.012230873107910156,0.014919787645339966, 0.02212822437286377, 0.030744820833206177, 0.0274658203125, 0.03476142883300781]
						#   res mem [1, fanout-1]: 0.1538984477519989 GB
						# mem size of fanout degree bucket by formula (GB):  12.915439903736115
				capacity_imp = 0.1
		elif "20_backpack_" in self.selection_method:  # 1-layer
				estimated_mem = [0.00116005539894104, 0.002963840961456299, 0.007523596286773682, 0.012230873107910156, 0.014919787645339966, 0.02212822437286377, 0.030744820833206177, 0.0274658203125, 0.03476142883300781,0.04291534423828125, 0.04602670669555664, 0.04691183567047119, 0.053610652685165405, 0.06524473428726196, 0.06769225001335144, 0.06587505340576172, 0.06338059902191162, 0.07290244102478027, 0.07440447807312012]
						#   res mem [1, fanout-1]: 0.7528625428676605 GB
						# mem size of fanout degree bucket by formula (GB):  25.001004338264465
				capacity_imp = 0.44
		elif "800_backpack_" in self.selection_method:  # 1-layer
				ff = '/home/cc/Betty_baseline/pytorch/bucketing/fanout_est_mem/fanout_800_est_mem.txt'
				estimated_mem = read_est_mem(ff)[:-1]
						#   res mem [1, fanout-1]: 195.9 GB
						# mem size of fanout degree bucket by formula (GB):  18.56 GB
				capacity_imp = 15.9

		time_backpack_start = time.time()

		Groups_mem_list, G_BUCKET_ID_list = grouping_fanout_products(adjust, estimated_mem, capacity=capacity_imp)
		print("G_BUCKET_ID_list", G_BUCKET_ID_list)

		print("G_BUCKET_ID_list length", len(G_BUCKET_ID_list))
		g_bucket_nids_list = self.get_nids_by_degree_bucket_ID(G_BUCKET_ID_list, bkt_dst_nodes_list)


		for j in range(len(g_bucket_nids_list)):

			current_group_mem = get_sum(G_BUCKET_ID_list[j], estimated_mem)
			print('current_group_mem ', current_group_mem)
			global_split_batches_nid_list[j] = torch.cat(
				(global_split_batches_nid_list[j], g_bucket_nids_list[j]))
			# local_split_batches_nid_list[j]=torch.tensor(g_bucket_nids_list[j])

		weights_list = self.get_weights(bkt_dst_nodes_list, len(self.output_nids))
		
		return global_split_batches_nid_list, weights_list

	def _group_bucketing(self, bkt_dst_nodes_list):
		if '25_group_' in self.selection_method:
			print('group 1 start =========================')
			group1 = bkt_dst_nodes_list[:-1]
			group1 = torch.cat(group1)
			print()
			fanout_dst_nids = bkt_dst_nodes_list[-1]

			if self.args.num_batch > 1:
				fanout_batch_size = ceil(
					len(fanout_dst_nids)/(self.args.num_batch-1))
			indices = torch.arange(0, len(fanout_dst_nids)).long()
			map_output_list = fanout_dst_nids.view(
				-1)[indices].view(fanout_dst_nids.size())
			batches_nid_list = [map_output_list[i:i + fanout_batch_size]
								for i in range(0, len(map_output_list), fanout_batch_size)]

			batches_nid_list.insert(0, group1)

			print(len(batches_nid_list))
			length = len(self.output_nids)
			print('length ', length)
			print('group1 ', len(group1))

			self.weights_list = [
				len(batch_nids)/length for batch_nids in batches_nid_list]
			self.local_batched_seeds_list = batches_nid_list
			print(self.weights_list)
			return batches_nid_list, weights_list
		elif '50_group_' in self.selection_method:
			print('__ ')
			print(len(bkt_dst_nodes_list))
			# for ii in range(len(bkt_dst_nodes_list)):
			# 	print(bkt_dst_nodes_list[ii][:10])
			print('group 1 start =========================')
			group1 = bkt_dst_nodes_list[:40]
			# for ii in range(len(group1)):
			# 	print(group1[ii][:10])
			group1 = torch.cat(group1)
			print()
			print('group 2 start=========================')
			group2 = bkt_dst_nodes_list[40:49]
			# for ii in range(len(group2)):
			# 	print(group2[ii][:10])
			group2 = torch.cat(group2)
			# print('group 3 start=========================')
			# group3 =  bkt_dst_nodes_list[46:48]
			# group3 = torch.cat(group3)

			fanout_dst_nids = bkt_dst_nodes_list[-1]

			if self.args.num_batch > 2:
				fanout_batch_size = ceil(
					len(fanout_dst_nids)/(self.args.num_batch-2))
			indices = torch.arange(0, len(fanout_dst_nids)).long()
			map_output_list = fanout_dst_nids.view(
				-1)[indices].view(fanout_dst_nids.size())
			batches_nid_list = [map_output_list[i:i + fanout_batch_size]
								for i in range(0, len(map_output_list), fanout_batch_size)]
			# batches_nid_list.insert(0, group3)
			batches_nid_list.insert(0, group2)
			batches_nid_list.insert(0, group1)

			print(len(batches_nid_list))
			length = len(self.output_nids)
			print('length ', length)
			print('group1 ', len(group1))
			print('group2 ', len(group2))
			weights_list = [
				len(batch_nids)/length for batch_nids in batches_nid_list]
			
			# pdb.set_trace()
			return batches_nid_list, weights_list
		elif '100_group_' in self.selection_method:
			print('__ ')
			print(len(bkt_dst_nodes_list))

			print('group 1 start =========================')
			group1 = bkt_dst_nodes_list[:40]
			group1 = torch.cat(group1)

			print('group 2 start=========================')
			group2 = bkt_dst_nodes_list[40:53]
			group2 = torch.cat(group2)

			print('group 3 start=========================')
			group3 = bkt_dst_nodes_list[53:63]
			group3 = torch.cat(group3)

			print('group 4 start=========================')
			group4 = bkt_dst_nodes_list[63:70]
			group4 = torch.cat(group4)

			print('group 5 start=========================')
			group5 = bkt_dst_nodes_list[70:76]
			group5 = torch.cat(group5)

			print('group 6 start=========================')
			group6 = bkt_dst_nodes_list[76:81]
			group6 = torch.cat(group6)
			print('group 7 start=========================')
			group7 = bkt_dst_nodes_list[81:86]
			group7 = torch.cat(group7)
			print('group 8 start=========================')
			group8 = bkt_dst_nodes_list[86:90]
			group8 = torch.cat(group8)
			print('group 9 start=========================')
			group9 = bkt_dst_nodes_list[90:93]
			group9 = torch.cat(group9)
			print('group 10 start=========================')
			group10 = bkt_dst_nodes_list[93:96]
			group10 = torch.cat(group10)
			print('group 11 start=========================')
			group11 = bkt_dst_nodes_list[96:99]
			group11 = torch.cat(group11)
			fanout_dst_nids = bkt_dst_nodes_list[-1]

			if self.args.num_batch > 11:
				fanout_batch_size = ceil(
					len(fanout_dst_nids)/(self.args.num_batch-11))
			indices = torch.arange(0, len(fanout_dst_nids)).long()
			map_output_list = fanout_dst_nids.view(
				-1)[indices].view(fanout_dst_nids.size())
			batches_nid_list = [map_output_list[i:i + fanout_batch_size]
								for i in range(0, len(map_output_list), fanout_batch_size)]
			batches_nid_list.insert(0, group11)
			batches_nid_list.insert(0, group10)
			batches_nid_list.insert(0, group9)
			batches_nid_list.insert(0, group8)
			batches_nid_list.insert(0, group7)
			batches_nid_list.insert(0, group6)
			batches_nid_list.insert(0, group5)
			batches_nid_list.insert(0, group4)
			batches_nid_list.insert(0, group3)
			batches_nid_list.insert(0, group2)
			batches_nid_list.insert(0, group1)

			print(len(batches_nid_list))
			length = len(self.output_nids)
			print('length ', length)
			print('group1 ', len(group1))
			print('group2 ', len(group2))
			print('group7 ', len(group7))
			weights_list = [
				len(batch_nids)/length for batch_nids in batches_nid_list]
			return batches_nid_list, weights_list
		else:
			print('group  start =========================')
			
			group1 = torch.cat(bkt_dst_nodes_list[:-1])
			print()
			fanout_dst_nids = bkt_dst_nodes_list[-1]

			if self.args.num_batch > 1:
				fanout_batch_size = len(fanout_dst_nids) // self.args.num_batch
				if len(fanout_dst_nids) % self.args.num_batch != 0:
					fanout_batch_size += 1
			# fanout_batch_size = 2 ######
			indices = torch.arange(0, len(fanout_dst_nids)).long()
			map_output_list = fanout_dst_nids.view(
				-1)[indices].view(fanout_dst_nids.size())
			batches_nid_list = [map_output_list[i:i + fanout_batch_size]
								for i in range(0, len(map_output_list), fanout_batch_size)]

			batches_nid_list.insert(0, group1)

			print(len(batches_nid_list))
			length = len(self.output_nids)
			print('length ', length)
			print('group1 ', len(group1))

			weights_list = [
				len(batch_nids)/length for batch_nids in batches_nid_list]
			
			return batches_nid_list, weights_list


	def _custom_bucketing(self, bkt_dst_nodes_list):
		print('custom ')
		print(len(bkt_dst_nodes_list))
		for ii in range(len(bkt_dst_nodes_list)):
			print(bkt_dst_nodes_list[ii][:10])
		print('group 1=========================')
		group1 = bkt_dst_nodes_list[2:14]
		for ii in range(len(group1)):
			print(group1[ii][:10])
		group1 = torch.cat(group1)
		print('group 2=========================')
		group2 = bkt_dst_nodes_list[:2] + bkt_dst_nodes_list[14:24]
		for ii in range(len(group2)):
			print(group2[ii][:10])
		group2 = torch.cat(group2)
		split = bkt_dst_nodes_list[-1]
		num_split = int(len(split)/2)

		split_1 = split[:num_split]
		split_2 = split[num_split:]
		batches_nid_list = [group1, group2, split_1, split_2]

		print(len(batches_nid_list))
		length = len(self.output_nids)
		print('length ', length)
		print('group1 ', len(group1))
		weights_list = [
			len(batch_nids)/length for batch_nids in batches_nid_list]
		
		return batches_nid_list, weights_list

	def _range_bucketing(self, bkt_dst_nodes_list):
		fanout_batch_size = ceil((fanout_dst_nids)/(self.args.num_batch-1))
		indices = torch.arange(0, len(fanout_dst_nids)).long()
		map_output_list = fanout_dst_nids.view(
			-1)[indices].view(fanout_dst_nids.size())
		batches_nid_list = [map_output_list[i:i + fanout_batch_size]
							for i in range(0, len(map_output_list), fanout_batch_size)]
		weights_list = [
			len(batch_nids)/length for batch_nids in batches_nid_list]
		
		return batches_nid_list, weights_list

	def _random_bucketing(self, bkt_dst_nodes_list):
		fanout_batch_size = ceil((fanout_dst_nids)/(self.args.num_batch-1))
		indices = torch.randperm(len(fanout_dst_nids))
		map_output_list = fanout_dst_nids.view(
			-1)[indices].view(fanout_dst_nids.size())
		batches_nid_list = [map_output_list[i:i + fanout_batch_size]
							for i in range(0, len(map_output_list), fanout_batch_size)]
		weights_list = [
			len(batch_nids)/length for batch_nids in batches_nid_list]
		
		return batches_nid_list, weights_list


	def gen_batches_seeds_list(self, bkt_dst_nodes_list):
		if self.args.num_batch <= 1:
			print('no need to split fanout degree, full batch train ')
			# self.local_batched_seeds_list = bkt_dst_nodes_list
			return bkt_dst_nodes_list, [1]

		print('---||--'*20)
		print('self.num_batch, ', self.num_batch)
		length = len(self.output_nids)
		if 'random' in self.selection_method:
			self.global_batched_seeds_list, self.weights_list = self._random_bucketing(bkt_dst_nodes_list)
		elif 'range' in self.selection_method:
			self.global_batched_seeds_list, self.weights_list = self._range_bucketing(bkt_dst_nodes_list)
		
		elif "bucketing" in self.selection_method:
			total_len = len(bkt_dst_nodes_list)
			print('len(bkt_dst_nodes_list) ', len(bkt_dst_nodes_list))
			tensor_lengths = [t.numel() for t in bkt_dst_nodes_list]
			# print('bkt_dst_nodes_list ', bkt_dst_nodes_list)
			if 'fanout' in self.selection_method:
				self.global_batched_seeds_list, self.weights_list = fanout_bucketing(self, bkt_dst_nodes_list, self.output_nids)
			
			elif '50_backpack_' in self.selection_method:
				self.global_batched_seeds_list, self.weights_list = _50_backpack_bucketing(self, bkt_dst_nodes_list)
				
			elif 'products' in self.selection_method:
				if "_backpack_" in self.selection_method:
					self.global_batched_seeds_list, self.weights_list = _products_backpack_bucketing(self, bkt_dst_nodes_list)
					
					
			elif 'cora_' in self.selection_method or 'pubmed_' in self.selection_method:

				print('memory_constraint: ', self.memory_constraint)

				adjust = 1000
				time_backpack_start = time.time()
				if '30_backpack_' in self.selection_method:
					if 'cora_' in self.selection_method:
						if self.hidden == 2048:
							est_mem_dict = {1: 0.4396008476614952, 2: 1.374441683292389, 3: 1.5948124453425407, 4: 2.2839434891939163, 5: 1.8786997273564339, 6: 1.5771530494093895, 7: 0.6665662229061127,
											8: 0.4780137687921524, 9: 0.8689616918563843, 10: 0.6201625987887383, 11: 0.16984806954860687, 12: 0.40121957659721375, 19: 0.27158090472221375, 21: 0.42578747123479843, 30: 1.1442232578992844}
						# if self.num_batch == 2:
						# 	capacity_imp = 0.045
						# elif self.num_batch == 4:
						# 	capacity_imp = 0.023

					elif 'pubmed_' in self.selection_method:
						est_mem_dict = {1: 2.2797432839870453, 2: 0.9267187714576721, 3: 1.3459078073501587, 4: 0.9429364800453186, 5: 0.6352187097072601, 6: 1.315835952758789, 7: 0.07466062903404236, 8: 1.1979998052120209,
										9: 0.36177903413772583, 10: 0.15526315569877625, 11: 0.35477203130722046, 17: 1.8221600353717804, 18: 1.1670382618904114, 22: 1.2326273918151855, 29: 0.24611574411392212, 30: 1.5610657632350922}
					Groups_mem_list, G_BUCKET_ID_list = grouping_cora(
						adjust, est_mem_dict, capacity_imp, 30, self.K)
					print('sum(estimated_mem)')
					print(sum(est_mem_dict.values()))
					print(len(est_mem_dict))

					capacity_imp = self.memory_constraint
					if max(est_mem_dict.values()) > capacity_imp:
						print('max degree bucket (1-fanout) >capacity')
						print('we can reschedule split K-->K+1 ')
						self.K = self.K + 1
					print('self.K ', self.K)
				elif '25_backpack_' in self.selection_method:
					print('enter 25_backpack_')
					if 'cora_' in self.selection_method:
						print('enter 25_backpack_cora')
						if self.hidden == 256:
							est_mem_dict = {1: 0.009952336549758911, 2: 0.029204897582530975, 3: 0.039294563233852386, 4: 0.054936736822128296, 5: 0.04942861944437027, 6: 0.03956902027130127, 7: 0.017872966825962067,
											8: 0.014537237584590912, 9: 0.02214217185974121, 10: 0.018099479377269745, 11: 0.005473785102367401, 12: 0.014345057308673859, 19: 0.010031260550022125, 21: 0.014677919447422028, 25: 0.028916627168655396}
						elif self.hidden == 1024:
							est_mem_dict = {1: 0.010982304811477661, 2: 0.03177981823682785, 3: 0.04331143945455551, 4: 0.059468597173690796, 5: 0.05329100042581558, 6: 0.042967915534973145, 7: 0.019314922392368317,
											8: 0.015773199498653412, 9: 0.02399611473083496, 10: 0.01964443176984787, 11: 0.006040267646312714, 12: 0.015581019222736359, 19: 0.011009730398654938, 21: 0.015759386122226715, 25: 0.03033846616744995}
							if self.num_batch == 2:
								capacity_imp = 0.21
							elif self.num_batch == 3:
								capacity_imp = 0.196
							elif self.num_batch == 4:
								capacity_imp = 0.10
							elif self.num_batch == 5:
								capacity_imp = 0.08
							elif self.num_batch == 6:
								capacity_imp = 0.07
							elif self.num_batch == 7:
								capacity_imp = 0.06
						print('self.num_batch cora_', self.num_batch)
						# capacity_imp = self.memory_constraint
						print('self.hidden ', self.hidden)
						print('capacity_imp ', capacity_imp)
					if 'pubmed_' in self.selection_method:
						if self.hidden == 256:
							est_mem_dict = {1: 0.008472561836242676, 2: 0.004270613193511963, 3: 0.00498005747795105, 4: 0.0036263465881347656, 5: 0.0025652647018432617, 6: 0.008285075426101685, 7: 0.001278609037399292,
											8: 0.007463514804840088, 9: 0.0038363635540008545, 10: 0.0014977455139160156, 11: 0.002338886260986328, 17: 0.010301828384399414, 18: 0.006973743438720703, 22: 0.005353689193725586, 25: 0.015277057886123657}
							if self.num_batch == 2:
								capacity_imp = 0.045
							elif self.num_batch == 4:
								capacity_imp = 0.023
						elif self.hidden == 1024:
							print('enter pubmed_')
							est_mem_dict = {1: 0.028659939765930176, 2: 0.014467298984527588, 3: 0.018009155988693237, 4: 0.012896060943603516, 5: 0.009054064750671387, 6: 0.03022339940071106, 7: 0.0040080249309539795,
											8: 0.027238905429840088, 9: 0.01423904299736023, 10: 0.004381656646728516, 11: 0.007746219635009766, 17: 0.03718400001525879, 18: 0.006973743438720703, 22: 0.019000768661499023, 25: 0.05457034707069397}
							# sum is 0.288 GB
							# modified sum 1.699 GB
							if self.num_batch == 2:
								capacity_imp = 0.145
							elif self.num_batch == 3:
								capacity_imp = 0.1
							elif self.num_batch == 4:
								capacity_imp = 0.09
							elif self.num_batch == 5:
								capacity_imp = 0.06
							elif self.num_batch == 6:
								capacity_imp = 0.055
							elif self.num_batch == 7:
								capacity_imp = 0.055
							elif self.num_batch == 8:
								capacity_imp = 0.055
					print('sum(estimated_mem) ', sum(est_mem_dict.values()))

					print('len(estimated_mem) ', len(est_mem_dict))
					# time_backpack_start = time.time()

					Groups_mem_list, G_BUCKET_ID_list = grouping_cora(
						adjust, est_mem_dict, capacity_imp, 25, self.K)
					# print('sum(estimated_mem) ', sum(est_mem_dict.values()))
					if len(Groups_mem_list) != self.num_batch:
						print('!!!! len(Groups_mem_list) ',
							len(Groups_mem_list))
						print('!!!!  self.num_batch ', self.num_batch)
						if len(Groups_mem_list) > self.num_batch:
							print('Groups_mem_list', Groups_mem_list)
							return

				elif '10_backpack_' in self.selection_method:
					if 'cora_' in self.selection_method:
						est_mem_dict = {1: 0.0019218027591705322, 2: 0.004804506897926331, 3: 0.007495030760765076, 4: 0.008455932140350342, 5: 0.007206760346889496,
										6: 0.006341949105262756, 7: 0.002690523862838745, 8: 0.0023061633110046387, 9: 0.003459244966506958, 10: 0.009609013795852661}
					capacity_imp = sum(est_mem_dict.values())/self.num_batch
					Groups_mem_list, G_BUCKET_ID_list = grouping_cora(
						adjust, est_mem_dict, capacity_imp, 10, self.K)
				print("G_BUCKET_ID_list", G_BUCKET_ID_list)
				print("Groups_mem_list ", Groups_mem_list)

				print("G_BUCKET_ID_list length", len(G_BUCKET_ID_list))
				g_bucket_nids_list = self.get_nids_by_degree_bucket_ID(
					G_BUCKET_ID_list, bkt_dst_nodes_list)

				time_backpack_end = time.time()
				print('backpack scheduling spend ',
					time_backpack_end-time_backpack_start)
				split_batches_nid_list = []

				time_batch_gen_start = time.time()
				for j in range(len(g_bucket_nids_list)):
					tensor_group = torch.tensor(
						g_bucket_nids_list[j], dtype=torch.long)
					current_group_mem = get_sum(
						G_BUCKET_ID_list[j], list(est_mem_dict.values()))
					print("current group_mem ", current_group_mem)

					split_batches_nid_list.append(tensor_group)

				time_batch_gen_end = time.time()
				print('batches output list generation spend ',
					time_batch_gen_end-time_batch_gen_start)
				length = len(self.output_nids)
				self.weights_list = [
					len(batch_nids)/length for batch_nids in split_batches_nid_list]
				print('self.weights_list ', self.weights_list)

				self.local_batched_seeds_list = split_batches_nid_list
				return
			elif 'reddit_' in self.selection_method:
				if '_backpack_' in self.selection_method:
					time_split_start = time.time()
					fanout_dst_nids = bkt_dst_nodes_list[-1]
					fanout = len(bkt_dst_nodes_list)

					if self.args.num_batch >= 1:
						fanout_batch_size = ceil(len(fanout_dst_nids)/(self.K))
					indices = torch.arange(0, len(fanout_dst_nids)).long()
					map_output_list = fanout_dst_nids.view(
						-1)[indices].view(fanout_dst_nids.size())

					# split_batches_nid_list = split_list(map_output_list, self.K)

					split_batches_nid_list = [map_output_list[i:i + fanout_batch_size]
											for i in range(0, len(map_output_list), fanout_batch_size)]
					# print("print(split_batches_nid_list) ",split_batches_nid_list)
					# return
					# ct = time.time()
					# src_list, weights_list, time_collection = generate_K_hop_neighbors(self.full_batch_dataloader, self.args, split_batches_nid_list)
					# print('generate_K_hop_neighbors time ', time.time()-ct)
					# redundant_ratio = []
					# for (i, input_nodes) in enumerate(src_list):
					# 	print(len(input_nodes)/len(split_batches_nid_list[i])/fanout/0.226)
					# 	redundant_ratio.append(len(input_nodes)/len(split_batches_nid_list[i])/fanout/0.226)
					# print('the split redundant ratio ', )
					# print(redundant_ratio)
					# print(len(redundant_ratio))
					# time_split_end = time.time()
					# print('split fanout degree bucket spend /sec: ', time_split_end - time_split_start)

					adjust = 1000
					if '10_backpack_' in self.selection_method:
						estimated_mem = [0.05615083873271942, 0.09857681393623352, 0.1275201290845871, 0.15920841693878174,
										0.18568933010101318, 0.21192803978919983, 0.23001256585121155, 0.25802743434906006, 0.2524971216917038]
						if self.num_batch == 3:
							capacity_imp = self.memory_constraint-17.45
						elif self.num_batch == 4:
							capacity_imp = self.memory_constraint-17.58
						elif self.num_batch == 5:
							capacity_imp = self.memory_constraint-17.65
						elif self.num_batch == 6:
							capacity_imp = self.memory_constraint-17.70
						elif self.num_batch == 7:
							capacity_imp = self.memory_constraint-17.74
						elif self.num_batch == 8:
							capacity_imp = self.memory_constraint-17.74
						if self.model == "GAT":  # GAT+lstm
							if self.num_batch == 4:
								capacity_imp = 0.4
							elif self.num_batch == 5:
								capacity_imp = 0.344
							elif self.num_batch == 6:
								capacity_imp = 0.3
							elif self.num_batch == 7:
								capacity_imp = 0.29
							elif self.num_batch == 8:
								capacity_imp = 0.26
							elif self.num_batch == 9:
								capacity_imp = 0.258
							elif self.num_batch == 10:
								capacity_imp = 0.258
							elif self.num_batch == 12:
								capacity_imp = 0.258
							elif self.num_batch == 14:
								capacity_imp = 0.258
							elif self.num_batch == 16:
								capacity_imp = 0.258
							elif self.num_batch == 18:
								capacity_imp = 0.258
					elif '25_backpack_' in self.selection_method:
						if self.hidden == 256:
							# fanout split 136.9*8/50 = 22 GB # graphsage + lstm
							estimated_mem = [1.3390388304942122, 1.1511607853654562, 0.9968803828513182, 1.1330254407898046, 1.273968707844615, 1.4271276523896628, 1.482011009895538, 1.6097993905165318, 1.6116104018773965, 1.6784923510901304, 1.8729928371372757, 1.915525670833232,
											2.0752913576516465, 2.0264824781498296, 2.1645393796554804, 2.1370398432040925, 2.1997352567082333, 2.1771046663129945, 2.3384675421966064, 2.2770233221822953, 2.454792454762997, 2.4747481289605364, 2.518576996875674, 2.593600036529913]
							if self.num_batch == 51:
								# maximum of est mem [1,fanout-1]
								capacity_imp = 2.6
							if self.model == "GAT":
								if self.num_batch == 180:  # GAT+lstm
									# maximum of est mem [1,fanout-1]
									capacity_imp = 2.6
								elif self.num_batch == 200:  # GAT+lstm
									# maximum of est mem [1,fanout-1]
									capacity_imp = 2.6
					# elif '30_backpack_' in self.selection_method:
					# 	estimated_mem = [11.09133707869788, 7.949351660231331, 4.247930594170108, 3.767815723282392, 3.521911913357682, 3.3171698192934964, 3.1136675746243436, 2.9493490757618086, 2.783152518733855, 2.690315310282632, 2.4780320925231085, 2.4736405822131586, 2.424331795810457, 2.3423791134065306, 2.2508307132173453, 2.0888736283425056, 2.1393341709313427, 2.0144231711761864, 2.028492122175591, 1.8952586057017728, 1.9272310948984677, 1.7648288977543771, 1.8025470147872276, 1.7435488087790354, 1.624197941655698, 1.6781451757272114, 1.585553364875989, 1.5579015641599088, 1.508019266507371]
					# 	capacity_imp = self.memory_constraint-10 # nb 11 capcity = 8

					time_backpack_start = time.time()

					print('self.K ', self.K)
					Groups_mem_list, G_BUCKET_ID_list = grouping_fanout_arxiv(
						adjust, estimated_mem, capacity_imp, fanout, self.K)
					print("G_BUCKET_ID_list", G_BUCKET_ID_list)
					print("Groups_mem_list ", Groups_mem_list)
					if len(G_BUCKET_ID_list) > self.K:
						print('------------errror-----------------')

					print("G_BUCKET_ID_list length", len(G_BUCKET_ID_list))
					g_bucket_nids_list = self.get_nids_by_degree_bucket_ID(
						G_BUCKET_ID_list, bkt_dst_nodes_list)

					time_backpack_end = time.time()
					print('backpack scheduling spend ',
						time_backpack_end-time_backpack_start)

					time_batch_gen_start = time.time()
					print(len(g_bucket_nids_list))
					print(len(split_batches_nid_list))
					for j in range(len(g_bucket_nids_list)):
						tensor_group = torch.tensor(
							g_bucket_nids_list[j], dtype=torch.long)
						current_group_mem = get_sum(
							G_BUCKET_ID_list[j], estimated_mem)
						print("current group_mem ", current_group_mem)

						split_batches_nid_list[j] = torch.cat(
							(split_batches_nid_list[j], tensor_group))  # split + group
						# split_batches_nid_list[j] = tensor_group # group only

					time_batch_gen_end = time.time()
					print('batches output list generation spend ',
						time_batch_gen_end-time_batch_gen_start)
					length = len(self.output_nids)
					self.weights_list = [
						len(batch_nids)/length for batch_nids in split_batches_nid_list]
					# print('self.weights_list ', self.weights_list)

					self.local_batched_seeds_list = split_batches_nid_list

					return

			elif 'arxiv_' in self.selection_method:
				if '_25' in self.selection_method:
					print('_25_   len(bkt_dst_nodes_list) ', len(bkt_dst_nodes_list))
					self.global_batched_seeds_list, self.weights_list = self._arxiv_2_layer_backpack_bucketing(bkt_dst_nodes_list)
				elif '_10'  in self.selection_method:
					self.global_batched_seeds_list, self.weights_list = self._arxiv_1_layer_backpack_bucketing(bkt_dst_nodes_list)

			elif 'group_' in self.selection_method:
				self.global_batched_seeds_list, self.weights_list = self._group_bucketing(bkt_dst_nodes_list)
			
			elif 'custom' in self.selection_method:
				self.global_batched_seeds_list, self.weights_list = self._custom_bucketing(bkt_dst_nodes_list)

		return

	def get_src_len(self, seeds):
		in_nids = self.layer_block.in_edges(seeds)[0]
		src = torch.unique(in_nids)
		return src.size()

	def get_partition_src_len_list(self):
		partition_src_len_list = []
		for seeds_nids in self.local_batched_seeds_list:
			partition_src_len_list.append(self.get_src_len(seeds_nids))

		self.partition_src_len_list = partition_src_len_list
		self.partition_len_list = partition_src_len_list
		return

	def buckets_partition(self):

		bkt_dst_nodes_list_global = self.get_in_degree_bucketing()  # degree bucketing
		print('bkt_dst_nodes_list_global ', bkt_dst_nodes_list_global)
		# based on memory estiamtion to group
		self.gen_batches_seeds_list(bkt_dst_nodes_list_global)

		return self.global_batched_seeds_list, self.weights_list, self.partition_len_list
