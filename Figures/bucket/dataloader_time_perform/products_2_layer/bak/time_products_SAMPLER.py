import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../../')
sys.path.insert(0,'../../../pytorch/utils')
sys.path.insert(0,'../../../pytorch/bucketing')
sys.path.insert(0,'../../../pytorch/models')
sys.path.insert(0,'../../../memory_logging')
# from runtime_nvidia_smi import start_memory_logging, stop_memory_logging
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/utils')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/models')
from bucketing_dataloader import generate_dataloader_bucket_block
from bucketing_dataloader import dataloader_gen_bucketing
from bucketing_dataloader import dataloader_gen_bucketing_time
import dgl
from dgl.data.utils import save_graphs
import numpy as np
from statistics import mean
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os



import dgl.nn.pytorch as dglnn
import time
import argparse


import random
from graphsage_model_wo_mem import GraphSAGE
import dgl.function as fn
from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate, prepare_data, load_pubmed

from load_graph import load_ogbn_dataset
from memory_usage import see_memory_usage, nvidia_smi_usage

from cpu_mem_usage import get_memory
from statistics import mean

from my_utils import parse_results


import pickle
from utils import Logger
import os 




def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.device >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True
		dgl.seed(args.seed)
		dgl.random.seed(args.seed)

def CPU_DELTA_TIME(tic, str1):
	toc = time.time()
	print(str1 + ' spend:  {:.6f}'.format(toc - tic))
	return toc


def compute_acc(pred, labels):
	"""
	Compute the accuracy of prediction given the labels.
	"""
	labels = labels.long()
	return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeats, labels, train_nid, val_nid, test_nid, device, args):
	"""
	Evaluate the model on the validation set specified by ``val_nid``.
	g : The entire graph.
	inputs : The features of all the nodes.
	labels : The labels of all the nodes.
	val_nid : the node Ids for validation.
	device : The GPU device to evaluate on.
	"""
	# train_nid = train_nid.to(device)
	# val_nid=val_nid.to(device)
	# test_nid=test_nid.to(device)
	nfeats=nfeats.to(device)
	g=g.to(device)
	# print('device ', device)
	model.eval()
	with torch.no_grad():
		# pred = model(g=g, x=nfeats)
		pred = model.inference(g, nfeats,  args, device)
	model.train()

	train_acc= compute_acc(pred[train_nid], labels[train_nid].to(pred.device))
	val_acc=compute_acc(pred[val_nid], labels[val_nid].to(pred.device))
	test_acc=compute_acc(pred[test_nid], labels[test_nid].to(pred.device))
	return (train_acc, val_acc, test_acc)


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = nfeat[input_nodes].to(device)
	batch_labels = labels[seeds].to(device)
	return batch_inputs, batch_labels

def load_block_subtensor(nfeat, labels, blocks, device,args):
	"""
	Extracts features and labels for a subset of nodes
	"""

	# if args.GPUmem:
	# 	see_memory_usage("----------------------------------------before batch input features to device")
	batch_inputs = nfeat[blocks[0].srcdata[dgl.NID]].to(device)
	# if args.GPUmem:
	# 	see_memory_usage("----------------------------------------after batch input features to device")
	batch_labels = labels[blocks[-1].dstdata[dgl.NID]].to(device)
	
	# if args.GPUmem:
	# 	see_memory_usage("----------------------------------------after  batch labels to device")
	return batch_inputs, batch_labels

def get_compute_num_nids(blocks):
	res=0
	for b in blocks:
		res+=len(b.srcdata['_ID'])
	return res


def get_FL_output_num_nids(blocks):

	output_fl =len(blocks[0].dstdata['_ID'])
	return output_fl
# def swap(split_list):
# 	index1 = 0
# 	index2 = len(split_list)-1
# 	temp = split_list[index1]
# 	split_list[index1] = split_list[index2]
# 	split_list[index2] = temp
# 	return split_list
# def split_tensor(tensor, num_parts):
# 	N = tensor.size(0)
# 	split_size = N // num_parts
# 	if N % num_parts != 0:
# 		split_size += 1

# 	# Split the tensor into two parts
# 	split_tensors = torch.split(tensor, split_size)

# 	# Convert the split tensors into a list
# 	split_list = list(split_tensors)
 
# 	split_list = swap(split_list)
	
# 	weight_list = [len(part) / N for part in split_tensors]
# 	return split_list, weight_list

# def dataloader_gen(full_batch_dataloader,g,processed_fan_out, num_batch):
# 	block_dataloader = []
# 	blocks_list=[]
# 	weights_list=[]
	
# 	for step , (src, dst, full_blocks) in enumerate(full_batch_dataloader):
		
# 		dst_list, weights_list = split_tensor(dst, num_batch) #######
# 		final_dst_list = dst_list
# 		pre_dst_list=[]
# 		for layer , full_block in enumerate(reversed(full_blocks)):
# 			layer_block_list=[]
			
# 			layer_graph = dgl.edge_subgraph(g, full_block.edata['_ID'],relabel_nodes=False,store_ids=True)
# 			src_len = len(full_block.srcdata['_ID'])
# 			layer_graph.ndata['_ID']=torch.tensor([-1]*len(layer_graph.nodes()))
# 			layer_graph.ndata['_ID'][:src_len] = full_block.srcdata['_ID']

# 			if layer == 0:
# 				print('the output layer ')
# 				for i,dst_new in enumerate(dst_list) :
# 					sg1 = dgl.sampling.sample_neighbors_range(layer_graph, dst_new, processed_fan_out[-1])
# 					block = dgl.to_block(sg1,dst_new, include_dst_in_src= True)
# 					pre_dst_list.append(block.srcdata[dgl.NID]) 
# 					layer_block_list.append(block)
# 			elif layer == 1:
# 				print('input layer')
# 				src_list=[]
# 				for i,dst_new in enumerate(pre_dst_list):
# 					sg1 = dgl.sampling.sample_neighbors_range(layer_graph, dst_new, processed_fan_out[0])
# 					block = dgl.to_block(sg1,dst_new, include_dst_in_src= True)
# 					layer_block_list.append(block)
# 					src_list.append(block.srcdata[dgl.NID]) 
# 				final_src_list = src_list

# 			blocks_list.append(layer_block_list)

# 		blocks_list = blocks_list[::-1]
# 		for batch_id in range(num_batch):
# 			cur_blocks = [blocks[batch_id] for blocks in blocks_list]
# 			dst = final_dst_list[batch_id]
# 			src = final_src_list[batch_id]
# 			block_dataloader.append((src, dst, cur_blocks))
# 	return block_dataloader, weights_list

#### Entry point
def run(args, device, data):
	if args.GPUmem:
		see_memory_usage("----------------------------------------start of run function ")
	# Unpack data
	g, nfeats, labels, n_classes, train_nid, val_nid, test_nid = data
	print('labels ', labels)
	in_feats = len(nfeats[0])
	# print('in feats: ', in_feats)
	nvidia_smi_list=[]

	if args.selection_method =='metis':
		args.o_graph = dgl.node_subgraph(g, train_nid)

	fan_out_list = [fanout for fanout in args.fan_out.split(',')]
	fan_out_list = ' '.join(fan_out_list).split()
	processed_fan_out = [int(fanout) for fanout in fan_out_list] # remove empty string

	
	full_batch_size = len(train_nid)
	args.num_workers = 0
	model = GraphSAGE(
					in_feats,
					args.num_hidden,
					n_classes,
					args.aggre,
					args.num_layers,
					F.relu,
					args.dropout).to(device)

	loss_fcn = nn.CrossEntropyLoss()

	# if args.GPUmem:
	# 	see_memory_usage("----------------------------------------after model to device")
	logger = Logger(args.num_runs, args)
	num_input_list=[]
	pure_train_time_list =[]
	dur = []
	data_loader_gen_time_list=[]
	for run in range(args.num_runs):
		model.reset_parameters()
		# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		for epoch in range(args.num_epochs):
			print('epoch ', epoch)
			model.train()
			if epoch >= args.log_indent:
				t0 = time.time()
			loss_sum=0
			# start of data preprocessing part---s---------s--------s-------------s--------s------------s--------s----
			if args.load_full_batch:
				full_batch_dataloader=[]
				file_name=r'/home/cc/dataset/fan_out_'+args.fan_out+'/'+args.dataset+'_'+str(epoch)+'_items.pickle'
				with open(file_name, 'rb') as handle:
					item=pickle.load(handle)
					full_batch_dataloader.append(item)
			if epoch >= args.log_indent:
				print('load pickle file time ', time.time()-t0) 
			time_dataloader_start=time.time()
			if args.num_batch > 1:
				block_dataloader, weights_list= dataloader_gen_bucketing(full_batch_dataloader,g,processed_fan_out, args)
			else:
				weights_list = [1,]
				block_dataloader=full_batch_dataloader
			time_dataloader_end= time.time()
			print('dataloader gen time ', time_dataloader_end-time_dataloader_start)
			loader_gen_time = time_dataloader_end-time_dataloader_start
			if epoch >= args.log_indent:
				data_loader_gen_time_list.append(loader_gen_time)
			pseudo_mini_loss = torch.tensor([], dtype=torch.long)
			num_input_nids=0
			pure_train_time=0
			print('weights_list', weights_list)
			for step, (input_nodes, seeds, blocks) in enumerate(block_dataloader):
				print('step ', step)
			
				num_input_nids	+= len(input_nodes)
				batch_inputs, batch_labels = load_block_subtensor(nfeats, labels, blocks, device,args)#------------*

				blocks = [block.int().to(device) for block in blocks]#------------*
				t1= time.time()
				batch_pred = model(blocks, batch_inputs)#------------*
				# see_memory_usage("----------------------------------------after batch_pred = model(blocks, batch_inputs)")
					
				loss = loss_fcn(batch_pred, batch_labels)#------------*
				# see_memory_usage("----------------------------------------after loss function")
				pseudo_mini_loss = loss*weights_list[step]#------------*
				pseudo_mini_loss.backward()#------------*
				t2 = time.time()
				pure_train_time += (t2-t1)
				loss_sum += pseudo_mini_loss#------------*

			t3=time.time()
			optimizer.step()
			optimizer.zero_grad()
			t4=time.time()
			pure_train_time += (t4-t3)
			pure_train_time_list.append(pure_train_time)
			print('pure train time ',pure_train_time)
			num_input_list.append(num_input_nids)
			if args.GPUmem:
					see_memory_usage("-----------------------------------------after optimizer zero grad")
			print('----------------------------------------------------------pseudo_mini_loss sum ' + str(loss_sum.tolist()))
		
			if epoch >= args.log_indent:
				dur.append(time.time() - t0)
		print('Total (block generation + training)time/epoch {}'.format(np.mean(dur)))
		print('Total (block generation + training)time/epoch list{}'.format(dur))
		print('pure train time/epoch {}'.format(np.mean(pure_train_time_list[4:])))
		print('dataloader time ', data_loader_gen_time_list)
		print('dataloader time avg per epoch {}'.format(np.mean(data_loader_gen_time_list[4:])))
		print('')
		print('num_input_list ', num_input_list)
		# print('avg block dataloader generation time', np.mean(block_generation_time_list))


					


def main():
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--device', type=int, default=0,
		help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)
	argparser.add_argument('--setseed', type=bool, default=True)
	argparser.add_argument('--GPUmem', type=bool, default=True)
	argparser.add_argument('--load-full-batch', type=bool, default=True)
	# argparser.add_argument('--root', type=str, default='../my_full_graph/')
	# argparser.add_argument('--dataset', type=str, default='ogbn-arxiv')
	# argparser.add_argument('--dataset', type=str, default='ogbn-mag')
	argparser.add_argument('--dataset', type=str, default='ogbn-products')
	# argparser.add_argument('--dataset', type=str, default='cora')
	# argparser.add_argument('--dataset', type=str, default='karate')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	# argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--aggre', type=str, default='lstm')
	argparser.add_argument('--model', type=str, default='SAGE')
	# argparser.add_argument('--selection-method', type=str, default='arxiv_25_backpack_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='products_10_backpack_bucketing')
	argparser.add_argument('--selection-method', type=str, default='products_25_backpack_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='range_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='random_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='fanout_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='group_bucketing')
	argparser.add_argument('--num-batch', type=int, default=14)
	argparser.add_argument('--mem-constraint', type=float, default=18)

	argparser.add_argument('--num-runs', type=int, default=1)
	argparser.add_argument('--num-epochs', type=int, default=10)

	argparser.add_argument('--num-hidden', type=int, default=128)

	argparser.add_argument('--num-layers', type=int, default=2)
	# argparser.add_argument('--fan-out', type=str, default='2,4')
	argparser.add_argument('--fan-out', type=str, default='10,25')
	# argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--fan-out', type=str, default='10')



	argparser.add_argument('--log-indent', type=float, default=0)
#--------------------------------------------------------------------------------------

	argparser.add_argument('--lr', type=float, default=1e-2)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument("--weight-decay", type=float, default=5e-4,
						help="Weight for L2 loss")
	argparser.add_argument("--eval", action='store_true', 
						help='If not set, we will only do the training part.')

	argparser.add_argument('--num-workers', type=int, default=4,
		help="Number of sampling processes. Use 0 for no extra process.")
	

	args = argparser.parse_args()
	if args.setseed:
		set_seed(args)
	device = "cpu"
	if args.GPUmem:
		see_memory_usage("-----------------------------------------before load data ")
	if args.dataset=='karate':
		g, n_classes = load_karate()
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='cora':
		g, n_classes = load_cora()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='pubmed':
		g, n_classes = load_pubmed()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='reddit':
		g, n_classes = load_reddit()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
	elif args.dataset == 'ogbn-arxiv':
		data = load_ogbn_dataset(args.dataset,  args)
		device = "cuda:0"

	elif args.dataset=='ogbn-products':
		g, n_classes = load_ogb(args.dataset,args)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='ogbn-mag':
		# data = prepare_data_mag(device, args)
		data = load_ogbn_mag(args)
		device = "cuda:0"
		# run_mag(args, device, data)
		# return
	else:
		raise Exception('unknown dataset')


	best_test = run(args, device, data)


if __name__=='__main__':
	main()