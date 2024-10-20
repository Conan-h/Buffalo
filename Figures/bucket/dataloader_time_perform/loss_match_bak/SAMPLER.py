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

	# sampler = dgl.dataloading.MultiLayerNeighborSampler(processed_fan_out)
	# sampler = dgl.dataloading.MultiLayerNeighborSampler(
	# 	[int(fanout) for fanout in args.fan_out.split(',')])
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
	data_loader_bucketing_time_list=[]
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
			
			if args.num_batch > 1:
				# print("generate_dataloader_bucket_block=======")
				# time_s = time.time()
				# b_block_dataloader, weights_list, time_collection = generate_dataloader_bucket_block(g, full_batch_dataloader, args)

				# data_loader_bucketing_time_list.append(time.time()-time_s)
				# print(' current epoch dataloader time ', time.time()-time_s)
				# connection_time, block_gen_time, _ = time_collection
				# print('connection check time: ', connection_time)
				# print('block generation time ', block_gen_time)
				# print('')
				# pure_train_time = 0
				# time_start = time.time()
				# num_input =0
				# for step, (input_nodes, seeds, blocks) in enumerate(b_block_dataloader):
				# 	print(' micro batch ', step )
				# 	for layer, bi in enumerate(reversed(blocks)):
				# 		if layer == 0:
				# 			print('the output layer ')
				# 			gi = dgl.block_to_graph(bi)
				# 			print('the output layer graph ')
				# 			print(gi)
				block_dataloader=[]
				weights_list = []
				for step , (src, dst, full_blocks) in enumerate(full_batch_dataloader):
					print(' micro batch ', step )
					print('src ', src)
					print('dst ', dst)
					ds_list = [[2,1],[0,3]]
					ds_list = [[2,0,1,3]]
					# ds_list = [[2,0],[1,3]]
					for i,dst_new in enumerate(ds_list) :
						weights_list.append(len(dst_new)/len(dst))
						block_list=[]
						for layer , full_block in enumerate(reversed(full_blocks)):
							if layer == 0:
								print('the output layer ')
								print('output layer fanout ', processed_fan_out[-1])
								
								print('full_block ', full_block)
								layer_graph = dgl.edge_subgraph(g, full_block.edata['_ID'],relabel_nodes=False,store_ids=True)
								print('layer_graph.edges() before remove isolated nodes', layer_graph.edges())
								print('layer_graph.nodes() ', layer_graph.nodes())
								# print('layer_graph.ndata[_ID] ', layer_graph.ndata['_ID'])
								print('layer_graph.edata[_ID] ', layer_graph.edata['_ID'])
								
								print('layer_graph ', layer_graph)
								print('layer_graph.edges ', layer_graph.edges())
								
								dst_len = len(full_block.srcdata['_ID'])
								layer_graph.ndata['_ID']=torch.tensor([-1]*len(layer_graph.ndata['train_mask']))
								layer_graph.ndata['_ID'][:dst_len] = full_block.srcdata['_ID']
								print('layer_graph.nodes ', layer_graph.ndata)
								
								sg1 = dgl.sampling.sample_neighbors_range(layer_graph, dst_new, processed_fan_out[-1])
								block = dgl.to_block(sg1,dst_new, include_dst_in_src= True)
								
								print(block)
								print(block.srcdata['_ID'])
								# print(block.ndata['_ID'])
								print('block.edges() ', block.edges())
								# print(block.edata[])
								print('block.srcdata[dgl.NID] ', block.srcdata[dgl.NID])
								print('block.dstdata[dgl.NID] ',block.dstdata[dgl.NID])
								dst_new = block.srcdata[dgl.NID]
								block_list.append(block)
							if layer == 1:
								print('input layer')
								print('full_block ', full_block)
								print('full_block.edata[_ID]', full_block.edata['_ID'])
								layer_graph = dgl.edge_subgraph(g, full_block.edata['_ID'],relabel_nodes=False,store_ids=True)
								dst_len = len(full_block.srcdata['_ID'])
								layer_graph.ndata['_ID']=torch.tensor([-1]*len(layer_graph.ndata['train_mask']))
								layer_graph.ndata['_ID'][:dst_len] = full_block.srcdata['_ID']
								print('layer_graph.edges() ', layer_graph.edata)
								print('layer_graph.edges() ', layer_graph.edges())
								
								sg1 = dgl.sampling.sample_neighbors_range(layer_graph, dst_new, processed_fan_out[0])
								print('sg1.nodes',sg1.nodes())
								print('sg1.edges',sg1.edges())
								block = dgl.to_block(sg1,dst_new, include_dst_in_src= True)
								
								print('new block ')
								print(block)
								print(block.ndata['_ID'])
								# print(block.edges())
								# print(block.edata)
								print('block.srcdata[dgl.NID] ', block.srcdata[dgl.NID])
								print('block.dstdata[dgl.NID] ', block.dstdata[dgl.NID])
								block_list.insert(0,block)
						block_dataloader.append((block.srcdata[dgl.NID], ds_list[i], block_list))

				pseudo_mini_loss = torch.tensor([], dtype=torch.long)
				num_input_nids=0
				pure_train_time=0
				print('weights_list', weights_list)
				for step, (input_nodes, seeds, blocks) in enumerate(block_dataloader):
					print('step ', step)
					print('input_nodes', input_nodes)
					print('seeds', seeds)
					print('blocks', blocks)
					num_input_nids	+= len(input_nodes)
					batch_inputs, batch_labels = load_block_subtensor(nfeats, labels, blocks, device,args)#------------*
					print('batch_inputs ', batch_inputs)
					print('batch_labels ', batch_labels)
					blocks = [block.int().to(device) for block in blocks]#------------*
					t1= time.time()
					batch_pred = model(blocks, batch_inputs)#------------*
					see_memory_usage("----------------------------------------after batch_pred = model(blocks, batch_inputs)")
						
					loss = loss_fcn(batch_pred, batch_labels)#------------*
					see_memory_usage("----------------------------------------after loss function")
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
		print('pure train time/epoch {}'.format(np.mean(pure_train_time_list[4:])))
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
	# argparser.add_argument('--dataset', type=str, default='ogbn-products')
	# argparser.add_argument('--dataset', type=str, default='cora')
	argparser.add_argument('--dataset', type=str, default='karate')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	# argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--aggre', type=str, default='lstm')
	argparser.add_argument('--model', type=str, default='SAGE')
	# argparser.add_argument('--selection-method', type=str, default='arxiv_backpack_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='products_10_backpack_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='products_25_backpack_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='range_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='random_bucketing')
	argparser.add_argument('--selection-method', type=str, default='fanout_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='custom_bucketing')
	argparser.add_argument('--num-batch', type=int, default=3)
	argparser.add_argument('--mem-constraint', type=float, default=18)

	argparser.add_argument('--num-runs', type=int, default=1)
	argparser.add_argument('--num-epochs', type=int, default=1)

	argparser.add_argument('--num-hidden', type=int, default=256)

	argparser.add_argument('--num-layers', type=int, default=2)
	argparser.add_argument('--fan-out', type=str, default='2,4')
	# argparser.add_argument('--fan-out', type=str, default='10,25')
	# argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--fan-out', type=str, default='10')



	argparser.add_argument('--log-indent', type=float, default=0)
#--------------------------------------------------------------------------------------

	argparser.add_argument('--lr', type=float, default=1e-3)
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