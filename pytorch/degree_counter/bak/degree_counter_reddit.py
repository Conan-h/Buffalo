import sys
sys.path.insert(0,'../utils/')
sys.path.insert(0,'../models/')
import dgl
from dgl.data.utils import save_graphs
import numpy as np
from statistics import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
# from block_dataloader import generate_dataloader

# from block_dataloader import reconstruct_subgraph, reconstruct_subgraph_manually
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
# import deepspeed
import random
from graphsage_model import GraphSAGE
import dgl.function as fn
from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate, prepare_data, load_pubmed

from load_graph import load_ogbn_dataset,load_karate_isolated_node
from memory_usage import see_memory_usage, nvidia_smi_usage
import tracemalloc
from cpu_mem_usage import get_memory
from statistics import mean

from my_utils import parse_results
# from utils import draw_graph_global
# from draw_nx import draw_nx_graph

import pickle
from utils import Logger
import os 
import numpy
from torchsummary import summary





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


def save_full_batch(args, epoch,item):
    # /home/cc/Betty_baseline/pytorch/micro_batch_train
    # /home/cc/dataset/fan_out_10
	newpath = r'../../../dataset/fan_out_'+args.fan_out+'/'
	if not os.path.exists(newpath):
		os.makedirs(newpath)
	file_name=r'../../../dataset/fan_out_'+args.fan_out+'/'+args.dataset+'_'+str(epoch)+'_items.pickle'
	# cwd = os.getcwd() 
	with open(file_name, 'wb') as handle:
		pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print(' full batch blocks save')
	return




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
	# print('\t \t ===============   load_block_subtensor ============================\t ')
	# print('blocks[0].srcdata[dgl.NID]')
	# print(blocks[0].srcdata[dgl.NID])
	# print()
	# print('blocks[0].dstdata[dgl.NID]')
	# print(blocks[0].dstdata[dgl.NID])
	# print()
	# print('blocks[0].edata[dgl.EID]..........................')
	# print(blocks[0].edata[dgl.EID])
	# print()
	# print()
	# print('blocks[-1].srcdata[dgl.NID]')
	# print(blocks[-1].srcdata[dgl.NID])
	# print()
	# print('blocks[-1].dstdata[dgl.NID]')
	# print(blocks[-1].dstdata[dgl.NID])
	# print()
	
	# print('blocks[-1].edata[dgl.EID]..........................')
	# print(blocks[-1].edata[dgl.EID])
	# print()
	# batch_inputs_size = sys.getsizeof(nfeat[blocks[0].srcdata[dgl.NID]])
	# batch_labels_size = sys.getsizeof(labels[blocks[-1].dstdata[dgl.NID]])
	if args.GPUmem:
		see_memory_usage("----------------------------------------before batch input features to device")
	batch_inputs = nfeat[blocks[0].srcdata[dgl.NID]].to(device)
	if args.GPUmem:
		see_memory_usage("----------------------------------------after batch input features to device")
	batch_labels = labels[blocks[-1].dstdata[dgl.NID]].to(device)
	if args.GPUmem:
		see_memory_usage("----------------------------------------after batch labels to device")
	return batch_inputs, batch_labels
def get_compute_num_nids(blocks):
	res=0
	for b in blocks:
		res+=len(b.srcdata['_ID'])
	return res
	
#### Entry point
def run(args, device, data):
	if args.GPUmem:
		see_memory_usage("----------------------------------------start of run function ")
	# Unpack data
	g, nfeats, labels, n_classes, train_nid, val_nid, test_nid = data
	in_feats = len(nfeats[0])
	print('in feats: ', in_feats)

	fanout_list = [int(fanout) for fanout in args.fan_out.split(',')]
	sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout_list)
	full_batch_size = len(train_nid)
	A = g.in_degrees()
	counts = torch.bincount(A)
	print('counts ', counts)
	print(len(counts))
	counts = counts[1:]
	import matplotlib.pyplot as plt
	plt.figure(figsize=(4, 2))
	x_values = range(0,len(counts))
	plt.plot(x_values, counts, linestyle='-', color='b')
	plt.title('In-degree Frequency of Reddit')
	# plt.title('In-degree Frequency of '+args.dataset)
	plt.xlabel('in-degree', fontsize=14)
	plt.ylabel('Frequency', fontsize=14)
	x_ticks = range(0, len(counts), 5000)
	plt.xticks(x_ticks)
	plt.subplots_adjust(left=0.2, right=0.98, bottom=0.25) 
	# plt.grid(True)
	plt.savefig('./degree_Frequencies_'+args.dataset+'.pdf', format='pdf')

	plt.show()
	



	
def main():
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("single-gpu training")
	argparser.add_argument('--device', type=int, default=0,
		help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)
	argparser.add_argument('--setseed', type=bool, default=True)
	argparser.add_argument('--GPUmem', type=bool, default=True)
	argparser.add_argument('--load-full-batch', type=bool, default=False)
	argparser.add_argument('--gen-full-batch', type=bool, default=True)
	# argparser.add_argument('--load-full-batch', type=bool, default=True)
	# argparser.add_argument('--gen-full-batch', type=bool, default=False)
	
	# argparser.add_argument('--dataset', type=str, default='ogbn-arxiv')
	# argparser.add_argument('--dataset', type=str, default='ogbn-papers100M')
	# argparser.add_argument('--dataset', type=str, default='ogbn-products')
	# argparser.add_argument('--aggre', type=str, default='lstm')
	# argparser.add_argument('--dataset', type=str, default='cora')
	# argparser.add_argument('--dataset', type=str, default='pubmed')
	# argparser.add_argument('--dataset', type=str, default='karate')
	argparser.add_argument('--dataset', type=str, default='reddit')
	argparser.add_argument('--aggre', type=str, default='mean')
	
	argparser.add_argument('--balanced_init_ratio', type=float, default=0.2)
	argparser.add_argument('--num-runs', type=int, default=1)
	argparser.add_argument('--num-epochs', type=int, default=3)
	# argparser.add_argument('--num-runs', type=int, default=10)
	# argparser.add_argument('--num-epochs', type=int, default=500)
	argparser.add_argument('--num-hidden', type=int, default=6)

	# argparser.add_argument('--num-layers', type=int, default=3)
	# argparser.add_argument('--fan-out', type=str, default='10,25,50')
	
	# argparser.add_argument('--num-layers', type=int, default=2)
	# argparser.add_argument('--fan-out', type=str, default='10,25')
	
	argparser.add_argument('--num-layers', type=int, default=1)
	argparser.add_argument('--fan-out', type=str, default='10')

#---------------------------------------------------------------------------------------
	argparser.add_argument('--num_batch', type=int, default=1)
	# argparser.add_argument('--batch-size', type=int, default=2) # karate
	# argparser.add_argument('--batch-size', type=int, default=70) # cora
	# argparser.add_argument('--batch-size', type=int, default=30) # pubmed
	# argparser.add_argument('--batch-size', type=int, default=76716) # reddit
	argparser.add_argument('--batch-size', type=int, default=45471) # ogbn-arxiv

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
	elif args.dataset=='karate_iso':
		g, n_classes = load_karate_isolated_node()
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
	elif args.dataset=='ogbn-papers100M':
		# data = prepare_data_mag(device, args)
		g, n_classes = load_ogb(args.dataset, args)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
		
	else:
		raise Exception('unknown dataset')
	
	best_test = run(args, device, data)
	

if __name__=='__main__':
	main()

