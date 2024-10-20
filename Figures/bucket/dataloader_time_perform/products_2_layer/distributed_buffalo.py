import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../../')
sys.path.insert(0,'../../pytorch/utils')
sys.path.insert(0,'../../pytorch/bucketing')
sys.path.insert(0,'../../pytorch/models')
sys.path.insert(0,'../../memory_logging')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/utils')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/models')
sys.path.insert(0,'/home/shuangyan/Betty_baseline/pytorch/bucketing')
sys.path.insert(0,'/home/shuangyan/Betty_baseline/pytorch/utils')
sys.path.insert(0,'/home/shuangyan/Betty_baseline/pytorch/models')
from bucketing_dataloader import generate_dataloader_bucket_block
from bucketing_dataloader import dataloader_gen_bucketing
from bucketing_dataloader import dataloader_gen_bucketing_time
# from runtime_nvidia_smi import start_memory_logging, stop_memory_logging


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
from graphsage_model_dist import DistSAGE
import dgl.function as fn
from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate, prepare_data, load_pubmed

from load_graph import load_ogbn_dataset
from memory_usage import see_memory_usage, nvidia_smi_usage

import pickle

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

def load_block_subtensor(nfeat, labels, blocks, device, args):
	"""
	Extracts features and labels for a subset of nodes
	"""
	print('enter function load_block_subtensor: device ', device)
	# batch_inputs = [torch.randn(bz, n_hidden).to(dev) for dev in devices]
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
	in_feats = len(nfeats[0])
	# print('in feats: ', in_feats)
	nvidia_smi_list=[]

	if args.selection_method =='metis':
		args.o_graph = dgl.node_subgraph(g, train_nid)


	# sampler = dgl.dataloading.MultiLayerNeighborSampler(
	# 	[int(fanout) for fanout in args.fan_out.split(',')])
	# full_batch_size = len(train_nid)
	fan_out_list = [fanout for fanout in args.fan_out.split(',')]
	fan_out_list = ' '.join(fan_out_list).split()
	processed_fan_out = [int(fanout) for fanout in fan_out_list] # remove empty string

	args.num_workers = 0

	model = DistSAGE(
					in_feats,
					args.num_hidden,
					n_classes,
					args.aggre,
					args.num_layers,
					F.relu,
					args.dropout).to(device)
	torch.distributed.init_process_group(backend='nccl')  
	# 获取当前进程的 rank  
	rank = torch.distributed.get_rank()  
	world_size = torch.distributed.get_world_size()  # 获取总进程数  
	device = torch.device("cuda:{}".format(rank))
 
	model=torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)
	

	# print('device ', device)
	loss_fcn = nn.CrossEntropyLoss()
	
	
 
	# print(f"Available CUDA devices: {torch.cuda.device_count()}") 
	for run in range(args.num_runs):
		
		# model.reset_parameters()  # Reset on a single instance  
		# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		for epoch in range(args.num_epochs):
			
			if args.num_batch > 1:
				print("generate_dataloader_bucket_block=======")
				# block_dataloader, weights_list, backpack_schedule_time, connection_check_time, block_gen_time = \
				# 		dataloader_gen_bucketing_time(batch_dataloader,g,processed_fan_out, args)
				prefix = '/home/shuangyan/dataset/microbatch/nb_'+str(args.num_batch)+'/fan_out_'+args.fan_out+'/'+args.dataset+'_'
				nb_per_GPU= args.num_batch//2
				block_dataloader=[]
				for i in range(nb_per_GPU):
					print('i',i)
					print('i+rank*nb_per_GPU',i+rank*nb_per_GPU)
					with open(prefix+str(i+rank*nb_per_GPU)+'_micro_batch_block_dataloader.pkl', 'rb') as f:  
						block_dataloader.append(pickle.load(f)) 
				print('block_dataloader ',block_dataloader)
				print('len(block_dataloader) ',len(block_dataloader))

				loss_sum=0
				with model.join():
					for step, [(input_nodes, seeds, blocks)] in enumerate(block_dataloader):
						device = torch.device("cuda:{}".format(rank))
						print('before load_block_subtensor device : ',device)
						batch_inputs, batch_labels = load_block_subtensor(nfeats, labels, blocks, 'cpu', args)#------------*
						blocks = [block.int().to(device) for block in blocks]#------------*
						# Forward pass  
						batch_inputs = batch_inputs.to(device)
						batch_labels = batch_labels.to(device)
						batch_pred = model(blocks, batch_inputs)  # Ensure model is on the correct device  

						# Calculate loss  
						pseudo_mini_loss = loss_fcn(batch_pred, batch_labels)  
						pseudo_mini_loss = pseudo_mini_loss * len(seeds)/len(train_nid) 
						pseudo_mini_loss.backward()  

						loss_sum += pseudo_mini_loss.item()  # Use .item() for accumulating loss  

					# Optimizer step  
					optimizer.step()  
					optimizer.zero_grad()  
									
					torch.distributed.barrier()
					print('----------------------------------------------------------pseudo_mini_loss sum ' + str(loss_sum.tolist()))
					
			

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
	# argparser.add_argument('--selection-method', type=str, default='arxiv_backpack_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='reddit_10_backpack_bucketing')
	argparser.add_argument('--selection-method', type=str, default='products_25_backpack_bucketing')

	argparser.add_argument('--num-batch', type=int, default=4) ############=====================
	argparser.add_argument('--mem-constraint', type=float, default=70)
	argparser.add_argument('--cluster-coeff', type=float, default=0.411)
	argparser.add_argument('--num-runs', type=int, default=1)
	argparser.add_argument('--num-epochs', type=int, default=1)

	argparser.add_argument('--num-hidden', type=int, default=16)

	# argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--fan-out', type=str, default='10')

	argparser.add_argument('--num-layers', type=int, default=2)
	argparser.add_argument('--fan-out', type=str, default='10,25')
	# argparser.add_argument('--num-layers', type=int, default=3)
	# argparser.add_argument('--fan-out', type=str, default='10,25,30')

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