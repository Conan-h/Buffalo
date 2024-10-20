import sys
sys.path.insert(0,'..')
sys.path.insert(0,'..')
sys.path.insert(0,'../../pytorch/utils/')
sys.path.insert(0,'../../pytorch/bucketing/')
sys.path.insert(0,'../../pytorch/models/')
import dgl
from dgl.data.utils import save_graphs
import numpy as np
from statistics import mean
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from bucketing_dataloader import generate_dataloader_bucket_block

import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm

import random
from graphsage_model_wo_mem import GraphSAGE
import dgl.function as fn
from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate, prepare_data, load_pubmed

from load_graph import load_ogbn_dataset
from memory_usage import see_memory_usage, nvidia_smi_usage
import tracemalloc
from cpu_mem_usage import get_memory
from statistics import mean

from my_utils import parse_results


import pickle
from utils import Logger
import os 
import numpy




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
	# print('input global nids ', blocks[0].srcdata[dgl.NID])
	# print('input features: ', batch_inputs)
	# print('seeds global nids ', blocks[-1].dstdata[dgl.NID])
	# print('seeds labels : ',batch_labels)
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
	print('in feats: ', in_feats)
	nvidia_smi_list=[]

	if args.selection_method =='metis':
		args.o_graph = dgl.node_subgraph(g, train_nid)


	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])
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
	dur = []
	time_block_gen=[]
	# if args.GPUmem:
	# 			see_memory_usage("----------------------------------------after model to device")
	logger = Logger(args.num_runs, args)
	for run in range(args.num_runs):
		model.reset_parameters()
		# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		for epoch in range(args.num_epochs):
			print()
			print('=-=-'*30 )
			print('Epoch: ',epoch )
			num_src_node =0
			num_out_node_FL=0
			gen_block=0
			tmp_t = 0
			model.train()
			if epoch >= args.log_indent:
				t0 = time.time()
			loss_sum=0
			# start of data preprocessing part---s---------s--------s-------------s--------s------------s--------s----
			if args.load_full_batch:
				full_batch_dataloader=[]
				file_name=r'/home/cc/Betty_baseline/dataset/fan_out_'+args.fan_out+'/'+args.dataset+'_'+str(epoch)+'_items.pickle'
				with open(file_name, 'rb') as handle:
					item=pickle.load(handle)
					full_batch_dataloader.append(item)

			if args.num_batch > 1:
				b_block_dataloader, weights_list, time_collection = generate_dataloader_bucket_block(g, full_batch_dataloader, args)
				connect_check_time, block_gen_time_total, batch_blocks_gen_time =time_collection
				print('connection checking time: ', connect_check_time)
				print('block generation total time ', block_gen_time_total)
				# print('average batch blocks generation time: ', batch_blocks_gen_time)
				if epoch >= args.log_indent:
					gen_block=time.time() - t0
					time_block_gen.append(time.time() - t0)
					print('block dataloader generation time/epoch {}'.format(np.mean(time_block_gen)))
					tmp_t=time.time()

				data_loading_t=[]
				block_to_t=[]
				modeling_t=[]
				loss_cal_t=[]
				backward_t=[]
				data_size_transfer=[]
				blocks_size=[]
				num_input_nids = []
				num_output_nids= []
				for step, (input_nodes, seeds, blocks) in enumerate(b_block_dataloader):
					num_input_nids.append(len(input_nodes)) 
					num_output_nids.append(len(seeds))
					tt1=time.time()
					batch_inputs, batch_labels = load_block_subtensor(nfeats, labels, blocks, device,args)#------------*
					tt2=time.time()
					data_loading_t.append(tt2-tt1)

					tt51=time.time()
					blocks = [block.int().to(device) for block in blocks]#------------*
					tt5=time.time()
					block_to_t.append(tt5-tt51)
		
					# Compute loss and prediction
					tt3=time.time()
					batch_pred = model(blocks, batch_inputs)#------------*
					tt4=time.time()
					modeling_t.append(tt4-tt3)

					tt61=time.time()
					pseudo_mini_loss = loss_fcn(batch_pred, batch_labels)#------------*
					pseudo_mini_loss = pseudo_mini_loss*weights_list[step]#------------*
					tt6=time.time()
					loss_cal_t.append(tt6-tt61)

					pseudo_mini_loss.backward()#------------*
					loss_sum += pseudo_mini_loss#------------*
					tt8=time.time()
					backward_t.append(tt8-tt6)
    
				tte=time.time()
				optimizer.step()
				optimizer.zero_grad()
				ttend=time.time()
				see_memory_usage("----------------------------------------finish all batches")
				print('Number of first layer input nodes during this epoch each batch: ', num_input_nids)
				print('Number of first layer input nodes during this epoch total: ', sum(num_input_nids))
				print('Number of last layer output nodes during this epoch each batch: ', num_output_nids)
				print('Number of last layer output nodes during this epoch total: ', sum(num_output_nids))
				print('----------------------------------------------------------pseudo_mini_loss sum ' + str(loss_sum.tolist()))
				print()

			elif args.num_batch == 1:
				# print('orignal labels: ', labels)
				for step, (input_nodes, seeds, blocks) in enumerate(full_batch_dataloader):
					# print()
					# print('full batch src global ', input_nodes)
					print('full batch dst global ', seeds)
					# print('full batch eid global ', blocks[-1].edata['_ID'])
					batch_inputs, batch_labels = load_block_subtensor(nfeats, labels, blocks, device,args)#------------*
					print('batch_labels ')
					# print(batch_labels)
					print('blocks')
					# print(blocks[0].edata['_ID'])
					# print(blocks[-1].edata['_ID'])
					blocks = [block.int().to(device) for block in blocks]
					batch_pred = model(blocks, batch_inputs)
					loss = loss_fcn(batch_pred, batch_labels)
					print('full batch train ------ loss ' + str(loss.item()) )
				
					loss.backward()
					optimizer.step()
					optimizer.zero_grad()
					see_memory_usage("----------------------------------------full batch")
					print()
			if epoch >= args.log_indent:
				tmp_t2=time.time()
				full_epoch=time.time() - t0
				dur.append(full_epoch)
				print('Total (block generation + training)time/epoch {} sec'.format(np.mean(dur)))
				
				print('Training time/epoch {} sec (without bucket block generation)'.format(tmp_t2-tmp_t))
				print('Training time without block to device /epoch {} sec'.format(tmp_t2-tmp_t-sum(block_to_t)))
				print('Pure Training time without any dataloading part /epoch {} sec'.format(sum(modeling_t)+sum(loss_cal_t)+sum(backward_t)+ttend-tte))
				print('load block tensor time/epoch {} sec'.format(np.sum(data_loading_t)))
				print('block to device time/epoch {} sec'.format(np.sum(block_to_t)))
				# print('input features size transfer per epoch {}'.format(np.sum(data_size_transfer)/1024/1024/1024))
				# print('blocks size to device per epoch {}'.format(np.sum(blocks_size)/1024/1024/1024))

				print()
					

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
	argparser.add_argument('--dataset', type=str, default='ogbn-arxiv')
	# argparser.add_argument('--dataset', type=str, default='ogbn-mag')
	# argparser.add_argument('--dataset', type=str, default='ogbn-products')
	# argparser.add_argument('--dataset', type=str, default='cora')
	# argparser.add_argument('--dataset', type=str, default='karate')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	# argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--aggre', type=str, default='lstm')

	argparser.add_argument('--selection-method', type=str, default='random_bucketing')

	argparser.add_argument('--num-batch', type=int, default=3)


	argparser.add_argument('--num-runs', type=int, default=1)
	argparser.add_argument('--num-epochs', type=int, default=10)

	argparser.add_argument('--num-hidden', type=int, default=128)

	# argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--fan-out', type=str, default='10')

	argparser.add_argument('--num-layers', type=int, default=2)
	argparser.add_argument('--fan-out', type=str, default='10,25')
	
	# argparser.add_argument('--num-layers', type=int, default=3)
	# argparser.add_argument('--fan-out', type=str, default='10,25,30')



	# argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--fan-out', type=str, default='4')
	# argparser.add_argument('--num-layers', type=int, default=2)
	# argparser.add_argument('--fan-out', type=str, default='2,4')


	argparser.add_argument('--log-indent', type=float, default=3)
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
