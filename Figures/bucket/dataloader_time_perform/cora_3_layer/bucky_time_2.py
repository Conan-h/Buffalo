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
	in_feats = len(nfeats[0])
	# print('in feats: ', in_feats)
	nvidia_smi_list=[]

	if args.selection_method =='metis':
		args.o_graph = dgl.node_subgraph(g, train_nid)


	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])
	full_batch_size = len(train_nid)
	fan_out_list = [fanout for fanout in args.fan_out.split(',')]
	fan_out_list = ' '.join(fan_out_list).split()
	processed_fan_out = [int(fanout) for fanout in fan_out_list] # remove empty string


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


	for run in range(args.num_runs):
		model.reset_parameters()
		# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		epoch_time_list=[]
		loader_gen_time_list=[]
		loading_time_list=[]
		train_time_list =[]
		feature_time_list=[]
		backpack_time_list =[]
		connection_check_time_list=[]
		block_gen_time_list=[]
		num_input_list =[]
		pure_train_time_list=[]
		for epoch in range(args.num_epochs):
			model.train()
			epoch_start_time = time.time()#
			loss_sum=0 #
			full_batch_dataloader=[]
			# start of data preprocessing part---s---------s--------s-------------s--------s------------s--------s----
			if args.load_full_batch:
				loading_start_time = time.time()#

				file_name=r'/home/cc/dataset/fan_out_'+args.fan_out+'/'+args.dataset+'_'+str(epoch)+'_items.pickle'
				with open(file_name, 'rb') as handle:
					item=pickle.load(handle)
					full_batch_dataloader.append(item)

				loading_end_time = time.time()#
				loading_time_list.append(loading_end_time-loading_start_time)#

			if args.num_batch > 1:
				features_load_time=0
				loader_s = time.time()#
				# b_block_dataloader, weights_list, time_collection = generate_dataloader_bucket_block(g, full_batch_dataloader, args)
				
				# block_dataloader, weights_list = dataloader_gen_bucketing(full_batch_dataloader,g,processed_fan_out, args)
				block_dataloader, weights_list, backpack_schedule_time, connection_check_time, block_gen_time = \
						dataloader_gen_bucketing_time(full_batch_dataloader,g,processed_fan_out, args)
				loader_e = time.time()#
				loader_gen_time = loader_e-loader_s#
				loader_gen_time_list.append(loader_gen_time)#
				backpack_time_list.append(backpack_schedule_time)#
				connection_check_time_list.append(connection_check_time)#
				block_gen_time_list.append(block_gen_time)#
				train_s = time.time()#
				num_input =0
				pure_train_time = 0
				for step, (input_nodes, seeds, blocks) in enumerate(block_dataloader):
					print('step ', step )
					num_input += len(input_nodes)
					f_s= time.time()#

					batch_inputs, batch_labels = load_block_subtensor(nfeats, labels, blocks, device,args)#------------*
					blocks = [block.int().to(device) for block in blocks]#------------*

					f_e = time.time()#
					features_load_time += (f_e-f_s)#
					see_memory_usage("----------------------------------------before batch_pred = model(blocks, batch_inputs)")
					time11 = time.time()
					batch_pred = model(blocks, batch_inputs)#------------*
					see_memory_usage("----------------------------------------after batch_pred = model(blocks, batch_inputs)")
					pseudo_mini_loss = loss_fcn(batch_pred, batch_labels)#------------*
					
					see_memory_usage("----------------------------------------after loss function")
					pseudo_mini_loss = pseudo_mini_loss*weights_list[step]#------------*
					pseudo_mini_loss.backward()#------------*
					time12= time.time()
					pure_train_time += (time12-time11)
					loss_sum += pseudo_mini_loss#------------*
					
					
				time13= time.time()
				optimizer.step()
				optimizer.zero_grad()
				time_end = time.time()
    
				num_input_list.append(num_input)
				see_memory_usage("----------------------------------------after optimizer")

				pure_train_time += (time_end-time13)
				pure_train_time_list.append(pure_train_time)
				print('----------------------------------------------------------pseudo_mini_loss sum ' + str(loss_sum.tolist()))
				# print('pure train time : ', pure_train_time )
				train_e = time.time()#
				train_time = train_e-train_s#
				train_time_list.append(train_time)#
				feature_time_list.append(features_load_time)#
			

			elif args.num_batch == 1:
				# print('orignal labels: ', labels)
				for step, (input_nodes, seeds, blocks) in enumerate(full_batch_dataloader):
					# print()
					print('full batch src global ', len(input_nodes))
					print('full batch dst global ', len(seeds))
					# print('full batch eid global ', blocks[-1].edata['_ID'])
					batch_inputs, batch_labels = load_block_subtensor(nfeats, labels, blocks, device,args)#------------*
					see_memory_usage("----------------------------------------after load_block_subtensor")
					blocks = [block.int().to(device) for block in blocks]
					see_memory_usage("----------------------------------------after block to device")

					batch_pred = model(blocks, batch_inputs)
					see_memory_usage("----------------------------------------after model")

					loss = loss_fcn(batch_pred, batch_labels)
					print('full batch train ------ loss ' + str(loss.item()) )
					see_memory_usage("----------------------------------------after loss")

					loss.backward()
					see_memory_usage("----------------------------------------after loss backward")

					optimizer.step()
					optimizer.zero_grad()
					print()
					see_memory_usage("----------------------------------------full batch")
			epoch_end_time = time.time()
			epoch_time_list.append(epoch_end_time-epoch_start_time)
		print('epoch_time_list ', epoch_time_list)

		print()
		print('loading_time list  ', loading_time_list)
		print()
		if args.num_batch>1:
			print(' data loader gen time ', loader_gen_time_list)
			print('	---backpack schedule time ', backpack_time_list)
			print('	---connection_check_time_list ',connection_check_time_list)
			print('	---block_gen_time_list ', block_gen_time_list)
		print('training time ',train_time_list)
		print('---feature block loading time ', feature_time_list)
		print()
		print()
		print('epoch_time avg  ', np.mean(epoch_time_list[4:]))
		print('loading_time avg  ', np.mean(loading_time_list[4:]))
		if args.num_batch>1:
			
			print(' data loader gen time avg', np.mean(loader_gen_time_list[4:]))
			print('	---backpack schedule time avg', np.mean(backpack_time_list[4:]))
			print('	---connection_check_time avg ',np.mean(connection_check_time_list[4:]))
			print('	---block_gen_time avg ', np.mean(block_gen_time_list[4:])) 

		print('training time ', np.mean(train_time_list[4:]))
		print('---feature block loading time ', np.mean(feature_time_list[4:]))
		

		print('pure train time per /epoch ', pure_train_time_list)
		print('pure train time average ', np.mean(pure_train_time_list[3:]))

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
	argparser.add_argument('--dataset', type=str, default='cora')
	# argparser.add_argument('--dataset', type=str, default='karate')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	# argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--aggre', type=str, default='lstm')
	argparser.add_argument('--model', type=str, default='SAGE')
	# argparser.add_argument('--selection-method', type=str, default='arxiv_backpack_bucketing')
	argparser.add_argument('--selection-method', type=str, default='cora_30_backpack_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='products_25_backpack_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='range_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='random_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='fanout_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='custom_bucketing')
	argparser.add_argument('--num-batch', type=int, default=2)
	argparser.add_argument('--mem-constraint', type=float, default=7.5)
	argparser.add_argument('--cluster-coeff', type=float, default=0.24)
	argparser.add_argument('--num-runs', type=int, default=1)
	argparser.add_argument('--num-epochs', type=int, default=10)

	argparser.add_argument('--num-hidden', type=int, default=2048)


	argparser.add_argument('--num-layers', type=int, default=3)
	argparser.add_argument('--fan-out', type=str, default='10,25,30')



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