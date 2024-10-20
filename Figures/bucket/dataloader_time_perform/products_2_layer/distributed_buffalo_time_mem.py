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
from graphsage_model_dist import GraphSAGE
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

def load_block_subtensor(nfeat, labels, blocks, device, args):
	"""
	Extracts features and labels for a subset of nodes
	"""
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
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	
	# Wrap the model for DataParallel 
	if torch.cuda.device_count() > 1: 
		print('torch.cuda.device_count() ', torch.cuda.device_count())
		ngpu= torch.cuda.device_count()
		devices = [torch.device('cuda:{}'.format(gpu)) for gpu in range(ngpu)]
		print('devices ', devices)
		model = model.to(devices[0])
		model = nn.parallel.replicate(model, devices)
		print('model_list ', model)
	
	
	# print('devices ', devices)
	loss_fcn = nn.CrossEntropyLoss()

	# if args.GPUmem:
	# 	see_memory_usage("----------------------------------------after model to device")
	logger = Logger(args.num_runs, args)
	num_input_list=[]
	pure_train_time_list =[]
	dur = []
	data_loader_gen_time_list=[]
	backpack_scheduling_time_list =[]
	block_gen_time_list =[]
	connection_check_time_list =[]
	for run in range(args.num_runs):
		
		# model.reset_parameters()  # Reset on a single instance  
		# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		
		for epoch in range(args.num_epochs):
			# model.train()
			if epoch >= args.log_indent:
				t0 = time.time()
			loss_sum=0
			# start of data preprocessing part---s---------s--------s-------------s--------s------------s--------s----
			if args.load_full_batch:
				full_batch_dataloader=[]
				file_name=r'/home/shuangyan/dataset/fan_out_'+args.fan_out+'/'+args.dataset+'_'+str(epoch)+'_items.pickle'
				with open(file_name, 'rb') as handle:
					item=pickle.load(handle)
					full_batch_dataloader.append(item)
			if epoch >= args.log_indent:
				print('load pickle file time ', time.time()-t0) 
			time_dataloader_start=time.time()
			if args.num_batch > 1:
				print("generate_dataloader_bucket_block=======")
				time_s = time.time()
				# b_block_dataloader, weights_list, time_collection = generate_dataloader_bucket_block(g, full_batch_dataloader, args)
				time_dataloader_start= time.time()
				# block_dataloader, weights_list = dataloader_gen_bucketing(full_batch_dataloader,g,processed_fan_out, args)
				block_dataloader, weights_list, backpack_schedule_time, connection_check_time, block_gen_time = \
						dataloader_gen_bucketing_time(full_batch_dataloader,g,processed_fan_out, args)
				time_dataloader_end= time.time()
				print('dataloader gen time ', time_dataloader_end-time_dataloader_start)
				backpack_scheduling_time_list.append(backpack_schedule_time)
				connection_check_time_list.append(connection_check_time)
				block_gen_time_list.append(block_gen_time)

				loader_gen_time = time_dataloader_end-time_dataloader_start
				if epoch >= args.log_indent:
					data_loader_gen_time_list.append(loader_gen_time)
				# connection_time, block_gen_time, _ = time_collection
				pure_train_time = 0
				time_start = time.time()
				num_input =0
				for step, (input_nodes, seeds, blocks) in enumerate(block_dataloader):
					print('step ', step )
					num_input += len(input_nodes)
					batch_inputs, batch_labels = load_block_subtensor(nfeats, labels, blocks, devices[step%2], args)#------------*
					blocks = [block.int().to(devices[step%2]) for block in blocks]#------------*
					print('blocks', blocks)
					time11= time.time()
					see_memory_usage("----------------------------------------before batch_pred = model(blocks, batch_inputs)")
					# batch_pred = nn.parallel.parallel_apply(model, batch_inputs)
					batch_pred = model[step%2](blocks, batch_inputs)#------------*
					see_memory_usage("----------------------------------------after batch_pred = model(blocks, batch_inputs)")
					pseudo_mini_loss = loss_fcn(batch_pred, batch_labels)#------------*
					
					see_memory_usage("----------------------------------------after loss function")
					pseudo_mini_loss = pseudo_mini_loss*weights_list[step]#------------*
					pseudo_mini_loss.backward()#------------*
					time12= time.time()
					pure_train_time += (time12-time11)
					pseudo_mini_loss = pseudo_mini_loss.to(devices[0])  # 假设主设备是 cuda:0  
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
				print('pure train time : ', pure_train_time )
				print('train time : ', time_end-time_start )
				print('end to end time : ', time_end-time_s )
				# print('connection check time: ', connection_time)
				# print('block generation time ', block_gen_time)
    

			elif args.num_batch == 1: # full batch train can't use nn.parallel, as one block is the basic unit
				# print('orignal labels: ', labels)
				for step, (input_nodes, seeds, blocks) in enumerate(full_batch_dataloader):
					# print()
					print('full batch src global ', len(input_nodes))
					print('full batch dst global ', len(seeds))
					# print('full batch eid global ', blocks[-1].edata['_ID'])
					batch_inputs, batch_labels = load_block_subtensor(nfeats, labels, blocks, device, args)#------------*
					see_memory_usage("----------------------------------------after load_block_subtensor")
					blocks = [block.int().to(device) for block in blocks]
					see_memory_usage("----------------------------------------after block to device")
					# batch_pred = nn.parallel.parallel_apply(model, batch_inputs)
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
			if epoch >= args.log_indent:
			
				full_epoch=time.time() - t0
				print('end to end time ', full_epoch)
				dur.append(full_epoch)
		print('Total (block generation + training)time/epoch {}'.format(np.mean(dur[3:])))
		print('Total (block generation + training)time/epoch all {}'.format(dur))
		print('pure train time per /epoch ', pure_train_time_list)
		print('pure train time average ', np.mean(pure_train_time_list[3:]))
		print('dataloader time ', data_loader_gen_time_list)
		print('dataloader time avg per epoch {}'.format(np.mean(data_loader_gen_time_list[4:])))
		print('')
		print('input num list ', num_input_list)

		print('      backpack schedule time ', backpack_scheduling_time_list)
		print('------backpack schedule time avg', np.mean(backpack_scheduling_time_list[4:]))
		print('		connection_check_time_list ',connection_check_time_list)
		print('------connection_check time avg', np.mean(connection_check_time_list[4:]))
		print('		block_gen_time_list ', block_gen_time_list) 
		print('------block gen time avg', np.mean(block_gen_time_list[4:]))


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
	# argparser.add_argument('--selection-method', type=str, default='range_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='random_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='fanout_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='custom_bucketing')
	argparser.add_argument('--num-batch', type=int, default=2) ############=====================
	argparser.add_argument('--mem-constraint', type=float, default=70)
	argparser.add_argument('--cluster-coeff', type=float, default=0.411)
	argparser.add_argument('--num-runs', type=int, default=1)
	argparser.add_argument('--num-epochs', type=int, default=1)

	argparser.add_argument('--num-hidden', type=int, default=128)

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