import argparse

import dgl
# import dgl.nn as dglnn
# from gatconv import GATConv
from gat_model import GAT
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
import time
import random
import numpy as np
import sys
sys.path.insert(0,'../../pytorch/bucketing')
sys.path.insert(0,'../../pytorch/utils/')
sys.path.insert(0,'../../pytorch/micro_batch_train/')
sys.path.insert(0,'../../pytorch/models/')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing/GAT')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/micro_batch_train')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/utils')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/models')
from load_graph import load_reddit, inductive_split, load_cora, load_karate, prepare_data, load_pubmed
from load_graph import load_ogbn_dataset,load_ogb
from load_graph import load_karate_isolated_node
# from block_dataloader import generate_dataloader_block
from gat_bucketing_dataloader import generate_dataloader_bucket_block
from gat_bucketing_dataloader import dataloader_gen_bucketing_time
import pickle
sys.path.insert(0,'../../pytorch/utils/')
from memory_usage import see_memory_usage, nvidia_smi_usage
from torch.optim.lr_scheduler import StepLR

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
  
  



# def evaluate(g, features, labels, mask, model):
#     model.eval()
#     with torch.no_grad():
#         logits = model(g, features)
#         logits = logits[mask]
#         labels = labels[mask]
#         _, indices = torch.max(logits, dim=1)
#         correct = torch.sum(indices == labels)
#         return correct.item() * 1.0 / len(labels)


# def train(g, features, labels, masks, model):
#     # define train/val samples, loss function and optimizer
#     train_mask = masks[0]
#     val_mask = masks[1]
#     loss_fcn = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

#     # training loop
#     for epoch in range(200):
#         model.train()
#         logits = model(g, features)
#         loss = loss_fcn(logits[train_mask], labels[train_mask])
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         acc = evaluate(g, features, labels, val_mask, model)
#         print(
#             "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
#                 epoch, loss.item(), acc
#             )
#         )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--dataset",
#         type=str,
#         default="cora",
#         help="Dataset name ('cora', 'citeseer', 'pubmed').",
#     )
#     parser.add_argument(
#         "--dt",
#         type=str,
#         default="float",
#         help="data type(float, bfloat16)",
#     )
#     args = parser.parse_args()
#     print(f"Training with DGL built-in GATConv module.")

#     # load and preprocess dataset
#     transform = (
#         AddSelfLoop()
#     )  # by default, it will first remove self-loops to prevent duplication
#     if args.dataset == "cora":
#         data = CoraGraphDataset(transform=transform)
#     elif args.dataset == "citeseer":
#         data = CiteseerGraphDataset(transform=transform)
#     elif args.dataset == "pubmed":
#         data = PubmedGraphDataset(transform=transform)
#     else:
#         raise ValueError("Unknown dataset: {}".format(args.dataset))
#     g = data[0]
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     g = g.int().to(device)
#     features = g.ndata["feat"]
#     labels = g.ndata["label"]
#     masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

#     # create GAT model
#     in_size = features.shape[1]
#     out_size = data.num_classes
#     model = GAT(in_size, 8, out_size, heads=[8, 1]).to(device)

#     # convert model and graph to bfloat16 if needed
#     if args.dt == "bfloat16":
#         g = dgl.to_bfloat16(g)
#         features = features.to(dtype=torch.bfloat16)
#         model = model.to(dtype=torch.bfloat16)

#     # model training
#     print("Training...")
#     train(g, features, labels, masks, model)

#     # test the model
#     print("Testing...")
#     acc = evaluate(g, features, labels, masks[2], model)
#     print("Test accuracy {:.4f}".format(acc))
	
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
	batch_inputs = nfeat[blocks[0].srcdata[dgl.NID]].to(device)
	batch_labels = labels[blocks[-1].dstdata[dgl.NID]].to(device)
	return batch_inputs, batch_labels

def get_compute_num_nids(blocks):
	res=0
	for b in blocks:
		res+=len(b.srcdata['_ID'])
	return res

	
# def get_FL_output_num_nids(blocks):
# 	output_fl =len(blocks[0].dstdata['_ID'])
# 	return output_fl


    
def run(args, device, data):
    # Unpack data
	g, nfeats, labels, n_classes, train_nid, val_nid, test_nid = data
	in_feats = len(nfeats[0])
	print('in feats: ', in_feats)
	nvidia_smi_list=[]

	# if args.selection_method =='metis':
	# 	args.o_graph = dgl.node_subgraph(g, train_nid)


	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])
	full_batch_size = len(train_nid)
	
	fan_out_list = [fanout for fanout in args.fan_out.split(',')]
	fan_out_list = ' '.join(fan_out_list).split()
	processed_fan_out = [int(fanout) for fanout in fan_out_list] # remove empty string

	args.num_workers = 0
	full_batch_dataloader = dgl.dataloading.DataLoader(
		g,
		train_nid,
		sampler,
		# device='cpu',
		batch_size=full_batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)
	if args.num_batch == 1:
		args.batch_size = full_batch_size
	# if args.GPUmem:
		# see_memory_usage("----------------------------------------before model to device ")

	model = GAT(in_feats,args.aggre, args.num_hidden, n_classes, heads=[4, 1]).to(device)

	loss_fcn = nn.CrossEntropyLoss()

	dur = []
	pure_train_time_list=[]
	num_input_list =[]
	for run in range(args.num_runs):
		model.reset_parameters()
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
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		step_size = 50  # Adjust this as needed
		gamma = 0.5     # Adjust this as needed
		scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

		for epoch in range(args.num_epochs):
			# scheduler.step()
			if epoch > 60:
				scheduler.step()
			# if epoch >=250:
			# 	optimizer = torch.optim.Adam(model.parameters(), lr=0.00125, weight_decay=args.weight_decay)
			current_lr = optimizer.param_groups[0]['lr']
			print(f"Epoch {epoch + 1}, Learning Rate: {current_lr}")

			pure_train_time=0
			model.train()
			if epoch >= args.log_indent:
				t0 = time.time()
			loss_sum=0
			# start of data preprocessing part---s---------s--------s-------------s--------s------------s--------s----
			if args.load_full_batch:
				loading_start_time = time.time()#
				full_batch_dataloader=[]
				file_name=r'/home/cc/dataset/fan_out_'+args.fan_out+'/'+args.dataset+'_'+str(epoch)+'_items.pickle'
				with open(file_name, 'rb') as handle:
					item=pickle.load(handle)
					full_batch_dataloader.append(item)
				loading_end_time = time.time()#
				loading_time_list.append(loading_end_time-loading_start_time)#

			if epoch >= args.log_indent:
				t01 = time.time()
				# print('loading full batch data spends ', t01-t0)
			if args.num_batch > 1:
				features_load_time=0
				loader_s = time.time()#
				# block_dataloader, weights_list, time_collection = generate_dataloader_block(g, full_batch_dataloader, args)
				# b_block_dataloader, weights_list, time_collection = generate_dataloader_bucket_block(g, full_batch_dataloader, args)
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
					print('seeds ', seeds)
					print('input_nodes ', input_nodes)
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
					
					# batch_pred_labels = torch.argmax(batch_pred, dim=1) 
					# all_true_labels.extend(batch_labels.tolist())
					# all_predicted_labels.extend(batch_pred_labels.tolist())
					

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
				print('number of input ', num_input)
				see_memory_usage("----------------------------------------after optimizer")

				pure_train_time += (time_end-time13)
				pure_train_time_list.append(pure_train_time)
				print('----------------------------------------------------------pseudo_mini_loss sum ' + str(loss_sum.tolist()))
				print('pure train time : ', pure_train_time )
				train_e = time.time()#
				train_time = train_e-train_s#
				train_time_list.append(train_time)#
				feature_time_list.append(features_load_time)

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
			# if epoch >= args.log_indent:
			# 	dur.append(time.time() - t0)
			# # train_acc, val_acc, test_acc = evaluate(model, g, nfeats, labels, train_nid, val_nid, test_nid, device, args)
			# from sklearn.metrics import f1_score
			# # print("Run {:02d} | Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f}".format(run, epoch, loss_sum.item(), train_acc, val_acc, test_acc))
			# f1 = f1_score(all_true_labels, all_predicted_labels, average='micro')  # You can use 'macro', 'weighted', or None for different averaging options
			# print("Micro F1 Score:", f1)
	# print('Total (block generation + training)time/epoch {}'.format(np.mean(dur)))
	# print('pure train time/epoch {}'.format(np.mean(pure_train_time_list[4:])))
	# print('')
	# print('num_input_list ', num_input_list)
		# evaluate(model, g, nfeats, labels, train_nid, val_nid, test_nid, device, args)
				
			
	

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
	# argparser.add_argument('--dataset', type=str, default='pubmed')
	# argparser.add_argument('--dataset', type=str, default='cora')
	argparser.add_argument('--dataset', type=str, default='karate_iso')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	argparser.add_argument('--aggre', type=str, default='lstm')
	# argparser.add_argument('--aggre', type=str, default='sum')
	# argparser.add_argument('--selection-method', type=str, default='arxiv_25_backpack_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='cora_25_backpack_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='reddit_25_backpack_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='range_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='random_bucketing')
	argparser.add_argument('--selection-method', type=str, default='fanout_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='custom_bucketing')
	# argparser.add_argument('--selection-method', type=str, default='REG')
	argparser.add_argument('--num-batch', type=int, default=2)
	argparser.add_argument('--batch-size', type=int, default=0)
	argparser.add_argument('--mem-constraint', type=float, default=18)
	argparser.add_argument('--model', type=str, default='GAT')
	# # argparser.add_argument('--re-partition-method', type=str, default='random')
	# argparser.add_argument('--num-re-partition', type=int, default=0)

	# argparser.add_argument('--balanced_init_ratio', type=float, default=0.2)
	argparser.add_argument('--num-runs', type=int, default=1)
	argparser.add_argument('--num-epochs', type=int, default=2)

	argparser.add_argument('--num-hidden', type=int, default=256)
	# argparser.add_argument('--num-hidden', type=int, default=8)
	argparser.add_argument('--num-layers', type=int, default=2)
	argparser.add_argument('--fan-out', type=str, default='2,4')
	
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
	

	argparser.add_argument('--log-every', type=int, default=5)
	argparser.add_argument('--eval-every', type=int, default=5)
	
	args = argparser.parse_args()
	if args.setseed:
		set_seed(args)
	device = "cpu"
	# if args.GPUmem:
		# see_memory_usage("-----------------------------------------before load data ")
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
		g, n_classes = load_ogb(args.dataset, args)
		data=prepare_data(g, n_classes, args, device)
		device = "cuda:0"
	else:
		raise Exception('unknown dataset')
		
	
	best_test = run(args, device, data)
	

if __name__=='__main__':
	main()