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
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim  
import argparse  
import numpy as np  

import pickle
from bucketing_dataloader import dataloader_gen_bucketing_time  
from graphsage_model_wo_mem import GraphSAGE  
from load_graph import prepare_data, load_ogb, load_reddit, load_cora, load_karate, load_pubmed, load_ogbn_dataset  

def set_seed(args):  
    torch.manual_seed(args.seed)  
    if args.device >= 0:  
        torch.cuda.manual_seed_all(args.seed)  
        torch.backends.cudnn.enabled = False  
        torch.backends.cudnn.deterministic = True  

def load_data(args):  
    if args.dataset == 'karate':  
        g, n_classes = load_karate()  
    elif args.dataset == 'cora':  
        g, n_classes = load_cora()  
    elif args.dataset == 'pubmed':  
        g, n_classes = load_pubmed()  
    elif args.dataset == 'reddit':  
        g, n_classes = load_reddit()  
    elif args.dataset == 'ogbn-products':  
        g, n_classes = load_ogb(args.dataset, args)  
    else:  
        g, n_classes = load_ogbn_dataset(args.dataset, args)  
    
    return g, n_classes  
def load_subtensor(nfeat, labels, seeds, input_nodes, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = nfeat[input_nodes].to(device)
	batch_labels = labels[seeds].to(device)
	return batch_inputs, batch_labels

def run(args, device, data):  
    g, nfeats, labels, n_classes, train_nid, val_nid, test_nid = data  
    in_feats = len(nfeats[0])  

    model = GraphSAGE(in_feats, args.num_hidden, n_classes, args.aggre, args.num_layers, F.relu, args.dropout)  
    model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model  
    model.to(device)  

    loss_fcn = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  

    for run in range(args.num_runs):  
        model.module.reset_parameters() if isinstance(model, nn.DataParallel) else model.reset_parameters()  

        for epoch in range(args.num_epochs):  
            model.train()  
            if args.load_full_batch:  
                file_name = f'/home/shuangyan/dataset/fan_out_{args.fan_out}/{args.dataset}_{epoch}_items.pickle'  
                with open(file_name, 'rb') as handle:  
                    full_batch_dataloader = [pickle.load(handle)]  
            
            block_dataloader, weights_list, _, _, _ = dataloader_gen_bucketing_time(full_batch_dataloader, g, [int(fanout) for fanout in args.fan_out.split(',')], args)  

            for step, (input_nodes, seeds, blocks) in enumerate(block_dataloader):  
                batch_inputs, batch_labels = load_subtensor(nfeats, labels, seeds, input_nodes, device)  
                blocks = [block.int().to(device) for block in blocks]  

                batch_pred = model(blocks, batch_inputs)  
                loss = loss_fcn(batch_pred, batch_labels)  

                optimizer.zero_grad()  
                loss.backward()  
                optimizer.step()  

def main():  
    parser = argparse.ArgumentParser("multi-gpu training")  
    parser.add_argument('--device', type=int, default=0, help="GPU device ID. Use -1 for CPU training")  
    parser.add_argument('--seed', type=int, default=1236)  
    parser.add_argument('--dataset', type=str, default='ogbn-products')  
    parser.add_argument('--selection-method', type=str, default='products_25_backpack_bucketing')
    parser.add_argument('--aggre', type=str, default='lstm')  
    parser.add_argument('--num-hidden', type=int, default=128)  
    parser.add_argument('--num-layers', type=int, default=2)  
    parser.add_argument('--fan-out', type=str, default='10,25') 
    parser.add_argument('--mem-constraint', type=float, default=70)
    parser.add_argument('--cluster-coeff', type=float, default=0.411)
    parser.add_argument('--num-batch', type=int, default=1) 
    parser.add_argument('--num-runs', type=int, default=1)  
    parser.add_argument('--num-epochs', type=int, default=10)  
    parser.add_argument('--lr', type=float, default=1e-2)  
    parser.add_argument('--dropout', type=float, default=0.5)  
    parser.add_argument('--load-full-batch', type=bool, default=True)  

    args = parser.parse_args()  
    set_seed(args)  

    device = "cuda:0" if torch.cuda.is_available() else "cpu"  
    g, n_classes = load_data(args)  
    data = prepare_data(g, n_classes, args, device)  

    run(args, device, data)  

if __name__ == '__main__':  
    main()