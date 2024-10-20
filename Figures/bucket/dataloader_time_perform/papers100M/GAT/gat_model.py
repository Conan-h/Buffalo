from gatconv import GATConv,GATConv2
from torch import nn
import torch.nn.functional as F
import torch
import dgl
import tqdm

class GAT(nn.Module):
    def __init__(self, 
                 in_size,
                 aggre,
                 hid_size, 
                 out_size, 
                 heads):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GAT
        self.n_hidden = hid_size
        self.n_classes = out_size
        self.num_heads = heads
        self.aggre = aggre
        self.layers.append(
            GATConv(
                in_size,
                hid_size,
                heads[0],
                self.aggre,
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        self.layers.append(
            GATConv2(
                hid_size * heads[0],
                out_size,
                heads[1],
                self.aggre,
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            )
        )

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    # def forward(self, g, inputs):
    #     h = inputs
    #     for i, layer in enumerate(self.layers):
    #         h = layer(g, h)
    #         if i == 1:  # last layer
    #             h = h.mean(1)
    #         else:  # other layer(s)
    #             h = h.flatten(1)
    #     return h
    
    def forward(self, blocks, x):
        h=x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i == 1:  # last layer
                h = h.view(h.shape[0]*h.shape[1], h.shape[2])
            else:
                h = h.flatten(1)
        return h

	
    def inference(self, g, x, args, device):
        """
		Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
		g : the entire graph.
		x : the input of entire node set.

		The inference code is written in a fashion that it could handle any number of nodes and
		layers.
		"""
		# During inference with sampling, multi-layer blocks are very inefficient because
		# lots of computations in the first few layers are repeated.
		# Therefore, we compute the representation of all nodes layer by layer.  The nodes
		# on each layer are of course splitted in batches.
		# TODO: can we standardize this?
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        
        y = torch.zeros(g.num_nodes(), self.n_classes)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.NodeDataLoader(
            g,
            torch.arange(g.num_nodes(),dtype=torch.long).to(g.device),
            sampler,
            device=device,
            # batch_size=24,
            batch_size=int(g.num_nodes()),
            # batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers)

        
        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            for i in range(len(blocks)):
                # print("layer ", i )
                block = blocks[i]
                block = block.int().to(device)
                if i == 0:  # first layer
                    h = x[input_nodes].to(device)
                    # print('shape of h = x[input_nodes].to(device) ', h.size())
                
                    h = self.layers[0](block, h)
                    # print('shape of h =layer(block, h) ', h.size())
                    h = h.flatten(1)
                    print('h.size()', h.size() )
                else: # i == 1:  # last layer
                    
                    h = self.layers[1](block, h)
                    h = h.view(h.shape[0]*h.shape[1], h.shape[2])
                    # print('shape of h.view(h.shape[0]*h.shape[1], h.shape[2]) ', h.size())
                    y[output_nodes] = h.cpu()
                    # print('shape of y[output_nodes] ', y[output_nodes].size())
            # x = y    
        x = y
        
        # print()
        return y