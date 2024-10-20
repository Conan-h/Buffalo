import dgl
import cugraph_dgl

dataset = dgl.data.CoraGraphDataset()
dgl_g = dataset[0]
# Add self loops as cugraph
# does not support isolated vertices yet
dgl_g = dgl.add_self_loop(dgl_g)
cugraph_g = cugraph_dgl.convert.cugraph_storage_from_heterograph(dgl_g, single_gpu=True)

# Drop in replacement for dgl.nn.SAGEConv
from dgl.nn import CuGraphSAGEConv as SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l_id, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l_id != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


# Create the model with given dimensions
feat_size = cugraph_g.ndata["feat"]["_N"].shape[1]

model = SAGE(feat_size, 256, dataset.num_classes).to("cuda")


acc = evaluate(model, features, labels, dataloader)
print("Epoch {:05d} | Acc {:.4f} | Loss {:.4f} ".format(epoch, acc, total_loss))


def evaluate(model, features, labels, dataloader):
    with torch.no_grad():
        model.eval()
        ys = []
        y_hats = []
        for it, (in_nodes, out_nodes, blocks) in enumerate(dataloader):
            with torch.no_grad():
                x = features[in_nodes]
                ys.append(labels[out_nodes])
                y_hats.append(model(blocks, x))
        num_classes = y_hats[0].shape[1]
        return MF.accuracy(
            torch.cat(y_hats),
            torch.cat(ys),
            task="multiclass",
            num_classes=num_classes,
        )

train(cugraph_g, model)