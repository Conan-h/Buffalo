"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
import dgl
import dgl.function as fn
from dgl import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.utils import Identity
from collections import Counter
import sys
sys.path.insert(0,'../pytorch/utils/')
sys.path.insert(0,'../pytorch/micro_batch_train/')
sys.path.insert(0,'../pytorch/models/')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/utils')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/models')
from memory_usage import see_memory_usage, nvidia_smi_usage

# pylint: enable=W0235
class GATConv(nn.Module):
    r"""Graph attention layer from `Graph Attention Network
    <https://arxiv.org/pdf/1710.10903.pdf>`__

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} &= \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} &= \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
        GATConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    bias : bool, optional
        If True, learns a bias term. Defaults: ``True``.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GATConv

    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> gatconv = GATConv(10, 2, num_heads=3)
    >>> res = gatconv(g, feat)
    >>> res
    tensor([[[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]]], grad_fn=<BinaryReduceBackward>)

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.heterograph({('A', 'r', 'B'): (u, v)})
    >>> u_feat = th.tensor(np.random.rand(2, 5).astype(np.float32))
    >>> v_feat = th.tensor(np.random.rand(4, 10).astype(np.float32))
    >>> gatconv = GATConv((5,10), 2, 3)
    >>> res = gatconv(g, (u_feat, v_feat))
    >>> res
    tensor([[[-0.6066,  1.0268],
            [-0.5945, -0.4801],
            [ 0.1594,  0.3825]],
            [[ 0.0268,  1.0783],
            [ 0.5041, -1.3025],
            [ 0.6568,  0.7048]],
            [[-0.2688,  1.0543],
            [-0.0315, -0.9016],
            [ 0.3943,  0.5347]],
            [[-0.6066,  1.0268],
            [-0.5945, -0.4801],
            [ 0.1594,  0.3825]]], grad_fn=<BinaryReduceBackward>)
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        aggregator_type,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super(GATConv, self).__init__()
        
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        print('self._in_src_feats, ', self._in_src_feats)
        print('self._in_dst_feats', self._in_dst_feats)
        self._out_feats = out_feats
        self.hidden = out_feats
        self._aggre_type = aggregator_type
        self._allow_zero_in_degree = allow_zero_in_degree
        valid_aggre_types = {"sum", "lstm"}
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(
                 num_heads * self.hidden,  num_heads * self.hidden , batch_first=True
            )
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False
            )
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
        self.attn_l = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_r = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias
                )
                self.has_linear_res = True
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(
                th.FloatTensor(size=(num_heads * out_feats,))
            )
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        # print('mailbox .size', m.size())
        # print()
        last_two_dim_size = m.size(-2) * m.size(-1)
        m = m.view(m.shape[0], m.shape[1],last_two_dim_size)
        batch_size = m.shape[0]
        # print('batch size ', m.shape[0])
        h = (
            m.new_zeros((1, batch_size, last_two_dim_size)),
            m.new_zeros((1, batch_size, last_two_dim_size)),
        )
        
        
        _, (rst, _) = self.lstm(m, h)
        # print('------rst shape ', rst.size())
        return {"neigh": rst.squeeze(0)}

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            A 1D tensor of edge weight values.  Shape: :math:`(|E|,)`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, *, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        # print('graph ', graph)
        graph = graph.local_var()
        # print('graph.local ', graph)
        
        if (graph.in_degrees() == 0).any():
            print(graph.in_degrees() == 0)
            # graph = dgl.add_self_loop(graph.cpu()) # block can't add self-loop
            # print(graph)
            
            # raise DGLError(
            #     "There are 0-in-degree nodes in the graph, "
            #     "output for those nodes will be invalid. "
            #     "This is harmful for some applications, "
            #     "causing silent performance regression. "
            #     "Adding self-loop on the input graph by "
            #     "calling `g = dgl.add_self_loop(g)` will resolve "
            #     "the issue. Setting ``allow_zero_in_degree`` "
            #     "to be `True` when constructing this module will "
            #     "suppress the check and let the code run."
            # )

        if isinstance(feat, tuple):
                # print('feat size ', feat.size())
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
        else:
            # print('first layer else: feat size ', feat.size())
            
            src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
            h_src = h_dst = self.feat_drop(feat)
            # print('src_prefix_shape ', src_prefix_shape)
            # print('h_src = h_dst = self.feat_drop(feat) ', h_src.size())
            # print('self._num_heads', self._num_heads)
            # print('self._out_feats', self._out_feats)
            feat_src = feat_dst = self.fc(h_src).view(
                *src_prefix_shape, self._num_heads, self._out_feats
            )
            # print('feat_src = feat_dst = self.fc(h_src).view ', feat_src.size())
            if graph.is_block:
                # print('***** graph.is_block ')
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
                h_dst = h_dst[: graph.number_of_dst_nodes()]
                dst_prefix_shape = (
                    graph.number_of_dst_nodes(),
                ) + dst_prefix_shape[1:]
                # print('feat_dst ', feat_dst.size())
                # print('h_dst ', h_dst.size())
                # print('dst_prefix_shape ', dst_prefix_shape)
        # NOTE: GAT paper uses "first concatenation then linear projection"
        # to compute attention scores, while ours is "first projection then
        # addition", the two approaches are mathematically equivalent:
        # We decompose the weight vector a mentioned in the paper into
        # [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        # Our implementation is much efficient because we do not need to
        # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
        # addition could be optimized with DGL's built-in function u_add_v,
        # which further speeds up computation and saves memory footprint.
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        # print("el size", el.size())
        # print("er szie", er.size())
        graph.srcdata.update({"ft": feat_src, "el": el})
        # print("graph.srcdata[ft] ", graph.srcdata["ft"].size())
        graph.dstdata.update({"er": er})
        # print("graph.dstdata[ft] ", graph.dstdata["ft"].size())
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v("el", "er", "e"))
        e = self.leaky_relu(graph.edata.pop("e"))
        # compute softmax
        graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
        # print('------graph.edata a', graph.edata["a"])
        # print('------graph.edata a size', graph.edata["a"].size())
        if edge_weight is not None:
            graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
                1, self._num_heads, 1
            ).transpose(0, 2)
        msg_func = fn.u_mul_e("ft", "a", "m")
        # print('msg_func ', msg_func)
        # message passing
        
        if self._aggre_type == "sum":
            graph.update_all(msg_func, fn.sum("m", "ft"))
        
        if self._aggre_type == "lstm":
            graph_in = Counter(graph.in_degrees().tolist())
            # print('graph-in degree')
            graph_in= dict(sorted(graph_in.items()))
            # print(graph_in)
            graph.update_all(msg_func, self._lstm_reducer)
            graph.dstdata["ft"] = graph.dstdata['neigh']
            # print('graph.dstdata["ft"] ', graph.dstdata["ft"].size())
            graph.dstdata["ft"].shape[0]
            graph.dstdata["ft"]= graph.dstdata["ft"].view(graph.dstdata["ft"].shape[0],self._num_heads, self.hidden)
        rst = graph.dstdata["ft"]
        # print('rst.size()', rst.size())
        # residual
        if self.res_fc is not None:
            # Use -1 rather than self._num_heads to handle broadcasting
            resval = self.res_fc(h_dst).view(
                *dst_prefix_shape, -1, self._out_feats
            )
            rst = rst + resval
        # bias
        if self.has_explicit_bias:
            # print('self.has_explicit_bias --------')
            # print('self._out_feats ', self._out_feats)
            rst = rst + self.bias.view(
                *((1,) * len(dst_prefix_shape)),
                self._num_heads,
                self._out_feats
            )
        # activation
        if self.activation:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata["a"]
        else:
            return rst
        
        
class GATConv2(nn.Module):
    r"""Graph attention layer from `Graph Attention Network
    <https://arxiv.org/pdf/1710.10903.pdf>`__

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} &= \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} &= \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
        GATConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    bias : bool, optional
        If True, learns a bias term. Defaults: ``True``.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GATConv

    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> gatconv = GATConv(10, 2, num_heads=3)
    >>> res = gatconv(g, feat)
    >>> res
    tensor([[[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]],
            [[ 3.4570,  1.8634],
            [ 1.3805, -0.0762],
            [ 1.0390, -1.1479]]], grad_fn=<BinaryReduceBackward>)

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.heterograph({('A', 'r', 'B'): (u, v)})
    >>> u_feat = th.tensor(np.random.rand(2, 5).astype(np.float32))
    >>> v_feat = th.tensor(np.random.rand(4, 10).astype(np.float32))
    >>> gatconv = GATConv((5,10), 2, 3)
    >>> res = gatconv(g, (u_feat, v_feat))
    >>> res
    tensor([[[-0.6066,  1.0268],
            [-0.5945, -0.4801],
            [ 0.1594,  0.3825]],
            [[ 0.0268,  1.0783],
            [ 0.5041, -1.3025],
            [ 0.6568,  0.7048]],
            [[-0.2688,  1.0543],
            [-0.0315, -0.9016],
            [ 0.3943,  0.5347]],
            [[-0.6066,  1.0268],
            [-0.5945, -0.4801],
            [ 0.1594,  0.3825]]], grad_fn=<BinaryReduceBackward>)
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        aggregator_type,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
    ):
        super(GATConv2, self).__init__()
        
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.hidden = out_feats
        self._aggre_type = aggregator_type
        self._allow_zero_in_degree = allow_zero_in_degree
        valid_aggre_types = {"sum", "lstm"}
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(
                self.hidden, num_heads* out_feats, batch_first=True
            )
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False
            )
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
        self.attn_l = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_r = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias
                )
                self.has_linear_res = True
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(
                th.FloatTensor(size=(num_heads * out_feats,))
            )
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        # print('mailbox .size', m.size())
        # print()
        last_two_dim_size = m.size(-2) * m.size(-1)
        m = m.view(m.shape[0], m.shape[1],last_two_dim_size)
        batch_size = m.shape[0]
        # print('batch size ', m.shape[0])
        h = (
            m.new_zeros((1, batch_size, last_two_dim_size)),
            m.new_zeros((1, batch_size, last_two_dim_size)),
        )
        _, (rst, _) = self.lstm(m, h)
        # print('------rst shape ', rst.size())
        return {"neigh": rst.squeeze(0)}

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            A 1D tensor of edge weight values.  Shape: :math:`(|E|,)`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, *, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        graph = graph.local_var()
        
        if (graph.in_degrees() == 0).any():
            # graph = dgl.add_self_loop(graph)
            print('raise DGLError()')
            raise DGLError(
                "There are 0-in-degree nodes in the graph, "
                "output for those nodes will be invalid. "
                "This is harmful for some applications, "
                "causing silent performance regression. "
                "Adding self-loop on the input graph by "
                "calling `g = dgl.add_self_loop(g)` will resolve "
                "the issue. Setting ``allow_zero_in_degree`` "
                "to be `True` when constructing this module will "
                "suppress the check and let the code run."
            )

        if isinstance(feat, tuple):
                print('feat size ', feat.size())
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
        else:
            # print('--else: feat size ', feat.size())
            src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
            # print('src_prefix_shape ', src_prefix_shape)
            h_src = h_dst = self.feat_drop(feat)
            # print('h_src = h_dst = self.feat_drop(feat) ', h_src.size())
            feat_src = feat_dst = self.fc(h_src).view(
                *src_prefix_shape, self._num_heads, self._out_feats
            )
            # print('feat_src = feat_dst ', feat_src.size())
            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
                h_dst = h_dst[: graph.number_of_dst_nodes()]
                dst_prefix_shape = (
                    graph.number_of_dst_nodes(),
                ) + dst_prefix_shape[1:]
        # NOTE: GAT paper uses "first concatenation then linear projection"
        # to compute attention scores, while ours is "first projection then
        # addition", the two approaches are mathematically equivalent:
        # We decompose the weight vector a mentioned in the paper into
        # [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        # Our implementation is much efficient because we do not need to
        # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
        # addition could be optimized with DGL's built-in function u_add_v,
        # which further speeds up computation and saves memory footprint.
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        # print("el size", el.size())
        # print("er szie", er.size())
        graph.srcdata.update({"ft": feat_src, "el": el})
        # print("graph.srcdata[ft] ", graph.srcdata["ft"].size())
        graph.dstdata.update({"er": er})
        # print("graph.dstdata[ft] ", graph.dstdata["ft"].size())
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v("el", "er", "e"))
        e = self.leaky_relu(graph.edata.pop("e"))
        # compute softmax
        graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
        # print('------graph.edata a', graph.edata["a"])
        # print('------graph.edata a size', graph.edata["a"].size())
        if edge_weight is not None:
            graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
                1, self._num_heads, 1
            ).transpose(0, 2)
        msg_func = fn.u_mul_e("ft", "a", "m")
        # print('msg_func ', msg_func)
        # message passing
        
        if self._aggre_type == "sum":
            graph.update_all(msg_func, fn.sum("m", "ft"))
        
        if self._aggre_type == "lstm":
            graph_in = Counter(graph.in_degrees().tolist())
            # print('graph-in degree')
            graph_in= dict(sorted(graph_in.items()))
            
            graph.update_all(msg_func, self._lstm_reducer)
            graph.dstdata["ft"] = graph.dstdata['neigh']
            
            if self._num_heads == 1:
                graph.dstdata["ft"]= graph.dstdata["ft"].view(graph.dstdata["ft"].shape[0],self.hidden)
            else:# self._num_heads == 1:
                graph.dstdata["ft"]= graph.dstdata["ft"].view(graph.dstdata["ft"].shape[0],self._num_heads, self.hidden)
        rst = graph.dstdata["ft"]
        # print("rst = graph.dstdata[ft] ", rst.size())
        # residual
        if self.res_fc is not None:
            # Use -1 rather than self._num_heads to handle broadcasting
            resval = self.res_fc(h_dst).view(
                *dst_prefix_shape, -1, self._out_feats
            )
            rst = rst + resval
        # bias
        if self.has_explicit_bias:
            rst = rst + self.bias.view(
                *((1,) * len(dst_prefix_shape)),
                self._num_heads,
                self._out_feats
            )
        # activation
        if self.activation:
            rst = self.activation(rst)

        if get_attention:
            print('get_attention return rst, graph.edata')
            return rst, graph.edata["a"]
        else:
            # print('return rst ')
            # print(rst.size())
            return rst