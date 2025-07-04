"""
Define the DiGCN convolution operation
"""
import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, add_self_loops
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.nn.inits import glorot, zeros

class DIGCNConv(MessagePassing):
    r"""
    The graph convolutional operator takes from Pytorch Geometric.
    The spectral operation is the same with Kipf's GCN.
    DiGCN preprocesses the adjacency matrix and does not require a norm operation during the convolution operation.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        cached (bool, optional): If set to :obj:`True`, the layer will cache the adj matrix on first execution, 
                                and will use the cached version for further executions.
                                Please note that, all the normalized adj matrices (including undirected)
                                are calculated in the dataset preprocessing to reduce time comsume.
                                This parameter should only be set to :obj:`True` in transductive learning scenarios. 
                                (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn an additive bias. 
                                (default: :obj:`True`)
        **kwargs (optional): Additional arguments of :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=False,
                 **kwargs):
        
        super(DIGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None
    
    def forward(self, x, edge_index, edge_attr):
        
        x = torch.matmul(x, self.weight)
        
        print("+++++++++++DIGCNConv.py+++++++++++")
        print("---x---")
        print(x)
        # print("---edge_index---")
        # print(edge_index)
        # print("---edge_attr---")
        # print(edge_attr)
   
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if edge_attr is None:
                raise RuntimeError(
                    'Normalized adj matrix cannot be None. Please obtain the adj matrix in preprocessing.')
            else:
                norm = edge_attr
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        print("+++++++++++++DIGCNConv.py+++++++++++")
        print("-----DIGCNConv()->forward()------")
        print("-----size checking-----")
        print(norm.size())
        print(edge_attr.size()) 
        
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        
        print("+++++++++++++DIGCNConv.py++++++++++")
        print("-----DIGCNConv()->message()------")
        print("-----check dimension-----")
        print(norm.view(-1, 1).size())
        print(x_j.size())
        
        return norm.view(-1, 1) * x_j if norm is not None else x_j
        

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
