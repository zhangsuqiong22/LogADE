#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directed Graph Convolutional Network (DiGCN) model implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d

from digcnconv import DIGCNConv

class DiGCN(nn.Module):
    """
    Directed Graph Convolutional Network (DiGCN)
    
    This model processes directed graphs using specialized convolution operations
    that preserve directional information in the message passing.
    
    Parameters:
    -----------
    nfeat : int
        Number of input node features
    nhid : int
        Number of hidden dimensions
    nlayer : int
        Number of DiGCN layers
    dropout : float, optional
        Dropout rate (default: 0.0)
    bias : bool, optional
        Whether to use bias in the convolution layers (default: False)
    """
    def __init__(self, nfeat, nhid, nlayer=2, dropout=0.0, bias=False, **kwargs):
        super(DiGCN, self).__init__()
        
        self.dropout = dropout
        self.nlayer = nlayer
        
        # Create multiple layers based on nlayer parameter
        self.convs = nn.ModuleList()
        
        # First layer transforms input features to hidden dimension
        self.convs.append(DIGCNConv(nfeat, nhid, bias=bias))
        
        # Add additional layers if nlayer > 1
        for _ in range(1, nlayer):
            self.convs.append(DIGCNConv(nhid, nhid, bias=bias))
            
        # Optional batch normalization after each layer
        self.bns = nn.ModuleList([BatchNorm1d(nhid) for _ in range(nlayer)])
        
    def reset_parameters(self):
        """Reset all learnable parameters of the model"""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        
    def forward(self, data):
        """
        Forward pass through the network
        
        Parameters:
        -----------
        data : torch_geometric.data.Data
            Input graph data object containing:
            - x: Node features
            - edge_index: Edge indices
            - edge_attr: Edge attributes
            - batch: Batch indices for nodes
            
        Returns:
        --------
        list
            List of node embeddings for each graph in the batch
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Process through each layer
        for i in range(self.nlayer - 1):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Last layer (no activation)
        x = self.convs[-1](x, edge_index, edge_attr)
        
        # Organize embeddings by graph
        emb_list = []
        for g in range(data.num_graphs):
            emb = x[data.batch == g]
            emb_list.append(emb)
        
        return emb_list


class DiGCN_IB_Sum(nn.Module):
    """
    DiGCN with Inception Block and Sum aggregation
    
    This model enhances DiGCN with inception blocks that process the graph
    at different granularities and combines the results.
    
    Parameters:
    -----------
    nfeat : int
        Number of input node features
    nhid : int
        Number of hidden dimensions
    nlayer : int
        Number of DiGCN-IB layers (effectively multiplied by 3 due to inception blocks)
    dropout : float, optional
        Dropout rate (default: 0.1)
    bias : bool, optional
        Whether to use bias in the convolution layers (default: False)
    """
    def __init__(self, nfeat, nhid, nlayer=1, dropout=0.1, bias=False, **kwargs):
        super(DiGCN_IB_Sum, self).__init__()
        
        self.dropout_rate = dropout
        
        # Create inception blocks based on nlayer parameter
        self.blocks = nn.ModuleList()
        
        # First block transforms input features
        self.blocks.append(InceptionBlock(nfeat, nhid, bias=bias))
        
        # Add additional blocks if nlayer > 1
        for _ in range(1, nlayer):
            self.blocks.append(InceptionBlock(nhid, nhid, bias=bias))
            
    def reset_parameters(self):
        """Reset all learnable parameters of the model"""
        for block in self.blocks:
            block.reset_parameters()
        
    def forward(self, data):
        """
        Forward pass through the network
        
        Parameters:
        -----------
        data : torch_geometric.data.Data
            Input graph data object containing:
            - x: Node features
            - edge_index: Edge indices
            - edge_attr: Edge attributes
            - edge_index2: Second-order edge indices
            - edge_attr2: Second-order edge attributes
            - batch: Batch indices for nodes
            
        Returns:
        --------
        list
            List of node embeddings for each graph in the batch
        """
        x = data.x
        edge_index, edge_attr = data.edge_index, data.edge_attr
        edge_index2, edge_attr2 = data.edge_index2, data.edge_attr2
        
        # Process through each inception block
        for block in self.blocks:
            # Get outputs from three branches
            x0, x1, x2 = block(x, edge_index, edge_attr, edge_index2, edge_attr2)
            
            # Apply dropout to each branch
            x0 = F.dropout(x0, p=self.dropout_rate, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_rate, training=self.training)
            x2 = F.dropout(x2, p=self.dropout_rate, training=self.training)
            
            # Sum the branches
            x = x0 + x1 + x2
            
            # Apply dropout to combined output
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Organize embeddings by graph
        emb_list = []
        for g in range(data.num_graphs):
            emb = x[data.batch == g]
            emb_list.append(emb)
        
        return emb_list


class InceptionBlock(nn.Module):
    """
    Inception Block for DiGCN
    
    This block processes input features through three parallel paths:
    1. Linear transformation (identity path)
    2. First-order graph convolution
    3. Second-order graph convolution
    
    Parameters:
    -----------
    in_dim : int
        Input feature dimension
    out_dim : int
        Output feature dimension
    bias : bool, optional
        Whether to use bias in the layers (default: False)
    """
    def __init__(self, in_dim, out_dim, bias=False):
        super(InceptionBlock, self).__init__()
        
        # Three parallel processing paths
        self.ln = Linear(in_dim, out_dim, bias=bias)
        self.conv1 = DIGCNConv(in_dim, out_dim, bias=bias)
        self.conv2 = DIGCNConv(in_dim, out_dim, bias=bias)
        
    def reset_parameters(self):
        """Reset all learnable parameters of the block"""
        self.ln.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr, edge_index2, edge_attr2):
        """
        Forward pass through the inception block
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features
        edge_index : torch.Tensor
            Edge indices for first-order connections
        edge_attr : torch.Tensor
            Edge attributes for first-order connections
        edge_index2 : torch.Tensor
            Edge indices for second-order connections
        edge_attr2 : torch.Tensor
            Edge attributes for second-order connections
            
        Returns:
        --------
        tuple
            Three tensors corresponding to outputs from each branch
        """
        # Identity mapping through linear transformation
        x0 = self.ln(x)
        
        # First-order graph convolution
        x1 = self.conv1(x, edge_index, edge_attr)
        
        # Second-order graph convolution
        x2 = self.conv2(x, edge_index2, edge_attr2)
        
        return x0, x1, x2