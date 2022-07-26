"""
This file defines the GCN layer and the GCN model
Ref: https://github.com/tkipf/pygcn/tree/master/pygcn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    """Define GCN layer"""
    def __init__(self, inFeatures, outFeatures, bias=True):
        super(GraphConvolution, self).__init__()
        self.inFea = inFeatures
        self.outFea = outFeatures
        self.weight = Parameter(torch.FloatTensor(inFeatures, outFeatures))
        if bias:
            self.bias = Parameter(torch.FloatTensor(outFeatures))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.inFea) + ' -> ' \
               + str(self.outFea) + ')'



class GCN(nn.Module):
    """Define GCN model with two layers"""
    def __init__(self, nFeatures, nHidden, nClass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nFeatures, nHidden)
        self.gc2 = GraphConvolution(nHidden, nClass)
        self.dropout = dropout

    def bottleneck(self, path1, path2, adj, in_x):
        return F.relu(path2(F.relu(path1(in_x, adj)), adj))

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
