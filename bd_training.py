"""
This file defines the main traninig loop function and some options classes
"""

import torch
import os
import numpy as np
import random
import sys
import time
import scipy.sparse as sp
import sklearn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial import distance
from torch.autograd import Variable
import argparse

import bd_gcn

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, default='data/', help='path')
        self.parser.add_argument('--dataset', type=str, default='[]')
        self.parser.add_argument('--num_hidden', type=int, default=16, help='number of features')
        self.parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
        self.parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
        self.parser.add_argument('--num_hops', type=int, default=3, help='num_hops')

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain
        args = vars(self.opt)

        return self.opt

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--lr', type=float, default=0.002, help='initial learning rate')
        self.parser.add_argument('--optimizer', type=str, default='adam', help='[sgd | adam]')
        self.parser.add_argument('--epoch', type=int, default=500, help='number of training epochs')
        self.parser.add_argument('--lr_decay_epoch', type=int, default=5000, help='multiply by a gamma every set iter')
        self.parser.add_argument('--nb_heads', type=int, default=16, help='number of head attentions')
        self.parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for the leaky_relu')
        self.isTrain = True

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = False


#TODO the main training loop function
def training(fea, adj, label, trainIdx, valIdx, testIdx, nFea):

    bestACC = 0

    trainIdx = torch.from_numpy(trainIdx.astype(np.int64))
    valIdx = torch.from_numpy(valIdx.astype(np.int64))
    testIdx = torch.from_numpy(testIdx.astype(np.int64))

    opt = TrainOptions().parse()
    useGPU = torch.cuda.is_available()

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    # if useGPU:
    #     torch.cuda.manual_seed(1)

    model, optimizer = None, None

    print("| Constructing the GCN model...")
    model = bd_gcn.GCN(
        nFeatures = nFea,
        nHidden = opt.num_hidden,
        nClass = 2,
        dropout = opt.dropout
    )

    if (opt.optimizer == 'sgd'):
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay,
            momentum=0.9
        )
    elif (opt.optimizer == 'adam'):
        optimizer = optim.Adam(
            model.parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay
        )
    else:
        raise NotImplementedError







    pass















