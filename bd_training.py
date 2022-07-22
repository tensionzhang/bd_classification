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
        self.parser.add_argument('--init_type', type=str, default='xavier', help='[uniform | xavier]')
        self.parser.add_argument('--model', type=str, default='basic', help='[basic]')

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
        self.isTrain = True


#TODO the main training loop function
def training(fea, adj, label, trainIdx, valIdx, testIdx, nFea):

    bestACC = 0

    pass















