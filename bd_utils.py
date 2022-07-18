import numpy as np
import torch
from scipy.spatial import distance
from sklearn import svm
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

import scipy.sparse as sp

nROI = 164

# make matrix lower triangle a vector for all subjects
# input: data - nROI x nROI matrix
# output: features - (nROI x nROI) x 1 vector

def matrix_lower_triangle(data):
    nFeatures = nROI * (nROI-1) / 2
    features = np.zeros((np.size(data,1), nFeatures))

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # degree
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).tocoo()

def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0])) # add self-loop
    indices = torch.from_numpy(np.asarray([adj_normalized.row, adj_normalized.col]).astype(int)).long()
    values = torch.from_numpy(adj_normalized.data.astype(np.float32))
    adj_normalized = torch.sparse.FloatTensor(indices, values, (len(adj), len(adj)))
    return adj_normalized