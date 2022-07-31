"""
This file defines some functions that need to be used
"""

import numpy as np
import torch
from scipy import stats
from sklearn.svm import SVC
import scipy.sparse as sp
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr
from sklearn.preprocessing import Normalizer
from scipy.spatial import distance

def create_graph(age, gen, edu):
    ageGraph = np.zeros((len(age), len(age)))
    ageDiffCriteria = 3
    for i in range(len(age)):
        for j in range(len(age)):
            ageDiff = abs(float(age[i]) - float(age[j]))
            if ageDiff <= ageDiffCriteria:
                ageGraph[i, j] = 1
                ageGraph[j, i] = 1

    genGraph = np.zeros((len(gen), len(gen)))
    for i in range(len(gen)):
        for j in range(len(gen)):
            if gen[i] == gen[j]:
                genGraph[i, j] = 1
                genGraph[j, i] = 1

    eduGraph = np.zeros((len(edu), len(edu)))
    for i in range(len(edu)):
        for j in range(len(edu)):
            eduDiff = abs(float(edu[i]) - float(edu[j]))
            if eduDiff < np.std(edu):
                eduGraph[i, j] = 1
                eduGraph[j, i] = 1

    graph = genGraph * ageGraph * eduGraph
    return graph

def PCA_processing(fea, trainIdx, nComponents):
    pca = PCA(nComponents)
    pca = pca.fit(fea[trainIdx])
    xPCA = pca.transform(fea)
    return xPCA

# Create adj matrix that contains graph info and non-graph info
def create_adj(fea, graph):
    distv = distance.pdist(fea, 'cityblock') # Pairwise distances between observations in n-dimensional space.
    dist = distance.squareform(distv) # Convert a vector-form distance vector to a square-form distance matrix, and vice-versa.
    sigma = np.mean(dist)
    sparseGraph = np.exp(- dist ** 2 / (2 * sigma ** 2)) # Calculate the feature similarity
    g = np.tril(sparseGraph, -1) + np.triu(sparseGraph, 1)
    g = g * graph
    adj_normalizd = preprocess_adj(g)
    return adj_normalizd

def preprocess_adj(adj):
    # normalizedAdj = row_normalize_adj(adj + sp.eye(adj.shape[0])) # add self-loop and normalize
    normalizedAdj = sym_normalize_adj(adj)

    # make it sparse
    indices = torch.from_numpy(np.asarray([normalizedAdj.row, normalizedAdj.col]).astype(int)).long() 
    values = torch.from_numpy(normalizedAdj.data.astype(np.float32))
    normalizedAdjSp = torch.sparse.FloatTensor(indices, values, (len(adj), len(adj)))
    return normalizedAdjSp

def row_normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    rInv = np.power(rowsum, -1).flatten()
    rInv[np.isinf(rInv)] = 0.
    rInvDiag = sp.diags(rInv)
    normAdj = rInvDiag.dot(adj)
    return normAdj

# compute L=D^-0.5 * (A+I) * D^-0.5
def sym_normalize_adj(adj):
    adj += sp.eye(adj.shape[0])
    D = np.array(adj.sum(1))
    dHat = sp.diags(np.power(D, -0.5).flatten())
    normAdj = dHat.dot(adj).dot(dHat)
    return normAdj







#### Not in use yet #################################################################################################
def matrix_lower_triangle(data):
    nROI = 164
    nFeatures = nROI * (nROI-1) / 2
    features = np.zeros((np.size(data,1), nFeatures))

def encode_onehot(labels):
    classes = set(labels)
    classesDict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labelsOnehot = np.array(list(map(classesDict.get, labels)), dtype=np.int32)
    return labelsOnehot








