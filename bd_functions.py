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

import bd_utils

def create_graph(age, gen, edu):
    genGraph = np.zeros((len(gen), len(gen)))
    for k in range(len(gen)):
        for j in range(len(gen)):
            if gen[k] == gen[j]:
                genGraph[k, j] = 1
                genGraph[j, k] = 1

    ageGraph = np.zeros((len(age), len(age)))
    for k in range(len(age)):
        for j in range(len(age)):
            val1 = abs(float(age[k]) - float(age[j]))
            if val1 < 1:
                ageGraph[k, j] = 1
                ageGraph[j, k] = 1

    eduGraph = np.zeros((len(edu), len(edu)))
    for k in range(len(edu)):
        for j in range(len(edu)):
            val = abs(float(edu[k]) - float(edu[j]))
            if val < np.std(edu):
                eduGraph[k, j] = 1
                eduGraph[j, k] = 1

    graph = genGraph * ageGraph * eduGraph
    return graph

def PCA_processing(features, trainIdx, nComponents):
    pca = PCA(nComponents)
    pca = pca.fit(features[trainIdx])
    xPCA = pca.transform(features)
    return xPCA

# Create adj matrix that contains graph info and non-graph info
def final_adj_matrix_created(features_selected, graph):
    distv = distance.pdist(features_selected, 'cityblock')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    sparseGraph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    g = np.tril(sparseGraph, -1) + np.triu(sparseGraph, 1)
    g = g * graph
    adj_normalizd = bd_utils.preprocess_adj(g)
    return adj_normalizd











