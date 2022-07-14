import numpy as np
import torch
from scipy.spatial import distance
from sklearn import svm
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

nROI = 164

# make matrix lower triangle a vector for all subjects
# input: data - nROI x nROI matrix
# output: features - (nROI x nROI) x 1 vector

def matrix_lower_triangle(data):
    nFeatures = nROI * (nROI-1) / 2
    features = np.zeros((np.size(data,1), nFeatures))