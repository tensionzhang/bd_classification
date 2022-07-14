import os.path
import os
import numpy as np
import torch
import scipy.io as sio
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold

import bd_functions

# import data matrix
dataFile = u'data/dynamic_features/Mean_dfc.mat'
data = sio.loadmat(dataFile, mat_dtype = True)['Mean_dfc']

nSubject = np.size(data,1)
nROI = np.size(data[0][0],1)
nFeatures = int(nROI * (nROI-1) / 2)

features = np.zeros((nSubject, nFeatures))
for i in range(nSubject):
    matlow = np.tril(data[0][i], -1)
    features[i] = matlow.ravel()[np.flatnonzero(matlow)]

# import subjects information: label, age, sex and education
labelFile = u'data/label_65.mat'
labelInfo = sio.loadmat(labelFile, mat_dtype = True)['label'].astype(int).reshape(-1)

ageFile = u'data/age.mat'
ageInfo = sio.loadmat(ageFile, mat_dtype = True)['age']

genFile = u'data/gender.mat'
genInfo = sio.loadmat(genFile, mat_dtype = True)['gender']

eduFile = u'data/edu.mat'
eduInfo = sio.loadmat(eduFile, mat_dtype = True)['edu']

# create the graph
graph = bd_functions.create_graph(ageInfo, genInfo, eduInfo)

# cross validation
skf = StratifiedKFold(n_splits=10, shuffle=False) # change this shuffle to True
cvSplits = list(skf.split(features, labelInfo))

for i in range(len(cvSplits)):

    train_val = cvSplits[i][0]
    np.random.shuffle(train_val)
    idx_train = train_val[:int(train_val.shape[0] * 0.7)]
    idx_val = train_val[int(train_val.shape[0] * 0.7):]
    idx_test = cvSplits[i][1]
    print("---------", idx_train.shape, idx_val.shape, idx_test.shape)

    trainData = cvSplits[i][0]
    np.random.shuffle(trainData)
    trainIdx = trainData[:int(trainData.shape[0])]
    valIdx = trainData[int(trainData.shape[0]):]

    testIdx = cvSplits[i][1]

    print("---------", trainIdx.shape, valIdx.shape, testIdx.shape)

    pass

pass














































