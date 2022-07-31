"""
This file is the main program of BD Classification
"""

import numpy as np
import torch
import scipy.io as sio
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold

import os
import os.path

import bd_utils
import bd_training

# import data matrix
dataFile = u'data/dynamic_features/Mean_dfc.mat'
data = sio.loadmat(dataFile, mat_dtype = True)['Mean_dfc']

nSubject = np.size(data,1)
nROI = np.size(data[0][0],1)

features = np.zeros((nSubject, int(nROI * (nROI-1) / 2)))
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

# create the non-image information graph, an entry is 1 when age, gender and education is similar for two subjects
nonImgInfoGraph = bd_utils.create_graph(ageInfo, genInfo, eduInfo)

# cross validation, cvSplits contains groups of features and labels
skf = StratifiedKFold(n_splits=10, shuffle=False) # change this shuffle to True
cvSplits = list(skf.split(features, labelInfo))

for i in range(len(cvSplits)):

    trainData = cvSplits[i][0]
    np.random.shuffle(trainData)
    trainIdx = trainData[:int(trainData.shape[0] * 0.7)] #? why multiply 0.7 here?
    valIdx = trainData[int(trainData.shape[0] * 0.7):]
    testIdx = cvSplits[i][1]

    print("---------", trainIdx.shape, valIdx.shape, testIdx.shape)

    featuresPCA = bd_utils.PCA_processing(features, trainIdx, 0.9)
    allFeatureSelected = featuresPCA

    # Train the model
    adj = bd_utils.create_adj(allFeatureSelected, nonImgInfoGraph) # merge feature information (image) and non-image information together
    
    nFeatures = np.size(allFeatureSelected, 1)
    featureSelected = torch.from_numpy(allFeatureSelected.astype(np.float32))
    label = labelInfo.squeeze()
    label = torch.from_numpy(label.astype(np.int64))

    pred = bd_training.training(featureSelected, adj, label, trainIdx, valIdx, testIdx, nFeatures)
    pred = pred.cpu().detach().numpy()

    pred = np.array(torch.from_numpy(pred).max(1)[1])

    pass

pass














































