# -*- coding: utf-8 -*-
"""
@author: Khadijeh
"""
from utilities import *
from models import *
from training import *
from evaluation import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
import scipy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
import mat73
from   sklearn.model_selection import LeaveOneGroupOut
from   sklearn.model_selection import train_test_split
from   sklearn.metrics import roc_auc_score
from   torch.utils.data import DataLoader
from   progress.bar import Bar
import pickle
import pathlib
from   pathlib import Path
from   dgl.nn.pytorch import GraphConv
import dgl
import warnings
import time
from google.colab import files
import mne
from mne_connectivity import spectral_connectivity_epochs
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union

#%% reads data from mat file and creates graphs
mat    = mat73.loadmat('/mygoogledrive/MyDrive/GNN+CNN/modif1_RawFeat_ banana_normalized_12s_11s_10s.mat')
label  = np.array(mat['labelVec']).reshape(39,)
groups = mat['groupVec']
feats  = np.array(mat['nodeFeat']).reshape(39,)
conMat = []

#%%# graph thresholding and preparing graph labels
graphThreshold = 0.1
graphs, labels = prepare_graphs_labels(conMat, label, feats, graphThreshold, SpDisAdj=True)
del mat

#%%
# Initialize parameters
# Dictionary for saving 39 models (one per subject)
models_dict = {}  
logo = LeaveOneGroupOut()
subject = 0 
Adversarial = False
Ablation = False
nepochs = 12
save_results = False
upload_results = True
eegMontage = 'banana'
windowLength = 12  # length of the sliding window in seconds
device = "cuda:0"
batch_size = 256
gradient_accumulations = 4

# List containing 39 dictionaries which contain the output of the analysis process in each validation session
resultList = []
targets_all = []
scores_all = []
AUC_abl = []
KAPPA_abl = []

# Perform Leave-one-subject-out cross-validation (LOSOCC)
for subject in range(39):
    subject += 1
    X_train, X_test, y_train, y_test, d_train, d_test = [], [], [], [], [], []
    tempGroup = np.copy(groups)
    test_index = np.where(tempGroup == subject)[0]
    tempGroup -= (subject)
    train_index = np.nonzero(tempGroup)[0]
    d_train = torch.zeros(len(train_index), 1)
    d_test = torch.ones(len(test_index), 1)

    # Prepare training and testing data
    for k in train_index:
        X_train.append(graphs[k])
        y_train.append(labels[k])
    for j in test_index:
        X_test.append(graphs[j])
        y_test.append(labels[j])

    # Prepare trainset and testset
    trainset = list(zip(X_train, y_train, d_train))
    testset = list(zip(X_test, y_test, d_test))
    start = time.time()

    # Create model
    model = Classifier3(windowLength * 32, 256, 2, domain_adapt=Adversarial)
    model.cuda()

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train the model
    model, test_Y, score_Y, att_test = model_train(
        model, optimizer, scheduler, trainset, testset, nepochs, batch_size, Adversarial, d_train, d_test)

    # Compute AUC of test data
    AUC_test = round(roc_auc_score(test_Y, score_Y), 3)

    # Compute AUC of train data
    data_loader_test = DataLoader(trainset, batch_size=256, collate_fn=perform_collation, shuffle=False, pin_memory=True)
    score_all = torch.Tensor()

    # Other computations
    if Ablation:
    # width&chans: cnn block2 pool2:90,32 ,cnn block3 pool3:41,8, cnn block4 conv8:33,1
        auc_abl, kappa_abl = ablation(trainset, testset,model, layer="conv8", 
                                  classifier=classifier_ablation,
                                  conv_chans=1, conv_width=33)
        AUC_abl.append(auc_abl)
        KAPPA_abl.append(kappa_abl)
        print(f"AUC_conv4 = {auc_abl} , Kappa_conv4 = {kappa_abl}")
    # print the AUCs of each iteration
    print('execution time = ', (time.time()-start)/60)
    # Save the true labels and scores 
    results = {'target' : test_Y , 'score' : score_Y}
    targets_all.append(test_Y.squeeze())
    scores_all.append(score_Y.squeeze())
    # print the AUCs of each subject
    resultList.append(results)
    print(f'subject: {subject}/39')
    print(f'AUC_test =  {AUC_test}')
    print("AUC_train =", AUC_train)
    print('----------------------')
# Converts the result list to a panda dataframe
resultsdict = {"target": targets_all, "score": scores_all}
df = pd.DataFrame(resultsdict)
# Save results if required
if save_results:
  df.to_csv("out.csv")
  #files.download("out.csv")
#df = pd.DataFrame(resultList)

#%% Compute performance
subject_range      = range(39)
analysisResultList = []
print("Number of Model's Parameters" ,sum(param.numel() for param in model.parameters()))
bar = Bar('Please wait', max = 39, check_tty = False, hide_cursor = False)

for subject in subject_range:

    targets = df['target'][subject]
    scores  = df['score'][subject]
    post_scores,acc,sen,spec,tpr,fpr,auc,auc_sk,auc_sk90,gdrs,fdhs,fdurs, kappa=compute_performance(scores,targets,10,30) # collar, MA



    analysisResults = {'target' : targets, 'score' : post_scores,
                       'acc'   : acc,                  'sen'    : sen ,    'spec'  : spec, 
                       'tpr'   : tpr,                  'fpr'    : fpr,     'auc'   : auc,
                       'auc_sklearn' : auc_sk, 'auc_sklearn90' : auc_sk90,
                       "gdr"  : gdrs, "fdh"  : fdhs, "fdur" : fdurs,  "kappa" : kappa}

    analysisResultList.append(analysisResults)
    bar.next()
bar.finish()

# converts the result list to a panda dataframe
analysisDataFrame = pd.DataFrame(analysisResultList)

# saves as .pkl file
#pklFileName = dataDirectory / f'{eegMontage}_plv_{model.conv1.weight.shape[1]}_{model.conv2.weight.shape[1]}.pkl'
#analysisDataFrame.to_pickle(pklFileName, protocol = 4)
# shows the auc values for all subjects
plt.bar(range(39), analysisDataFrame['auc'])
plt.plot(range(39), [analysisDataFrame['auc'].mean()] * 39, 'r--')
plt.xlabel('Subjects')
plt.ylabel('AUC')

# prints the average and median of AUCs across subjetcs
mean_auc   = analysisDataFrame['auc'].mean()
median_auc = analysisDataFrame['auc'].median()
Q3         = analysisDataFrame['auc'].quantile(0.75)
Q1         = analysisDataFrame['auc'].quantile(0.25)
print(f'Mean of AUC is equal to: {round(mean_auc,  3)}')
print(f'Median of AUC is equal to: {round(median_auc, 3)} ({round(Q1, 3)},{round(Q3, 3)})')

# prints the average and median of AUCs using sklearn function across subjetcs
mean_auc   = analysisDataFrame['auc_sklearn'].mean()
median_auc = analysisDataFrame['auc_sklearn'].median()
Q3         = analysisDataFrame['auc_sklearn'].quantile(0.75)
Q1         = analysisDataFrame['auc_sklearn'].quantile(0.25)
print(f'Mean of AUC_scikit is equal to: {round(mean_auc,  3)}')
print(f'Median of AUC_scikit is equal to: {round(median_auc, 3)} ({round(Q1, 3)},{round(Q3, 3)})')

# prints the average and median of AUC90 using sklearn function across subjetcs
mean_auc   = analysisDataFrame['auc_sklearn90'].mean()
median_auc = analysisDataFrame['auc_sklearn90'].median()
Q3         = analysisDataFrame['auc_sklearn90'].quantile(0.75)
Q1         = analysisDataFrame['auc_sklearn90'].quantile(0.25)
print(f'Mean of AUC_scikit_90% is equal to: {round(mean_auc,  3)}')
print(f'Median of AUC_scikit-90% is equal to: {round(median_auc, 3)} ({round(Q1, 3)},{round(Q3, 3)})')
print(f"The AUC of conv7: {np.array(AUC_abl_ts).mean()}")

#%% Print performance tabel
from prettytable import PrettyTable
kappa = analysisDataFrame['kappa']
metrics_to_show = ['acc', 'sen', 'spec', 'tpr', 'fpr', 'gdr', 'fdh', 'fdur', 'kappa']
best            = []
bests_dict      = {}
thr             = []    ### index of the maximum kappa

for i in subject_range:
  thr.append(int(kappa[i].argmax()))

for jj in range(len(metrics_to_show)):
  bestt = []
  for j in subject_range:
    bestt.append((analysisDataFrame.iloc[j][metrics_to_show[jj]][int(thr[j])]))
  bests_dict.update({metrics_to_show[jj]: bestt})
  del bestt

z = PrettyTable()
#z.add_column("threshold",np.round(np.linspace(0, 1, 50), 2))
z.add_column("Subject ", np.arange(1,len(subject_range)+1))
z.add_column("gdr (%)", np.round(bests_dict["gdr"],2))
z.add_column("fdh (/hour)", np.round(bests_dict["fdh"],2))
z.add_column("fdur (min)", np.round(bests_dict["fdur"],2))
z.add_column("acc ", np.round(bests_dict["acc"],2))
z.add_column("sen ", np.round(bests_dict["sen"],2))
z.add_column("spec ", np.round(bests_dict["spec"],2))
z.add_column("kappa ", np.round(bests_dict["kappa"],2))
z.add_column("AUC ", analysisDataFrame['auc_sklearn'])
z.add_column("Best threshold ", thr)

z.add_row(["  MEAN  ", np.round(np.array(bests_dict["gdr"]).mean(),2), np.round(np.array(bests_dict["fdh"]).mean(),2),
           np.round(np.array(bests_dict["fdur"]).mean(),2), np.round(np.array(bests_dict["acc"]).mean(),2),
           np.round(np.array(bests_dict["sen"]).mean(),2), np.round(np.array(bests_dict["spec"]).mean(),2),
           np.round(np.array(bests_dict["kappa"]).mean(),2),  np.round(analysisDataFrame['auc_sklearn'].mean(),3), 0.5])

z.add_row(["  MEDIAN  ", np.round(np.median(bests_dict["gdr"]),2), np.round(np.median(bests_dict["fdh"]),2),
           np.round(np.median(bests_dict["fdur"]),2), np.round(np.median(bests_dict["acc"]),2),
           np.round(np.median(bests_dict["sen"]),2), np.round(np.median(bests_dict["spec"]),2),
           np.round(np.median(bests_dict["kappa"]),2), np.round(analysisDataFrame['auc_sklearn'].median(),3), 0.5])

print(z.get_string(title=" based on max kappa"))

###############################################################################
# print model structure
print(model)
###############################################################################
# save the table as .txt, you can download tabel.txt
data = z.get_string()
with open('tabel.txt', 'wb') as f:
    f.write(data.encode())
