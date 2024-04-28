# This demo shows how to call the Viewpoint-Based Collaborative Feature-Weighted
# Multi-View Intuitionistic Fuzzy Clustering Using Neighborhood Information 
# For the demonstration, the prokaryotic dataset of the above paper is used.
# Courtesy of A.Golzari Oskouei

import numpy as np
from main import main
from calculateMetrics import calculateMetrics
import scipy.io
from parameters import parameters
from  Find_Neighbors import  Find_Neighbors
import time
import os
from Center_Points import center_Points

# Load the dataset.
dataset='prokaryotic'

# Algorithm parameters.
k, t_max, NR, alpha1, landa, q, gama, lable_true, number_viwe, data, row = parameters(dataset)

# initializations.
np.random.seed(1373)
center_points = {}
sample_weight = {}
viewpoint = {}

Neig, dm = Find_Neighbors (NR, data, landa, number_viwe);

# --------------------------------------------------------
# Clustering the samples using the proposed procedure.
# --------------------------------------------------------
print(f'========================================================')
print(f'Proposed Algorithm: start.')


for i in range(number_viwe):
    col = data[i].shape[1]
    center_points[i], viewpoint[i] = center_Points(data[i], dm[i], k, col, row, landa[i])

    # Update sample weight
    distance = np.zeros([row, k])
    for j in range(k):
        distance [:, j] = np.sum((np.exp((-1 * np.tile(landa[i], (row, 1))) * ((data[i]-np.tile(center_points[i][j, :], (row, 1))) ** 2))))                                        
    sample_weight[i] = np.sum((distance), 1) / k
    

# Execute proposed algorithm.
# Get the cluster assignments and other parameters.
start_time = time.time()
Cluster_elem = main(viewpoint, data, center_points, k, t_max, NR, alpha1, landa, q, gama, Neig, number_viwe, row, sample_weight)
end_time = time.time()

RunTime_repeat = end_time - start_time

lable_pre = Cluster_elem.argmax(axis=0)

if Cluster_elem is not None and len(np.unique(lable_pre))==k:
    ans = calculateMetrics(lable_pre, lable_true, row)
    ACC_repeat = ans[0]
    NMI_repeat = ans[1]
    PRE_repeat = ans[2]
    REC_repeat = ans[3]
    F_repeat = ans[4]
    R_index_repeat = ans[5]
    Average_R_index_repeat = ans[6]
    FMI_repeat = ans[7]
    JI_repeat = ans[8]

    print(f'The accurcy score is {ans[0]}.')
    print(f'The NMI score is {ans[1]}.')
    print(f'The precision score is {ans[2]}.')
    print(f'The recall score is {ans[3]}.')
    print(f'The F1 score is {ans[4]}.')
    print(f'The R_index score is {ans[5]}.')
    print(f'The Average_R_index score is {ans[6]}.')
    print(f'The FMI score is {ans[7]}.')
    print(f'The JI score is {ans[8]}.') 
    print(f'The runtime is {RunTime_repeat}.')
    print('========================================================')
    
else:
    ACC_repeat = np.nan
    NMI_repeat = np.nan
    PRE_repeat = np.nan
    REC_repeat = np.nan
    F_repeat = np.nan
    R_index_repeat = np.nan
    Average_R_index_repeat = np.nan
    FMI_repeat = np.nan
    JI_repeat = np.nan

    print(f'The accurcy score is NaN.')
    print(f'The NMI score is NaN.')
    print(f'The precision score is NaN.')
    print(f'The recall score is NaN.')
    print(f'The F1 score is NaN.')
    print(f'The R_index score is NaN.')
    print(f'The Average_R_index score is NaN.')
    print(f'The FMI score is NaN.')
    print(f'The JI score is NaN.')
    print(f'The runtime is {RunTime_repeat}.')
    print('========================================================')

