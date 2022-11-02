import torch
import os 

import pandas as pd
import numpy as np
# import scanpy as sc
import os
# from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

import functions as fx
import models as mx
import run as rx

import torch
import pandas as pd
import umap
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pylab as plt
import logging

import multiprocessing


os.environ['CUDA_LAUNCH_BLOCKING'] = "0,1,2,3"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info
print2log(torch.cuda.device_count())

def transform_indices(indices_df):    
    # Transforms indices of nearest neighbors from sample to fit the indices of the larger dataframe.
    indices_new = np.array([indices_df.index[x] for x in indices_df.values.flatten()]).reshape(indices_df.shape)
    indices_new = pd.DataFrame(indices_new, index=indices_df.index)
    return indices_new

df = pd.read_pickle('results/fulldata/x_train.pickle')
dfz = pd.read_pickle('results/fulldata/latent_nonpd1.pickle')

# Step 1: Get the distances matrix
sample = dfz#.sample(1000)
sample_clean = sample.drop(columns=["source"])
nrst_neigh = NearestNeighbors(n_neighbors = 10, algorithm = 'auto')
nrst_neigh.fit(sample_clean)
distances, indices = nrst_neigh.kneighbors(sample_clean)

indices_df = pd.DataFrame(indices, index=sample_clean.index, dtype="Int64")
indices_df['source'] = sample.source
indices_df.to_csv('results/fulldata/nn_indices.csv')

distances_df = pd.DataFrame(distances, index=sample_clean.index)
distances_df['source'] = sample.source
distances_df.to_csv('results/fulldata/nn_distances.csv')

# indices_df = transform_indices(pd.read_csv('results/fulldata/nn_indices.csv', index_col="id"))

# Replace all the neighbors from the same source with NaNs
source0 = indices_df[sample.source==0]
s0_ids = list(source0.index)

source1 = indices_df[sample.source==1]
s1_ids = list(source1.index)

source2 = indices_df[sample.source==2]
s2_ids = list(source2.index)


s0_indices = source0.replace(s0_ids, np.NaN)
s1_indices = source1.replace(s1_ids, np.NaN)
s2_indices = source2.replace(s2_ids, np.NaN)

ori = pd.read_csv("../data/processed/fulldata/mitsialis_bengsch_rahman_all.csv", index_col="id")

ori0 = ori.loc[s0_ids,:]
ori1 = ori.loc[s1_ids,:]
ori2 = ori.loc[s2_ids,:]


pd.options.mode.chained_assignment = None  # default='warn'

temp_ori = ori.drop(columns=['source'])
non_missing_columns = list(temp_ori[temp_ori.columns[~temp_ori.isnull().any()]].columns)

imputed_df = pd.DataFrame()

indices = list(indices_df.index)[:10]

for x in indices:
    sample = temp_ori.loc[x]
    sample_nans = temp_ori.loc 
    # Get the indices of all the neighbors from differnt sources (we set same source to be na)
    nn_indices = indices_df.loc[x,:].dropna() 
    # Get the actual neighbor samples
    neighbors = temp_ori.loc[nn_indices,:]
    # Get the markers that need to be imputed
    missing_markers = list(sample[sample.isna()].index)
    # # Get the mean for each marker from the neighbors 
    imputed_markers = pd.DataFrame(neighbors.mean()).T
    imputed_df = pd.concat([imputed_df, imputed_markers])

imputed_df.index = indices

# Get the existing markers
non_missing_imputed_df = imputed_df[non_missing_columns]
non_missing_ori = temp_ori[non_missing_columns]

non_missing_ori = non_missing_ori.loc[non_missing_imputed_df.index, non_missing_imputed_df.columns]
mse = mean_squared_error(non_missing_ori, non_missing_imputed_df)


non_missing_imputed_df.to_csv("results/fulldata/non_missing_imputed.csv")
non_missing_ori.to_csv("results/fulldata/non_missing_ori.csv")

print2log(mse)