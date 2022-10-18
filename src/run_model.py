import torch
import os 
import time

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
import logging

import multiprocessing


os.environ['CUDA_LAUNCH_BLOCKING'] = "0,1,2,3"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info
print2log(torch.cuda.device_count())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print2log(device)

# torch.set_num_threads(multiprocessing.cpu_count())
# torch.set_num_interop_threads(multiprocessing.cpu_count()) 

ori_data = pd.read_csv("../data/processed/mitsialis_bengsch_rahman.csv", index_col="id")
ori_data = ori_data.sample(frac=1)
# data = ori_data.sample(100)
data = ori_data
print2log("Data loaded")

excluded_markers = ['pt', "source"]
data_markers = [col for col in data.columns if col not in excluded_markers]

scaler = MinMaxScaler()
data[data_markers] = scaler.fit_transform(data[data_markers])
data

data = data[~data.index.duplicated(keep='first')]
data.shape
print2log("Data processed.")

dataset = fx.MyDataset(data[data_markers])
net = mx.VAE(x_dim=dataset.x_train.shape[1], device=device)
net = fx.MyDataParallel(net, device_ids=[0, 1, 2, 3])
net.to(device)

print2log("0 workers, batch 1000")
train_loader = DataLoader(dataset,shuffle=False,num_workers=0,batch_size=1000)
print2log("Training")

since = time.time()
base = time.time() - since
model, loss = rx.train(net, learning_rate = .0001, epochs = 100,
      train_loader = train_loader, device=device)
cost = time.time() - since
print2log("Model RunTime: " + str(cost) + str(cost/base))

loss.to_pickle('results/fulldata/loss.pt')
torch.save(model, 'results/fulldata/model_state')
    
r, latent = model(dataset.x_train)
latent_df = pd.DataFrame(latent.detach().cpu().numpy(), index = data.index)
latent_df['source'] = data.source
latent_df.to_pickle('results/fulldata/latent_nonpd1.pt')

x_train = pd.DataFrame(dataset.x_train.detach().cpu().numpy(), index = data.index, columns=data_markers)
x_train['source'] = data.source
x_train.to_pickle('results/fulldata/x_train.pt')

