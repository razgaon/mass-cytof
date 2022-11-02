import functions as fx
import models as mx
from torch import nn, optim
import torch
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

def train(net, train_loader, learning_rate=0.001, epochs=1, optimizer = 'SGD', device="cpu"):
    if optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr = learning_rate)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr = learning_rate, 
                              momentum = .95, nesterov = True)
    loss_list = []
    for epoch in range(epochs):
        training_loss = 0
        training_reconstruction_error = 0
        training_mmd = 0

        net.train()
        for batchnum, X in enumerate(train_loader):
            optimizer.zero_grad()
            X = X
            reconstruction, mu = net(X)
            reconstruction = reconstruction
            mu = mu
            true_samples = torch.randn(mu.shape)
            reconstruction_error, mmd = fx.vae_loss_function(inputs = X, encoded=mu, decoded=reconstruction, true_samples=true_samples , device=device)
            loss = reconstruction_error + mmd
            loss.backward()
            
            optimizer.step()

            training_reconstruction_error += reconstruction_error
            training_mmd  += mmd
            training_loss += loss

        training_reconstruction_error /= (batchnum+1)
        training_mmd  /= (batchnum+1)
        training_loss /= (batchnum+1)
        loss_list.append(loss.cpu().detach().numpy())
        print2log('Training loss for epoch %i is %.8f, Reconstruction is %.8f, mmd is %.8f'%(epoch, training_loss, training_reconstruction_error, training_mmd) )
    loss_df = pd.DataFrame(loss_list)
    return(net, loss_df)
        