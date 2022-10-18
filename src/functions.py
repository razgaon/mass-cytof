from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import torch


class MyDataset(Dataset):
 
    def __init__(self,df):
        x=df.values
        self.x_train=torch.tensor(x,dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)
   
    def __getitem__(self,idx):
        return self.x_train[idx]

class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
def compute_kernel(x, y, device):
    x_size = x.to(device).size(0)
    y_size = y.to(device).size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    tiled_x = x.expand(x_size, y_size, dim).to(device)
    tiled_y = y.expand(x_size, y_size, dim).to(device)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input)

def compute_mmd(x, y, device):
    x_kernel = compute_kernel(x, x, device)
    y_kernel = compute_kernel(y, y, device)
    xy_kernel = compute_kernel(x, y, device)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def vae_loss_function(inputs, encoded, decoded, true_samples, device):
    #reconstruction loss - decoding
    nll = (decoded.to(device) - inputs.to(device)).pow(2).mean()
    #distance + reconstruction loss
    mmd = compute_mmd(true_samples, encoded.to(device), device)
    return(nll, mmd)