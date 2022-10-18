from torch import nn, optim
import torch
from torch.nn import functional as F
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self, x_dim, latent_dim=5, hdim=50, device='cpu'):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(x_dim , hdim)#.to(device)
        self.fc21 = nn.Linear(hdim, latent_dim)#.to(device)
        self.fc22 = nn.Linear(hdim, latent_dim)#.to(device)
        self.fc3 = nn.Linear(latent_dim, hdim)#.to(device)
        self.fc4 = nn.Linear(hdim, x_dim)#.to(device)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        
    def encode(self, x):
        x_ = torch.cat([x], dim=-1)#.to(self.device)
        h1 = F.relu(self.fc1(x_))#.to(device)
        return self.fc21(h1), self.fc22(h1)
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))
    
    def reparameterize(self, mu, var):
        if self.training:
            std = var.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            result = eps.mul(std).add_(mu)
            return result
        else:
            return mu
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.x_dim))
        latent = self.reparameterize(mu, logvar)
        reconstructed = self.decode(latent)
        return reconstructed, latent