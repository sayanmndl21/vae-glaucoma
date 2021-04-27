import torch
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self, in_shape, n_latent):
        super(VAE, self).__init__()
        self.in_shape = in_shape
        self.lin_shape = 512
        self.n_latent = n_latent
        c,h,w = in_shape
        self.z_dim = h//2**5 # receptive field downsampled 3 times
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),  # 32, 112, 112
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64, 56, 56
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128, 28, 28
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 128, 28, 28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, self.lin_shape, kernel_size=4, stride=2, padding=1),  # 256, 14, 14
            nn.BatchNorm2d(self.lin_shape),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(self.lin_shape * (self.z_dim)**2, n_latent),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.lin_shape * (self.z_dim)**2, n_latent),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(n_latent, self.lin_shape * (self.z_dim)**2),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm2d(self.lin_shape),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.lin_shape, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def sample_z(self, mean, logvar):
        stddev = logvar.mul(0.5).cuda()
        noise = Variable(torch.randn(*mean.size())).cuda()
        if self.training:
            return (noise * stddev) + mean
        else:
            return mean

    def bottleneck(self, h):
        mean, logvar = self.fc1(h), self.fc2(h)
        z = self.sample_z(mean, logvar)
        return z, mean, logvar

    def encode(self, x):
        x = self.encoder(x)
        z, mean, logvar = self.bottleneck(x)
        return z, mean, logvar

    def decode(self, z):
        out = self.fc3(z)
        out = out.view(out.shape[0], self.lin_shape, self.z_dim,self.z_dim)
        out = self.decoder(out)
        return out

    def forward(self, x):
        z,mean, logvar = self.encode(x)
        out = self.decode(z)
        return out, mean, logvar
