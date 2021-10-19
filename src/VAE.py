#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Auto-Encoding Variational Bayes ~ Kingma and Welling, ICLR, 2014
[https://arxiv.org/abs/1312.6114]
"""

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Hyperparameters
SEED       = 0
EPOCH      = 100
BATCH_SIZE = 32
LR         = 1e-3

# Variational AutoEncoder
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, 784)
        )

    def encode(self, x):
        x = nn.ReLU(nn.Linear(784, 400)(x))
        x = self.encode(x)
        return nn.Linear(400, 20)(x), nn.Linear(400, 20)(x)

    def reparameterize(self, μ, logvar):
        σ = torch.exp(0.5*logvar)
        ε = torch.randn_like(σ)
        return μ + ε*σ

    def forward(self, x):
        μ, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(μ, logvar)
        z = self.decode(z)
        z = torch.sigmoid(z)
        return z, μ, logvar


def loss_function(recon_x, x, μ, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - μ.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for data, _ in trainloader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, μ, logvar = model(data)
        loss = loss_function(recon_batch, data, μ, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(F'Epoch: {epoch} | Average loss: {train_loss / len(trainloader.dataset):.4f}')


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(testloader):
            data = data.to(device)
            recon_batch, μ, logvar = model(data)
            test_loss += loss_function(recon_batch, data, μ, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'output/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(testloader.dataset)
    print(f'Test set loss: {test_loss:.4f}')


if __name__ == "__main__":
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.ToTensor()),
                        batch_size = BATCH_SIZE, shuffle=True)

    testloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, 
                        transform=transforms.ToTensor()),
                        batch_size = BATCH_SIZE, shuffle=True)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCH + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'output/sample_' + str(epoch) + '.png')
