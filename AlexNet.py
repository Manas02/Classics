#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AlexNet Paper implementation in PyTorch for ImageNet Dataset.
[https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf]
[https://arxiv.org/pdf/1404.5997.pdf]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_data
from tqdm import trange


# Hyperparameter
SEED      = 0
LR        = 1e-3
EPOCH     = 100
STEP_SIZE = 50
GAMMA     = 0.1


# AlexNet Model
class AlexNet(nn.Module):
    def __init__(self, n_classes: int = 1000, dropout: float = 0.5) -> None:
        super(AlexNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes)
        )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.model(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x


# Train-Test Functions
def train(model, device, train_loader, optimizer, epoch):
    """ Train Function
    
    Args:
        model        : AlexNet class instance.
        device       : CUDA or CPU for training.
        train_loader : DataLoader instance for training dataset. 
        optimizer    : Adadelta instance.
        epoch        : Number of epochs for training.
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print(f'Train Epoch: {epoch} [{batch_idx * len(data):>4d}/{len(train_loader.dataset)}] | Loss: {loss.item():.4f}')


def test(model, device, test_loader):
    """ Test Function
    
    Args:
        model        : AlexNet class instance.
        device       : CUDA or CPU for testing. 
        test_loader  : DataLoader instance for test dataset. 
    """ 
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f} | Accuracy: {correct}/{len(test_loader.dataset)}')


def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std  = [0.229, 0.224, 0.225])
        ])
    
    train_loader = torch.utils.data.DataLoader(get_data('train'), transform=transform)
    test_loader  = torch.utils.data.DataLoader(get_data('test'),  transform=transform)

    model = AlexNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr = LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = STEP_SIZE, gamma = GAMMA)
    
    for epoch in trange(1, EPOCH + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), "AlexNet.pt")


if __name__ == '__main__':
    main()
