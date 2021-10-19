import numpy as np
from torch import nn


class Generator(nn.Module):
    def __init__(self, z = 100):
        super(Generator, self).__init__()

        self.image_shape = (1,28,28)
        self.model = nn.Sequential(
            nn.Linear(z,128,bias=True),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(128,256,bias=True),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,512,bias=True),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1024,bias=True),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(1024,int(np.prod(self.image_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.image_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self,model_type):
        super(Discriminator, self).__init__()

        self.image_shape = (1,28,28)
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        output = self.models(img_flat)
        return output