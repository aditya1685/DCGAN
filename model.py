# model.py

import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, in_features, img_dimension):
        super().__init__()
        self.disc = nn.Sequential(
            self.block(in_features, img_dimension,4,2, 1),
        self.block(img_dimension, img_dimension*2,4,2,1),
        self.block(img_dimension*2, img_dimension*4, 4,2,1),
        self.block(img_dimension*4,img_dimension*8, 4, 2, 1),
        nn.Conv2d(img_dimension*8, 1, kernel_size=4, stride=2, padding=0),
        nn.Sigmoid()
        )
    def block(self,
             in_channels,
             out_channels,
             kernel_size,
             stride,
             padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
        )
    def forward(self,x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim,img_channels, img_dimension):
        super().__init__()
        self.gen = nn.Sequential(
            self.block(z_dim, img_dimension*16,4,2,0),
            self.block(img_dimension*16, img_dimension*8,4,2,1),
            self.block(img_dimension*8, img_dimension*4,4,2,1),
            self.block(img_dimension*4, img_dimension*2,4,2,1),
            nn.ConvTranspose2d(img_dimension*2, img_channels,kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self,x):
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        