import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.z_shape = config['z_shape']

        self.layer1 = nn.Sequential(
            nn.Linear(np.prod(self.z_shape), 256*7*7),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=(3, 3), padding='same'),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 1, kernel_size=(3, 3), padding='same'),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(x.shape[0], 256, 7, 7)
        out = self.layer2(out)
        return out

    def get_input_shape(self, batch_size):
        return (batch_size, ) + self.z_shape

class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding='same'),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 5), 
                stride=(2, 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 5), 
                stride=(2, 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15)
        )
        self.layer4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class Convnet2:
    def __init__(self):
        self.generator = Generator
        self.discriminator = Discriminator
