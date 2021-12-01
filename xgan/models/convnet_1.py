import torch
import torch.nn as nn
import numpy as np

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
        # self.holder = holder
    
    def forward(self, x):
        # Do your print / debug stuff here
        
        #print('***** Layer ******')
        #print('=== ', x.size())
        
        # # self.holder['val_3'] = reduce(lambda x, y: x*y, x.size()[1:])
        # print("******************")
        return x

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.z_shape = config['z_shape']

        self.layer1 = nn.Sequential(
            nn.Linear(np.prod(self.z_shape), 128*7*7),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            #PrintLayer(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            #PrintLayer(),
            nn.Conv2d(128, 64, kernel_size=(5, 5), padding='same'),
            #PrintLayer(),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            #PrintLayer(),
            nn.Conv2d(64, 1, kernel_size=(5, 5), padding='same'),
            nn.Sigmoid()
            #,PrintLayer()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(x.shape[0], 128, 7, 7)
        out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        return out

    def get_input_shape(self, batch_size):
        return (batch_size, ) + self.z_shape

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 5), 
                stride=(2, 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 5), 
                stride=(2, 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.layer3 = nn.Sequential(
            PrintLayer(),
            nn.Flatten(),
            PrintLayer()
            ,
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class Convnet1:
    def __init__(self):
        self.generator = Generator
        self.discriminator = Discriminator
