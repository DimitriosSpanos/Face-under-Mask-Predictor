"""
Neural Networks - Deep Learning
Autoencoder (predict face behind mask)
Author: Dimitrios Spanos Email: dimitrioss@ece.auth.gr
"""

import torch.nn as nn
import torch.nn.functional as F

class M1(nn.Module):

    def __init__(self,dropout_probability):
        super().__init__()

        self.p = dropout_probability
        self.e_batchnorm1 = nn.BatchNorm2d(32)
        self.e_batchnorm2 = nn.BatchNorm2d(64)
        self.e_batchnorm3 = nn.BatchNorm2d(128)
        self.e_batchnorm4 = nn.BatchNorm2d(256)


        # Encoding Phase
        self.e_conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), stride=(1,1), padding='same')
        self.e_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding='valid')
        self.e_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.e_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.e_conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.e_conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.e_conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.e_conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding='same')

        # Decoding Phase
        self.d_conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.d_conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.d_conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.d_conv4 = nn.Conv2d(in_channels=128, out_channels=164, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.d_conv5 = nn.Conv2d(in_channels=164, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.d_conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.d_conv7 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.d_conv8 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding='same')

    def forward(self, X):

        ## encoder ##
        X = F.relu(self.e_conv1(X))

        X = F.max_pool2d(X, kernel_size=(2, 2))

        X = self.e_conv2(X)
        X = self.e_batchnorm1(X)
        X = F.relu(X)

        X = F.relu(self.e_conv3(X))

        X = F.max_pool2d(X, kernel_size=(2, 2))

        X = self.e_conv4(X)
        X = self.e_batchnorm2(X)
        X = F.relu(X)

        X = self.e_conv5(X)
        X = self.e_batchnorm3(X)
        X = F.relu(X)

        X = F.relu(self.e_conv6(X))

        X = self.e_conv7(X)
        X = self.e_batchnorm4(X)
        X = F.relu(X)

        X = F.relu(self.e_conv8(X)) # encoded

        ## decoder ##

        X = F.relu(self.d_conv1(X))
        X = nn.Upsample(scale_factor=2)(X)

        X = self.d_conv2(X)
        X = nn.Dropout(p=self.p)(X)
        X = F.relu(X)

        X = F.relu(self.d_conv3(X))

        X = self.d_conv4(X)
        X = nn.Dropout(p=self.p)(X)
        X = F.relu(X)

        X = F.relu(self.d_conv5(X))
        X = nn.Upsample(scale_factor=2)(X)

        X = F.relu(self.d_conv6(X))
        X = nn.Upsample(scale_factor=2)(X)

        X = self.d_conv7(X)
        X = nn.Dropout(p=self.p)(X)
        X = F.relu(X)
        X = nn.Upsample(scale_factor=2)(X)

        X = self.d_conv8(X)

        return X




