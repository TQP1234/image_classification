import torch.nn as nn
from collections import OrderedDict


class SimpleConvNet(nn.Module):
    def __init__(self, input_channel, input_size, output_size):
        super().__init__()

        r = 1

        # convolutional layer formula
        # ((input_size - kernel_size + 2*padding) / stride) + 1

        self.ConvBlock1 = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(input_channel, 32, 3, 1, 1)),
                ('leaky1', nn.LeakyReLU(inplace=True)),
                ('pool1', nn.MaxPool2d(2, 2)),
                ('conv2', nn.Conv2d(32, 64, 3, 1, 1)),
                ('leaky2', nn.LeakyReLU(inplace=True)),
                ('pool2', nn.MaxPool2d(2, 2)),
                ('conv3', nn.Conv2d(64, 128, 3, 1, 1)),
                ('leaky3', nn.LeakyReLU(inplace=True)),
                ('pool3', nn.MaxPool2d(2, 2)),
                ('conv4', nn.Conv2d(128, 256, 3, 1, 1)),
                ('leaky4', nn.LeakyReLU(inplace=True))
            ])
        )

        self.SqueezeAndExcitation1 = nn.Sequential(
            OrderedDict([
                ('globalpool1', nn.AvgPool2d(input_size//8, 1)),  # global average pooling
                ('flat1', nn.Flatten()),  # convert the dimension from (B, C, 1, 1) to (B, C)
                ('fc1', nn.Linear(256, 256//r)),
                ('relu1', nn.ReLU(inplace=True)),
                ('fc2', nn.Linear(256//r, 256)),
                ('sigmoid1', nn.Sigmoid())
            ])
        )

        self.ConvBlock2 = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(256, 512, 3, 1, 1)),
                ('leaky1', nn.LeakyReLU(inplace=True)),
                ('pool1', nn.MaxPool2d(2, 2)),
                ('conv2', nn.Conv2d(512, 1024, 3, 1, 1)),
                ('leaky2', nn.LeakyReLU(inplace=True)),
                ('pool2', nn.MaxPool2d(2, 2))
            ])
        )

        self.SqueezeAndExcitation2 = nn.Sequential(
            OrderedDict([
                ('globalpool1', nn.AvgPool2d(input_size//32, 1)),  # global average pooling
                ('flat1', nn.Flatten()),  # convert the dimension from (B, C, 1, 1) to (B, C)
                ('fc1', nn.Linear(1024, 1024//r)),
                ('relu1', nn.ReLU(inplace=True)),
                ('fc2', nn.Linear(1024//r, 1024)),
                ('sigmoid1', nn.Sigmoid())
            ])
        )

        self.ConvBlock3 = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(1024, 1024, 3, 1, 1)),
                ('leaky1', nn.LeakyReLU(inplace=True))
            ])
        )

        # Taking the average value of each feature maps (reducing the dimension to B, C, 1, 1)
        # Then convert the dimension to (B, C) and feed it in to the dense layer
        # This is an alternative way to pure flattening
        # It reduces computational cost as well
        # Found that it achieved higher accuracy than just pure flattening
        self.AverageGlobalPooling = nn.Sequential(
            OrderedDict([
                ('globalpool1', nn.AvgPool2d(input_size//32, 1)),  # global average pooling
                ('flat1', nn.Flatten())
            ])
        )

        self.DenseLayer = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(1024, 1024)),
                ('drop1', nn.Dropout(0.5, inplace=True)),
                ('leaky1', nn.LeakyReLU(inplace=True)),
                ('out', nn.Linear(1024, output_size)),
                ('softmax', nn.LogSoftmax(dim=1))
            ])
        )

    def forward(self, X):
        # Conv block 1
        X = self.ConvBlock1(X)

        # Squeeze and Excitation Layer 1
        S = self.SqueezeAndExcitation1(X).view(-1, 256, 1, 1)
        # Scaling
        X = X * S.expand_as(X)

        # Conv block 2
        X = self.ConvBlock2(X)

        # Squeeze and Excitation Layer 2
        S = self.SqueezeAndExcitation2(X).view(-1, 1024, 1, 1)
        # Scaling
        X = X * S.expand_as(X)

        # Conv block 3
        X = self.ConvBlock3(X)

        # Dense layer
        X = self.AverageGlobalPooling(X)
        X = self.DenseLayer(X)

        return X
