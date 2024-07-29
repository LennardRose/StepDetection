""" By Lennard Rose 5112737"""

import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, activity):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=5, padding=2)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=1600, out_features=800)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        if not activity:
            self.linear2 = nn.Linear(in_features=800, out_features=1)
        else:
            self.linear2 = nn.Linear(in_features=800, out_features=4)

        self.activity = activity
        self.sigmoid = nn.Sigmoid()

        # initialize weights
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_()
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)

        if not self.activity:
            x = x.squeeze()
            x = self.sigmoid(x)
        else:
            # crossentrophyloss already includes softmax
            x = x.type(torch.float)

        return x