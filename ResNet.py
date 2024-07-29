""" By Lennard Rose 5112737"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()

        self.Block = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )


    def forward(self, x):
        return self.Block(x)


class ResNet(nn.Module):

    def __init__(self, in_dim, out1_dim, out2_dim, final_out_dim, activity):
        super(ResNet, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.ResBlock1 = ResBlock(out1_dim, out1_dim)
        self.ResBlock2 = ResBlock(out2_dim, out2_dim)
        self.fc1 = nn.Linear(in_dim, out1_dim)
        self.fc2 = nn.Linear(out1_dim, out2_dim)
        self.fc3 = nn.Linear(out2_dim, final_out_dim)
        self.bn1 = nn.BatchNorm1d(out1_dim)
        self.bn2 = nn.BatchNorm1d(out2_dim)

        self.activity = activity
        self.sigmoid = torch.nn.Sigmoid()

        # initialize weights
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_()
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, x):

        x1 = self.fc1(x)
        x2 = self.ResBlock1(x1)
        x3 = x1 + x2

        x4 = self.bn1(x3)
        x5 = F.relu(x4)

        x6 = self.fc2(x5)
        x7 = self.ResBlock2(x6)
        x8 = x6 + x7

        x9 = self.bn2(x8)
        x10 = F.relu(x9)

        x = self.fc3(x10)

        if not self.activity:
            x = x.squeeze()
            x = self.sigmoid(x)

        return x
