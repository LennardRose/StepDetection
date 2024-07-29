""" By Lennard Rose 5112737"""

import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_classes, activity, output_element=None):
        super(RNN, self).__init__()
        self.activity = activity
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          nonlinearity="relu",  # tanh/relu
                          batch_first=True)
        self.linear = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.output_element = output_element


    def forward(self, x):
        x, hidden_ = self.rnn(x)

        if self.activity:
            if self.output_element == "first":
                x = x[:, 0, :]
            elif self.output_element == "last":
                x = x[:, -1, :]

        x = self.linear(x)

        if not self.activity:
            x = x.squeeze()
            x = self.sigmoid(x)

        return x


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_classes, activity, output_element=None):
        super(LSTM, self).__init__()
        self.activity = activity
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.linear_1 = nn.Linear(in_features=hidden_size, out_features=100)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(in_features=100, out_features=n_classes)
        self.sigmoid = nn.Sigmoid()
        self.output_element = output_element


    def forward(self, x):

        x, hidden_ = self.lstm(x)
        if self.activity:
            if self.output_element == "first":
                x = x[:, 0, :]
            elif self.output_element == "last":
                x = x[:, -1, :]

        x = self.dropout(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        if not self.activity:
            x = x.squeeze()
            x = self.sigmoid(x)

        return x  # .argmax(dim=1)
