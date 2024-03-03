#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn

class MLP(nn.Module):
    """
    Multi layer perceptron.
    """
    def __init__(self, size_in, size_out, size_hidden=None, dropout=0.0):
        super().__init__()
        if size_hidden is None:
            size_hidden = []
        sizes = [size_in] + size_hidden + [size_out]

        net = []
        for i in range(len(sizes) - 2):
            net.append(nn.Linear(sizes[i], sizes[i+1]))
            net.append(nn.ReLU())
            net.append(nn.Dropout(dropout))

        net.append(nn.Linear(sizes[-2], sizes[-1]))
        net = nn.Sequential(*net)
        self.net = net

    def forward(self, x):
        """
        Forward method.
        """
        x = self.net(x)
        return x
