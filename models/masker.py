# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class Masker(nn.Module):
    def __init__(self, D_in, D_out, init_value):
        """

        """
        super(Masker, self).__init__()
        self.mask = nn.Conv2d(D_in, D_out, 1, 1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.mask(x)