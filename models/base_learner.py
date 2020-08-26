"""
Base learner
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLearner(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, prototype, feature):
