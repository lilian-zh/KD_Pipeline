'''
From https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/AT.py

'''

from __future__ import print_function

import torch.nn as nn
from .util import ConvReg


class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss
  