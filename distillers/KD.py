'''
From https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/AT.py

'''

from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        # change the 'size_average=False' to 'reduction='sum'，
        # to address UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0] 
        return loss

