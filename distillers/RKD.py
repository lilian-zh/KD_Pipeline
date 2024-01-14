'''
From https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/AT.py

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""
    def __init__(self, w_d=25, w_a=50):
        super(RKDLoss, self).__init__()
        # w_d and w_a represent the weights for distance loss and angle loss, respectively.
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        # Flatten the input student features f_s and teacher features f_t for subsequent calculations.
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)

        # RKD distance loss
        with torch.no_grad():
            # t_d represents the Euclidean distance (pairwise distance) between teacher features.
            t_d = self.pdist(teacher, squared=False)
            # Calculate the mean teacher feature distance mean_td.
            mean_td = t_d[t_d > 0].mean()
            # Normalize t_d (divide by mean distance).
            t_d = t_d / mean_td

        # Calculate the distance d between student features and normalize.
        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        # Use Smooth L1 Loss to compute distance loss loss_d, making student feature distances approach teacher feature distances.
        loss_d = F.smooth_l1_loss(d, t_d)

        # RKD Angle loss
        # t_angle represents the cosine similarity between teacher features' angles, obtained by normalizing features and computing dot products.
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        # Calculate the cosine similarity s_angle between student features' angles.
        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        # Use Smooth L1 Loss to compute angle loss loss_a, making student feature angles approach teacher feature angles.
        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        # Combine distance and angle losses with respective weights.
        loss = self.w_d * loss_d + self.w_a * loss_a

        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        # Compute the inner product of features.
        prod = e @ e.t()
        # Compute the distance matrix based on the definition of Euclidean distance.
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        # If squared is True, return the squared distances instead of the distances themselves.
        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res
