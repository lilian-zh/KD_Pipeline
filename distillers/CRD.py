'''
From https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/AT.py

'''

from torch import nn

from .util import NCEAverage, NCECriterion
from .util import Embed


class NCELoss(nn.Module):
    """NCE contrastive loss"""
    def __init__(self, n_data, feat_dim, nce_k, nce_t, nce_m, **kwargs):
        # docstring
        """
        Initialize NCELoss.

        Args:
            feat_dim (int): Dimension of feature vectors.
            nce_k (int): Number of negative samples.
            nce_t (float): Temperature for the softmax.
            nce_m (float): Momentum for updating the memory bank.
        """
        super(NCELoss, self).__init__()
        self.contrast = NCEAverage(feat_dim, n_data, nce_k, nce_t, nce_m)
        self.criterion_t = NCECriterion(n_data)
        self.criterion_s = NCECriterion(n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Forward pass of NCELoss.

        Args:
            f_s: Feature tensor of the student model.
            f_t: Feature tensor of the teacher model.
            idx: Index tensor.
            contrast_idx: Index tensor for contrastive samples.

        Returns:
            loss: NCE loss.
        """
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        #print("NCE_loss:",loss)
        return loss
