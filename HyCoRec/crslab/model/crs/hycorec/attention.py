# -*- coding: utf-8 -*-
# @Time   : 2021/4/1
# @Author : Chenzhan Shang
# @Email  : czshang@outlook.com

import torch
import torch.nn as nn
import torch.nn.functional as F


# Multi-head
class MHItemAttention(nn.Module):
    def __init__(self, dim, head_num):
        super(MHItemAttention, self).__init__()
        self.MHA = torch.nn.MultiheadAttention(dim, head_num, batch_first=True)
        self.keep_weights = False  # S6/S7 analysis hook: store head-averaged weights of the last call
        self.last_weights = None

    def forward(self, related_entity, context_entity):
        """
            input:
                related_entity: (n_r, dim)
                context_entity: (n_c, dim)
            output:
                related_context_entity: (n_c, dim)
        """
        context_entity = torch.unsqueeze(context_entity, 0)
        related_entity = torch.unsqueeze(related_entity, 0)
        output, w = self.MHA(context_entity, related_entity, related_entity, need_weights=self.keep_weights)
        if self.keep_weights:
            self.last_weights = w.squeeze(0).detach()  # (n_c, n_r)
        return torch.squeeze(output, 0)
