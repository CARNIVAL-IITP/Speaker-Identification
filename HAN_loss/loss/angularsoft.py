#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/CoinCheung/pytorch-loss (MIT License)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy

class LossFunction(nn.Module):
    def __init__(self, num_out, num_class, scale=15, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.s = scale
        self.in_feats = num_out
        self.W = torch.nn.Parameter(torch.randn(num_out, num_class), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)
        self.nClasses = num_class
        print('Initialised Normalized softmax')

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
   
        
        n_rows, n_cols = costh.shape
        shifts = -1 * label.unsqueeze(-1)
        arange1 = torch.arange(n_cols).view(( 1,n_cols)).repeat((n_rows,1)).cuda()
        arange2 = (arange1 - shifts) % n_cols
        costh2 = torch.gather(costh, 1, arange2)
        
        s_p = costh2[:,0]
        s_n = costh2[:,1:].flatten()
        s_p = s_p.unsqueeze(-1) 
        s_n = s_n.unsqueeze(0).repeat(n_rows, 1) #s_p, s_n
        
        s = torch.cat([s_p, s_n], dim=-1)
        cos_sim_matrix2 = s*self.s
        
        label2 = torch.from_numpy(numpy.zeros(n_rows).astype(int)).cuda()
        
        
        costh_s = self.s * costh
        loss    = self.ce(costh_s, label)
        
        prec1   = accuracy(costh_s.detach(), label.detach(), topk=(1,))[0]
        prec2   = accuracy(cos_sim_matrix2.detach(), label2.detach(), topk=(1,))[0]
        
        return loss, prec1, prec2

