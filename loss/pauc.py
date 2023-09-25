#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy

class LossFunction(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()
        self.m = 0.4
        print('Initialised AngleProto')

    def forward(self, x, label=None):

        assert x.size()[1] >= 2

        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        stepsize        = out_anchor.size()[0]

        cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w, 1e-6)

        
        s_p = cos_sim_matrix.diagonal()  # step size 
        s_n = cos_sim_matrix.flatten()[1:].view(stepsize-1, stepsize+1)[:,:-1].flatten() #stepsize*stepsize-1
        
        b_p = s_p.size()[0]
        b_n = s_n.size()[0]
        
        s_p = s_p.unsqueeze(-1)
        s_n = s_n.unsqueeze(0).repeat(b_p, 1)
        
        s = torch.cat([s_p, s_n], dim=-1)
        
        
        s_p = s_p.repeat(1, b_n) #b_p, b_n     

        
        
        
        label2   = torch.from_numpy(numpy.zeros(stepsize).astype(int)).cuda()
        prec1   = torch.tensor(numpy.mean((s_p>s_n).cpu().numpy()))*100
        prec2   = accuracy(s.detach(), label2.detach(), topk=(1,))[0]
        nloss = -torch.mean(torch.clamp((s_p-s_n)-self.m, max=0))
        return nloss, prec1, prec2
