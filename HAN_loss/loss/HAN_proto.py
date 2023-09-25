import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy
class LossFunction(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, m=0.3, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()
        self.m = m
        print('Initialised HAN Prototypicial loss')

    def forward(self, x, label=None):

        assert x.size()[1] >= 2

        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        stepsize        = out_anchor.size()[0]

        cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))

        s_p = cos_sim_matrix.diagonal()  # step size 
        s_n = cos_sim_matrix.flatten()[1:].view(stepsize-1, stepsize+1)[:,:-1].flatten() #stepsize*stepsize-1
        s_p = s_p.unsqueeze(-1) #s_p, 1
        s_n = s_n.unsqueeze(0).repeat(s_p.size()[0], 1) #s_p, s_n

        s = torch.cat([s_p, s_n], dim=-1)
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix2 = s * self.w + self.b


        label   = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda()
        label2   = torch.from_numpy(numpy.zeros(stepsize).astype(int)).cuda()
        nloss   = self.criterion(cos_sim_matrix2, label2)
        prec1   = accuracy(cos_sim_matrix.detach(), label.detach(), topk=(1,))[0]
        prec2   = accuracy(cos_sim_matrix2.detach(), label2.detach(), topk=(1,))[0]
        
        return nloss, prec1, prec2
