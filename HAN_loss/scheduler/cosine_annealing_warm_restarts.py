#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, lr_t0, lr_tmul, lr_eta_min, **kwargs):

    sche_fn = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=lr_t0, T_mult=lr_tmul, eta_min=lr_eta_min)
    #lr_step = 'epoch'
    print('Initialised CosineAnnealingWarmRestarts scheduler')
    #return sche_fn, lr_step
    return sche_fn
