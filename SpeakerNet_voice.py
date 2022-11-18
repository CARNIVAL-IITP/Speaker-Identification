#!/usr/bin/python
#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, pdb, sys, random, time, os, itertools, shutil, importlib
import numpy as np



class SpeakerNet(nn.Module):
    def __init__(self, model="ECAPA_SKN_TDNN4", num_out=192, eca_c=1024, eca_s=8, pooling_type="ASP", **kwargs):
        super(SpeakerNet, self).__init__()
        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        argsdict={'num_out':num_out, 'eca_c':eca_c, 'eca_s':eca_s, 'pooling_type':pooling_type}
        self.__S__ = SpeakerNetModel(**argsdict)


    def forward(self, data):
        self.eval()
        return self.__S__.forward(data.reshape(-1,data.size()[-1]).cuda(), aug=False) # from exp07

class ModelTrainer(object):
    def __init__(self, speaker_model, gpu=0, **kwargs):
        self.__model__  = speaker_model
        self.gpu = gpu
        
    def loadParameters(self, path):
        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d"%self.gpu)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
    def saveParameters(self, path):
        torch.save(self.__model__.module.state_dict(), path)
        
class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x)
