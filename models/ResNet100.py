import torch, torch.nn as nn, numpy as np,sys
from models.front_resnet import ResNet34, ResNet100,ResNet293,block2module
import models.pooling as pooling_func
import torch.nn.functional as F  
import profile 
from torchaudio import transforms


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

####################################################################
##### ResNet-Based #######################################
####################################################################

class ResNet34_based(nn.Module):
    def __init__(self, in_planes, block_type, pooling_layer, embd_dim, acoustic_dim, featCal, dropout=0,**kwargs):
        super(ResNet34_based, self).__init__()
        print('ResNet34 based model with %s block and %s pooling' %(block_type, pooling_layer))
        self.featCal = featCal
        self.front = ResNet34(in_planes, block_type)
        self.pooling = getattr(pooling_func, pooling_layer)(in_planes,acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embd_dim)
        self.drop = nn.Dropout(dropout) if dropout else None
    def forward(self, x):
        x = self.featCal(x)
        x = self.front(x.unsqueeze(dim=1))
        x = self.pooling(x)
        if self.drop:
            x = self.drop(x)
        x = self.bottleneck(x)
        return x


class ResNet100_based(nn.Module):
    def __init__(self, in_planes, block_type, pooling_layer, embd_dim, acoustic_dim, dropout=0, **kwargs):
        super(ResNet100_based, self).__init__()
        print('ResNet100 based model with %s and %s ' %(block_type, pooling_layer))
        self.featCal = transforms.MelSpectrogram(sample_rate=16000, 
                                                n_fft=512, 
                                                win_length=400, 
                                                hop_length=160, 
                                                n_mels=80)
        self.front = ResNet100(in_planes, block_type)
        block_expansion = block2module[block_type].expansion
        self.pooling = getattr(pooling_func, pooling_layer)(in_planes*block_expansion,acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embd_dim)
        self.drop = nn.Dropout(dropout) if dropout else None
        self.specaug = FbankAug()
    def forward(self, x, aug):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.featCal(x)
                x = torch.log (x + 1e-5)
                x = x-x.mean(axis=2).unsqueeze(2)
                if aug == True:
                    x = self.specaug(x)
        x = self.front(x.unsqueeze(dim=1))
        x = self.pooling(x)
        if self.drop:
            x = self.drop(x)
        x = self.bottleneck(x)
        
        return x

class ResNet293_based(nn.Module):
    def __init__(self, in_planes, block_type, pooling_layer, embd_dim, acoustic_dim, featCal, dropout=0,**kwargs):
        super(ResNet293_based, self).__init__()
        print('ResNet293 based model with %s and %s ' %(block_type, pooling_layer))
        self.featCal = featCal
        self.front = ResNet293(in_planes, block_type)
        block_expansion = block2module[block_type].expansion
        self.pooling = getattr(pooling_func, pooling_layer)(in_planes*block_expansion,acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embd_dim)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x, aug):
        x = self.featCal(x)
        x = self.front(x.unsqueeze(dim=1))
        x = self.pooling(x)
        if self.drop:
            x = self.drop(x)
        x = self.bottleneck(x)
        return x

def MainModel(num_out=256, **kwargs):
    model = ResNet100_based(in_planes=64,
                            block_type='SimAM',
                            pooling_layer='ASP',
                            embd_dim=256,
                            acoustic_dim=80)
    return model