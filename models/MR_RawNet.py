# -*- encoding: utf-8 -*-

import math
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from asteroid_filterbanks import Encoder, ParamSincFB

 

class GateModule(nn.Module):
	def __init__(self, channels, bottleneck=128, nb_input=3):
		super(GateModule, self).__init__()
		self.nb_input = nb_input
		self.aap = nn.AdaptiveAvgPool1d(1)
		self.attention = nn.Sequential(
			nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
			nn.ReLU(),
			nn.BatchNorm1d(bottleneck),
			nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
			nn.Softmax(dim = -1),
			)

	def forward(self, input):	
		x = self.aap(input).reshape(input.size(0),-1,self.nb_input)
		x = self.attention(x)
		
		output = None
		for i in range(self.nb_input):
			aw = x[:,:,i].unsqueeze(-1)
			if output == None: output = aw * input[:, x.size(1) * i : x.size(1) * (i+1)]
			else: output += aw * input[:, x.size(1) * i : x.size(1) * (i+1)]
		return output

class AFMS(nn.Module):
    """
    Alpha-Feature map scaling, added to the output of each residual block[1,2].

    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    [2] AMFS    : https://www.koreascience.or.kr/article/JAKO202029757857763.page
    """

    def __init__(self, nb_dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((nb_dim, 1)))
        self.fc = nn.Linear(nb_dim, nb_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        y = self.sig(self.fc(y)).view(x.size(0), x.size(1), -1)

        x = x + self.alpha
        x = x * y
        return x

class SpeedPerturbation(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_len = 320
        self.speed_perturbation = torchaudio.transforms.SpeedPerturbation(
            orig_freq = 16000, 
            factors = [0.9, 1.1, 1.0]) # 66%
    def forward(self, x):    
        x = self.speed_perturbation(x)[0].squeeze(1)
        x_len = x.size(-1)
        if x_len % self.add_len != 0:
            x = x[:, :-(x_len % self.add_len)]
        return x   
        
class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97) -> None:
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert (
            len(input.size()) == 2
        ), "The number of dimensions of input tensor must be 2!"
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), "reflect")
        return F.conv1d(input, self.flipped_filter)

class MRA(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, dilation, scale):
        super(MRA, self).__init__()
        self.block = Bottle2neck(inplanes, planes, kernel_size, dilation, scale)
        self.block_up = nn.Sequential(
            nn.ConvTranspose1d(inplanes, inplanes, kernel_size=2, stride=2, padding=0),
            Bottle2neck(inplanes, planes, kernel_size, dilation, scale),
            nn.AvgPool1d(2, stride=2),
        )
        self.block_down = nn.Sequential(
            nn.AvgPool1d(2, stride=2),
            Bottle2neck(inplanes, planes, kernel_size, dilation, scale),
            nn.ConvTranspose1d(planes, planes, kernel_size=2, stride=2, padding=0),
        )

        self.gate_moduel = GateModule(planes, planes//2, nb_input=3)


    def forward(self, x):
        out = self.block(x)
        out_up = self.block_up(x)
        out_down = self.block_down(x)
        x = x + self.gate_moduel(torch.cat((out, out_up, out_down), 1))

        return x
    
class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, dilation, scale):
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1

        bns = []
        convs = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for _ in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width

        self.afms = AFMS(planes)
 
        if inplanes != planes:  # if change in number of filters
            self.residual = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
            )
        else:
            self.residual = nn.Identity()


    def forward(self, x):
        residual = self.residual(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x_split = torch.split(x, self.width, 1)
        for i in range(self.nums):
            sp = x_split[i] if i == 0 else sp + x_split[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            x = sp if i == 0 else torch.cat((x, sp), 1)
        x = torch.cat((x, x_split[self.nums]), 1)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        
        x += residual
        x = self.afms(x)

        return x 



class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + 1e-8, 0.5) + self.beta
        return gLN_y
       

    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels,
                                   bias=False)

        prelu = nn.PReLU()
        norm = GlobalLayerNorm(in_channels)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # Put together
        self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

    def forward(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        return self.net(x)
    
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation):
        super(TemporalBlock, self).__init__()
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = GlobalLayerNorm(out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
                                        stride, padding, dilation)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        residual = x
        out = self.net(x)
        # TODO: when P = 3 here works fine, but when P = 2 maybe need to pad?
        return out + residual  # look like w/o F.relu is better than w/ F.relu
        # return F.relu(out + residual)
    
class MR_RawNet(nn.Module):
    def __init__(self, C=256, **kwargs):
        super(MR_RawNet, self).__init__()
        embedding_size = kwargs["nOut"]

        self.speed_perturbation = SpeedPerturbation()
        self.preprocess = nn.Sequential(
            PreEmphasis(), nn.InstanceNorm1d(1, eps=1e-4, affine=True)
        )

        F1 = 256
        K = 50
        M = 16
        self.n = 4
        for i in range(self.n): 
            conv1_padding = torch.nn.ConstantPad1d(padding=K * (2 ** i) // 2, value=0)
            conv1 = Encoder(
                ParamSincFB(
                    n_filters = F1,
                    kernel_size = K * (2 ** i) + 1,
                    stride =  K * (2 ** i) * 2 // 5 #20
                )
            )
            F2 = F1//2 
            conv_1x1 = nn.Conv1d(F1, F2, 1, bias=False)
            repeats = []
            R = 3 # R: Number of repeats
            X = 4 # X: Number of convolutional blocks in each repeat
            for _ in range(R):
                blocks = []
                for x in range(X):
                    dilation = 2**x
                    kernel_size = 3
                    blocks += [TemporalBlock(
                        in_channels=F2, # B: Number of channels in bottleneck 1 × 1-conv block
                        out_channels=F2*3//4, # H: Number of channels in convolutional blocks
                        kernel_size=kernel_size,
                        stride=1,
                        padding=(kernel_size-1)*dilation//2,
                        dilation=dilation)]
                repeats += [nn.Sequential(*blocks)]
            temporal_conv_net = nn.Sequential(*repeats)
            conv2 = nn.Conv1d(
                in_channels= F2,
                out_channels= F2,
                kernel_size= M // (2 ** i) + 1 ,
                stride = M // (2 ** (i+1)),
                padding= M // (2 ** (i+1)))
            bn2 = nn.BatchNorm1d(F2)

            setattr(self, '{}{:d}'.format("conv1_padding", i), conv1_padding)
            setattr(self, '{}{:d}'.format("conv1", i), conv1)
            setattr(self, '{}{:d}'.format("conv_1x1", i), conv_1x1)
            setattr(self, '{}{:d}'.format("tcn_block", i), temporal_conv_net)
            setattr(self, '{}{:d}'.format("conv2_relu_bn", i), nn.Sequential(conv2, nn.ReLU(), bn2))
            
        F = self.n * F2
        self.conv3 = nn.Conv1d(F, C, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(C)
        self.relu = nn.ReLU()

        self.depths = [3, 3, 3]
        self.dilations = [2, 2, 2]
        for i in range(len(self.depths)):
            stage = nn.Sequential(
                *[MRA(inplanes=C, planes=C, kernel_size=3, dilation=self.dilations[i], scale=4)
                   for _ in range(self.depths[i])]
            )
            setattr(self, '{}{:d}'.format("stage", i), stage)
            
        attention_dim = 1536
        self.layer4 = nn.Conv1d(3 * C, attention_dim, kernel_size=1)

        self.attention = nn.Sequential(
            nn.Conv1d(3 * attention_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, attention_dim, kernel_size=1),
            nn.Softmax(dim=2)
        )
        self.bn5 = nn.BatchNorm1d(2 * attention_dim)
        self.fc6 = nn.Linear(2 * attention_dim, embedding_size)
        self.bn6 = nn.BatchNorm1d(embedding_size)

        self.mp2 = nn.MaxPool1d(2)

                   
    def forward(self, input, label=None):
        # input shape: [B, T(48000)]
        with torch.no_grad():
            if label is not None:
                input = self.speed_perturbation(input)
            input = self.preprocess(input)

        for i in range(self.n):
            with torch.cuda.amp.autocast(enabled=False):
                x = getattr(self, '{}{:d}'.format("conv1_padding", i))(input)
                x = torch.abs(getattr(self, '{}{:d}'.format("conv1", i))(x))
                x = torch.log(x + 1e-6)
                x = x - torch.mean(x, dim=-1, keepdim=True)

            x = getattr(self, '{}{:d}'.format("conv_1x1", i))(x)
            if i != 0:
                x = x + _x
            x = getattr(self, '{}{:d}'.format("tcn_block", i))(x)
            if i != self.n - 1:
                _x = self.mp2(x)
            x = getattr(self, '{}{:d}'.format("conv2_relu_bn", i))(x)
            z = x if i == 0 else torch.cat((z, x), dim=1)

        x = self.conv3(z)
        x = self.relu(x)
        x = self.bn3(x)
        
        x1 = getattr(self, '{}{:d}'.format("stage", 0))(x)
        x2 = getattr(self, '{}{:d}'.format("stage", 1))(x1)
        x3 = getattr(self, '{}{:d}'.format("stage", 2))(x1 + x2)

        # concat shape: [B, 3*C, T]
        x = torch.cat((x1, x2, x3), dim=1)
        
        x = self.layer4(x)
        x = self.relu(x)

        # x shape: [B, E, T]
        time = x.size()[-1]
        temp1 = torch.mean(x, dim=2, keepdim=True).repeat(1, 1, time) # t1 shape: [B, E, T]
        temp2 = torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4, max=1e4)).repeat(1, 1, time) # t2 shape: [B, E, T]

        # gx shape: [B, 3*E, T]
        gx = torch.cat((x, temp1, temp2), dim=1)
        # w shape: [B, E, T]
        w = self.attention(gx)
        # mu shape: [B, E]
        mu = torch.sum(x * w, dim=2)
        # sg shape: [B, E]
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4, max=1e4))
        # x shape: [B, 2*E]
        x = torch.cat((mu, sg), 1)

        # x shape: [B, 2*E]
        x = torch.cat((mu, sg), 1)

        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x


def MainModel(**kwargs):

    model = MR_RawNet(**kwargs)
    return model