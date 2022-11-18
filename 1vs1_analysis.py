# 화자임베딩 추출모델 만들기 및 학습된 파라미터 불러오기
import sys, time, os, argparse, socket
import numpy as np
import pdb
import torch
import glob
import zipfile
import datetime
from SpeakerNet_voice import SpeakerNet, WrappedModel, ModelTrainer
import matplotlib.pyplot as pyplot

pretrained_model = 'baseline_v1.model'

#모델 불러오기
#print("\nLoading model...")
#tstart = time.time()

s = SpeakerNet()
s = WrappedModel(s).cuda(0)
trainer = ModelTrainer(s)

trainer.loadParameters(pretrained_model);
#telapsed = time.time() - tstart
#print("Model %s loaded!"%pretrained_model);
#print('Loading time: ' +str(telapsed)+' sec')


#입력 음성 1   path1 에서 불러오기 및 임베딩 추출
path1 = './wav/sample_audio1.wav'
wav1 = loadWAV(path1, max_frames=0, num_eval=1, eval_mode=True)
embed1 = trainer.__model__.forward(x=torch.cuda.FloatTensor(wav1))

#입력 음성 2  path2 에서 불러오기 및 임베딩 추출
path2 = './wav/sample_audio2.wav'
wav2 = loadWAV(path2, max_frames=0, num_eval=1, eval_mode=True)
embed2 = trainer.__model__.forward(x=torch.cuda.FloatTensor(wav2))


#입력음성1과 입력음성2 사이의 유사도 결과
sim = torch.nn.functional.cosine_similarity(embed1, embed2).cpu().detach().numpy()[0]


#LLR 결과 계산
p_pos=np.load('p_pos.npy')
p_neg=np.load('p_neg.npy')
x= np.load('x.npy')

ind_sim = np.abs(x-sim).argmin()
pos_lk = np.sum(p_pos[0:ind_sim])*100
neg_lk = np.sum(p_neg[ind_sim:])*100

#LLR (우도비) 결과값
llr = np.log10(pos_lk/neg_lk)


#결과 LR 그래프 그리기
pyplot.fill_between(x[0:ind_sim], p_pos[0:ind_sim])
pyplot.fill_between(x[ind_sim:], p_neg[ind_sim:])
pyplot.plot(x, p_pos, label='P(E|H0)')
pyplot.plot(x, p_neg, label='P(E|H1)')
pyplot.grid(True)
pyplot.legend()

#그래프 출력 저장
pyplot.savefig('LRgraph.png')