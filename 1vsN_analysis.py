import os, torch
import numpy as np
from SpeakerNet_voice_common_cpu import SpeakerNet, WrappedModel, ModelTrainer
import matplotlib.pyplot as plt
from DatasetLoader import loadWAV, loadWAV2
from sklearn.metrics.pairwise import cosine_similarity


##===========화자 임베딩 모델 불러오기=========##
#딥러닝 모델 구축 및 Class로 Wrapping
s = SpeakerNet()
s = WrappedModel(s).cuda(0)
trainer = ModelTrainer(s)

#딥러닝 모델에 학습된 파라미터 불러오기
pretrained_model = 'baseline_v1.model'
trainer.loadParameters(pretrained_model);
##=============================================##

##===========음성으로부터 화자 임베딩 추출==========================================##
#입력 음성 1   path1 에서 불러오기 
path1 = './wav/sample_audio1.wav'
wav1 = loadWAV2(path1, max_sec=0)
#딥러닝 모델로부터 임베딩 추출
with torch.no_grad():
    embed1 = trainer.__model__.forward(x=torch.cuda.FloatTensor(wav1)).cpu().detach().numpy()

#path2(폴더) 안에 있는 확장명이 wav인 음성파일경로 리스트 생성 
path2 = './wav/'
wav_list=[]
file_list = []
for (path, dir, files) in os.walk(path2):
        for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.wav':
                        wav_list.append(path+"/"+filename)
                        file_list.append(filename)
     
    
#해당 경로 리스트의 음성들로부터 임베딩 추출
embed_list = []
for wav in wav_list:
    wav_tmp = loadWAV2(wav, max_sec=0)
    with torch.no_grad():
        embed_tmp = trainer.__model__.forward(x=torch.cuda.FloatTensor(wav_tmp))
    embed_list.append(embed_tmp.squeeze().cpu().detach().numpy())
embed_list = np.array(embed_list)
file_list = np.array(file_list)
##==================================================================================##


##==================1vsN 유사도 결과 계산===========================================##
sims = cosine_similarity(embed_list, embed1).reshape(-1)
##==================================================================================##

#=========K개 고르기=================#
#default 값을 0으로 고르고, 0<K<N 값일때만 필터링

K = 0
if (K>0 &K<len(wav_list)):
    indices = [np.flip(np.argsort(sims))[0:K]]
    file_list = file_list[indices]
    sims = sims[indices]
else:
    indices = [np.flip(np.argsort(sims))]
    file_list = file_list[indices]
    sims = sims[indices]
#===================================#

#=========유사도 제한값으로 필터링하기==========#
#default 값은 -1 이 좋을 거 같습니다.
sim_limit = 0

filter_list = []
for sim in sims:
    if sim >= sim_limit:
        filter_list.append(True)
    else:
        filter_list.append(False)

sims = sims[filter_list]    
file_list = file_list[filter_list]
#+===============================================#

##=====결과 막대 그래프 그리기==========================##
x = np.arange(len(file_list))
plt.bar(x, sims)
plt.xticks(x, file_list)
#그래프 출력 저장
plt.savefig('1vsN_result.png')
##======================================================##