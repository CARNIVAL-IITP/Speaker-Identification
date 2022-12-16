# Speaker-Identification

Carnival system을 구성하는 Speech source separation 모델입니다. 과학기술통신부 재원으로 정보통신기획평가원(IITP) 지원을 받아 수행한 "원격 다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제 공개 코드입니다. (2021.05~2024.12)

본 멀티 채널 음성 분리 모델은 [SKA-TDNN](https://arxiv.org/abs/2204.01005v4) 기반으로 N개의 등록음성과, 입력 음성을 비교해 입력음성의 발화자를 찾아내는 speker identification 입니다. 본 실험은 VoxCeleb 및 SiTEC 한국어 음성 DB를 사용하여 진행되었습니다.

Done

* SKA-TDNN 기반 spekaer identification 성능 개선

To do

* Streaming speaker identification

Requirements
-------------
python==3.7.10     
torch==1.11.0    
torchaudio==0.11.0              
librosa==0.7.1         
soundifle==0.10.3              
numpy==1.19.4

scipy==1.7.3

Preprocessing
-------------
You can download the VoxCeleb Dataset in:
https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

    
Training
-------------
To train the model run this command:

    python trainSpeakerNet.py
    
    
    

Reference code:
-------------
* VoxCelebtrainer(ResNet): https://github.com/clovaai/voxceleb_trainer
* SKA-TDNN: https://github.com/msh9184/ska-tdnn
