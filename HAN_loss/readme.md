# HAN-loss-pyTorch

Implementation of "Improving Learning Objectives for Speaker Verification from the Perspective of Score Comaprison" (ICASSP 2023)


How to use?

First, use have to prepare the speech dataset, utterances & speaker label 

You can download the VoxCeleb Dataset in:
https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

    
Training
-------------
To train the model run this command:

    python trainSpeakerNet.py
    
    
Requirements
-------------
python==3.7.10     
torch==1.11.0    
torchaudio==0.11.0              
librosa==0.7.1         
soundifle==0.10.3              
numpy==1.19.4

scipy==1.7.3    

Paper
-------------
https://ieeexplore.ieee.org/document/10095828

