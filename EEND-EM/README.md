This folder contains core code implementation that is used in EEND-EM: End-to-End Neural Speaker Diarization with EM-Network (To be published at APSIPA2024)

Proposed method code is implemented based on ESPNet(https://github.com/espnet/espnet)

You can re implement the reuslt of the paper by replacing two codes from this page.
1. Instead of using num mels setting of conf/train_diar_eda.yaml we use specific batch size 64.

2. For espnet_model.py, you can replace this file to /espnet/espnet2/diar/espnet_model.py

3. For trainer.py, you can replace this file to /espnet/espnet2/train/trainer.py
