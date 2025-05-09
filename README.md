## This is repository containing the Project on Deep Learning Based Techniques for Audio Denoising.

### Members:
[Aalekhya Mukhopadhyay](mailto:mukhopadhyayaalekhya@gmail.com)  
[Mayank Nagar](mailto:nmayank1998@gmail.com)  
[Ayush Yadav](mailto:yadavayush7028@gmail.com)  
[Kalyani Gohokar](mailto:kalyani.gohokar2406@gmail.com)

### Problem Statement: Deep-Learning Based Audio Denoising & Enhancement.

### Data: [Voicebak-DEMAND](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k), Self-made ([LibriSpeech](https://www.openslr.org/12) + [MUSAN](https://www.kaggle.com/datasets/nhattruongdev/musan-noise/data))

### Model: [Wave-U-NET](https://arxiv.org/pdf/1806.03185), [SEGAN](https://arxiv.org/pdf/1703.09452)

### Responsibilities: 
Aalekhya: Dataset curation (download and organize), Preprocessing, DVC.  
Mayank & Ayush: Models’ building, training, testing and logging.  
Kalyani: MLOps integration and model deployment.

### Webapp at: https://denoising-audio.streamlit.app/

## Folder Structure

```
Audio-Denoising-Project/
├── BackEnd/
│   └── streamlit.py
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
├── src/
│   ├── models/
│   │   └── Wave-U-Net/
│   │       └── train.py
│   └── output/
│        └── Wave-U-Net-outputs/
└── README.md

```
