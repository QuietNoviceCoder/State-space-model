import torch
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import numpy as np

#自定义数据类
csv_path = 'data/train_dataset.csv'
df = pd.read_csv(csv_path)
print(df.head())
def load_audio(path):
    audio, sr = librosa.load(path)
    return audio, sr
audio_data = []
labels = []
for index, row in df.iterrows():
    try:
        audio = load_audio(row['file_path'])
        label = row['label']
        audio_data.append(audio)
        labels.append(label)
    except Exception as e:
        print(f"Error loading {row['file_path']}: {str(e)}")
print("nice")