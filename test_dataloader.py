import numpy as np
import os
from pathlib import Path
import csv
import pandas as pd
from torch.utils.data import Dataset
import librosa

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


label = []
target_length = 300
testa_dir = Path('data/test_a')
testb_dir = Path('data/test_b')

with open('data/testa_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # 写入表头
    writer.writerow(['file_path', 'label'])
    for file in sorted(os.listdir(testa_dir)):
        full_path = os.path.join(testa_dir, file)
        writer.writerow([full_path])
with open('data/testb_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # 写入表头
    writer.writerow(['file_path', 'label'])
    for file in sorted(os.listdir(testb_dir)):
        full_path = os.path.join(testb_dir, file)
        writer.writerow([full_path])
#统一输出帧的长度
def pad_truncate(mfcc, target_length):
    # 1秒音频（16kHz）
    len = mfcc.shape[1]
    if len > target_length:
        return mfcc[:,:target_length]
    else:
        return np.pad(mfcc,((0,0), (0,target_length - len)))


class AudioDataset(Dataset):
    def __init__(self, dataframe):
        self.file_paths = dataframe['file_path'].values
        self.labels = dataframe['label'].values

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 加载音频
        audio_path = self.file_paths[idx]
        audio, sr = librosa.load(audio_path,sr=None)
        # 处理标签（假设是字符串标签）
        label = self.labels[idx]
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc = pad_truncate(mfcc, target_length)
        return mfcc, label

dataset = AudioDataset(df)
features = []
labels = []
for mfcc, label in dataset:
    features.append(mfcc)
    labels.append(label)
features = np.array(features).swapaxes(1,2)
labels = np.array(labels)
#归一化
mean = features.mean(axis=(0, 1)).reshape(1,1,13)
std = features.std(axis=(0, 1)).reshape(1,1,13)+1e-8
features = (features - mean) / std
np.save('data/train_features.npy', features)
np.save('data/train_labels.npy', labels)
print("over")


