import numpy as np
import os
from pathlib import Path
import csv
import pandas as pd
from torch.utils.data import Dataset
import librosa
import torch

from dataloader import n_mfcc

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


label = []
target_length = 1000
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
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=128)
        mfcc = pad_truncate(mfcc, target_length)
        return mfcc, label

df1 = pd.read_csv('data/testa_dataset.csv')
df2 = pd.read_csv('data/testb_dataset.csv')

dataset_a = AudioDataset(df1)
dataset_b = AudioDataset(df2)
features_a = []
features_b = []
labels_a = []
labels_b = []
for mfcc, label in dataset_a:
    features_a.append(mfcc)
    labels_a.append(label)
for mfcc, label in dataset_a:
    features_b.append(mfcc)
    labels_b.append(label)
features_a = np.array(features_a).swapaxes(1,2)
features_b = np.array(features_b).swapaxes(1,2)
#归一化
def guiyi(features):
    mean = features.mean(axis=(0, 1)).reshape(1,1,n_mfcc)
    std = features.std(axis=(0, 1)).reshape(1,1,n_mfcc)+1e-8
    features = (features - mean) / std
    return features

features_a = guiyi(features_a)
features_b = guiyi(features_b)
a_test = torch.from_numpy(features_a)
b_test = torch.from_numpy(features_b)
torch.save({
    'a_test': a_test,
    'b_test': b_test,
}, 'data/test_128.pt')
print("over")


