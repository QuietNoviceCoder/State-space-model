import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np

#自定义数据类
csv_path = 'data/train_dataset.csv'
df = pd.read_csv(csv_path)
print(df.head())


print("nice")