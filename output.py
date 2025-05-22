import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SSM_function as sf
import pandas as pd
import os
import csv
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#定义网络
hidden_size = 128  # 隐藏层大小

input_size = 128  # 输入特征维度
channels1 = 64  #第一次映射维度
channels2 = 16   #第二次映射维度
channels3 = 16
t_len = 1000
len_1 = 500
len_2 = 125
len_3 = 125
step = 0.1/t_len
class SSMNet(nn.Module):
    def __init__(self):
        super(SSMNet, self).__init__()
        self.SSM1 = sf.SSM_model(64, step, "tanh",t_len,DPLR=True)
        self.SSM2 = sf.SSM_model(64, step, "tanh",len_1,DPLR=True)
        # self.SSM3 = sf.SSM_model(64, step, "relu",len_2,DPLR=True)

        self.fc= nn.Linear(input_size, channels1)
        self.fc2 = nn.Linear(channels1, channels2)
        self.fc3 = nn.Linear(channels2, channels3)
        self.fc4 = nn.Linear(channels2 * len_2, out_features=20)

        self.dropout = nn.Dropout(0.4)
        self.pool1 = nn.AdaptiveAvgPool1d(len_1)
        self.pool2 = nn.AdaptiveAvgPool1d(len_2)
        self.pool3 = nn.AdaptiveAvgPool1d(len_3)
        self.ln1 = nn.LayerNorm(channels1)
        self.ln2 = nn.LayerNorm(channels2)
        self.ln3 = nn.LayerNorm(channels3)
    def forward(self, input,fft=True):

        u1 = self.fc(input)
        u1 = self.dropout(u1)
        h1 = self.SSM1(u1, fft,DPLR=True).permute(0, 2, 1)
        h1 = self.pool1(h1).permute(0, 2, 1)

        u2 = self.fc2(h1)
        u2 = self.dropout(u2)
        h2 = self.SSM2(u2, fft,DPLR=True).permute(0, 2, 1)
        h2 = self.pool2(h2).permute(0, 2, 1)

        u4 = h2.flatten(1)
        y = self.fc4(u4)
        return y

model = SSMNet()
print(model)
model = torch.load('model/best_model.pth', weights_only=False)
model.dropout = nn.Dropout(0.6)
#加载测试数据
data = torch.load('data/test_128.pt')
test_tensor = data['a_test'].to(device)
#开始测试
model.to(device)
model.eval()
test = DataLoader(test_tensor, batch_size=16, shuffle=False)
out = []
for mfccs in test:
    outputs = model(mfccs)
    predictions = torch.argmax(outputs, dim=1)
    out.append(predictions)
out = torch.cat(out,dim=0).int().to('cpu').numpy()
#写入
label_list = ['aloe', 'burger', 'cabbage','candied_fruits', 'carrots', 'chips',
                  'chocolate','drinks', 'fries', 'grapes', 'gummies', 'ice-cream',
                  'jelly', 'noodles', 'pickles', 'pizza', 'ribs', 'salmon',
                  'soup', 'wings']
df = pd.read_csv('data/testa_dataset.csv')
name = df['file_path']
labels = []

with open('data/out.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # 写入表头
    writer.writerow(['name', 'label'])
    for i in range(len(out)):
        label = label_list[out[i]]
        writer.writerow([os.path.basename(name[i]),label])
print('shan')
