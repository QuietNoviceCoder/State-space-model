import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SSM_function as sf
import matplotlib
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
#自定义数据类
class featureDataset(Dataset):
    def __init__(self, mfccs, labels):
        self.mfccs = mfccs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.mfccs[idx], self.labels[idx]
# 加载数据
data = torch.load('data/featurs_128.pt')
x_train_tensor = data['x_train'].to(device)
y_train_tensor = data['y_train'].to(device)
x_test_tensor = data['x_test'].to(device)
y_test_tensor = data['y_test'].to(device)
train_dataset = featureDataset(x_train_tensor, y_train_tensor)
test_dataset = featureDataset(x_test_tensor, y_test_tensor)

if __name__ == '__main__':
    #定义网络
    hidden_size = 128  # 隐藏层大小

    input_size = 128  # 输入特征维度
    channels1 = 64  #第一次映射维度
    channels2 = 32   #第二次映射维度
    channels3 = 16
    t_len = 1000
    len_1 = 500
    len_2 = 250
    len_3 = 125
    step = 0.1/t_len
    class SSMNet(nn.Module):
        def __init__(self):
            super(SSMNet, self).__init__()
            self.SSM1 = sf.SSM_model(hidden_size, step, "tanh",t_len,DPLR=True)
            self.SSM2 = sf.SSM_model(hidden_size, step, "tanh",len_1,DPLR=True)
            self.SSM3 = sf.SSM_model(hidden_size, step, "relu",len_2,DPLR=True)

            self.fc= nn.Linear(input_size, channels1)
            self.fc2 = nn.Linear(channels1, channels2)
            self.fc3 = nn.Linear(channels2, channels3)
            self.fc4 = nn.Linear(channels3 * len_3, out_features=20)

            self.dropout = nn.Dropout(0.4)
            self.pool1 = nn.AdaptiveAvgPool1d(len_1)
            self.pool2 = nn.AdaptiveAvgPool1d(len_2)
            self.pool3 = nn.AdaptiveAvgPool1d(len_3)
        def forward(self, input,fft=True):

            u1 = self.fc(input)
            u1 = self.dropout(u1)
            h1 = self.SSM1(u1, fft,DPLR=True).permute(0, 2, 1)
            h1 = self.pool1(h1).permute(0, 2, 1)

            u2 = self.fc2(h1)
            u2 = self.dropout(u2)
            h2 = self.SSM2(u2, fft,DPLR=True).permute(0, 2, 1)
            h2 = self.pool2(h2).permute(0, 2, 1)

            u3 = self.fc3(h2)
            u3 = self.dropout(u3)
            h3 = self.SSM3(u3, fft,DPLR=True).permute(0, 2, 1)
            h3 = self.pool3(h3).permute(0, 2, 1)

            u4 = h3.flatten(1)
            y = self.fc4(u4)
            return y

    model = SSMNet()
    print(model)

    #设置训练参数
    epochs = 1000##还训练呢
    batch_size = 64
    lossF = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.9)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    history = {'Test Loss': [], 'Test Accuracy': []}
    model = torch.load('model/SSM_model_f128.pth', weights_only=False)
    #开始训练
    model.to(device)
    for epoch in range(1,epochs+1):
        train_loss, train_acc = 0, 0
        model.train()
        for train_mfccs,train_labels in train_loader:
            train_mfccs = train_mfccs
            train_labels = train_labels
            noise = torch.randn_like(train_mfccs).to(device)
            if epoch > 40:
                train_mfccs = train_mfccs + noise * 0.005 * train_mfccs
            model.zero_grad()
            outputs = model(train_mfccs)
            loss = lossF(outputs, train_labels)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(predictions == train_labels) / train_labels.shape[0]
            train_loss +=loss
            train_acc += accuracy
            #反向传播
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch+1)%10 == 0:
            train_loss = train_loss / len(train_loader)
            train_acc = train_acc / len(train_loader)
            correct, totalLoss = 0, 0
            model.eval()
            with torch.no_grad():
                for test_mfccs, labels in test_loader:
                    test_mfccs = test_mfccs
                    labels = labels
                    outputs = model(test_mfccs)
                    testloss = lossF(outputs, labels)
                    predictions = torch.argmax(outputs, dim=1)
                    totalLoss += testloss
                    correct += torch.sum(predictions == labels)

            testAccuracy = correct / (batch_size * len(test_loader))
            TestLoss = totalLoss / len(test_loader)
            history['Test Loss'].append(TestLoss.item())
            history['Test Accuracy'].append(testAccuracy.item())
            print(f'Epoch [{epoch + 1}/{epochs}], Train_Loss: {train_loss:.4f} , Train_Accuracy: {train_acc:.4f},'
                  f'Test_Loss: {TestLoss.item():.4f} , Test_Accuracy: {testAccuracy.item():.4f},')
            # if testAccuracy > max(history['Test Loss']):
            #     torch.save(model, 'best_model.pth')
            #     print('best_accuracy:', testAccuracy)
            train_loss, train_acc = 0, 0
    #%%
    matplotlib.pyplot.plot(history['Test Loss'], label='Test Loss')
    matplotlib.pyplot.legend(loc='best')
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.xlabel('Epoch')
    matplotlib.pyplot.ylabel('Loss')
    matplotlib.pyplot.show()

    matplotlib.pyplot.plot(history['Test Accuracy'], color='red', label='SSM Test Accuracy')
    matplotlib.pyplot.legend(loc='best')
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.xlabel('Epoch')
    matplotlib.pyplot.ylabel('Accuracy')
    matplotlib.pyplot.show()

    torch.save(model, 'model/SSM_model_f128%.pth')
    print("nice")