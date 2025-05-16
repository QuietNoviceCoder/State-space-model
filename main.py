import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import SSM_function as sf
from tqdm import tqdm
import sys
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
if __name__ == '__main__':
    # 加载数据
    train_features_path = 'data/train_features.npy'
    train_labels_path = 'data/train_labels.npy'
    train_features = np.load(train_features_path)
    train_labels = np.load(train_labels_path)
    x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.1)
    x_train_tensor = torch.from_numpy(x_train).to(device)
    x_test_tensor = torch.from_numpy(x_test).to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)
    y_test_tensor = torch.from_numpy(y_test).to(device)
    train_dataset = featureDataset(x_train_tensor, y_train_tensor)
    test_dataset = featureDataset(x_test_tensor, y_test_tensor)
    #定义网络
    input_size = 20  # 输入特征维度
    hidden_size = 128  # 隐藏层大小
    channels1 = 32  #第一次映射维度
    channels2 = 16  #第二次映射维度
    channels3 = 8   #第三次映射维度
    slide_window = 300
    t_len = 300
    step = 0.1/t_len
    class SSMNet(nn.Module):
        def __init__(self):
            super(SSMNet, self).__init__()
            self.SSM1 = sf.SSM_model(hidden_size,1/slide_window, "tanh",t_len,DPLR=True)
            self.SSM2 = sf.SSM_model(hidden_size,1/slide_window, "tanh",t_len,DPLR=True)
            self.SSM3 = sf.SSM_model(hidden_size,1/slide_window, "tanh",t_len,DPLR=True)
            self.fc1 = nn.Linear(input_size, channels1)
            self.fc2 = nn.Linear(channels1, channels2)
            self.fc3 = nn.Linear(channels2, channels3)
            self.fc4 = nn.Linear(channels3 * t_len, out_features=20)
        def forward(self, input,fft=True):
            x = self.fc1(input)
            h1 = self.SSM1(x, fft,DPLR=True)
            x1 = self.fc2(h1)
            h2 = self.SSM2(x1, fft,DPLR=True)
            x2 = self.fc3(h2)
            h3 = self.SSM3(x2, fft,DPLR=True)
            x3 = h3.flatten(1)
            y = self.fc4(x3)
            return y

    model = SSMNet()
    print(model)

    #设置训练参数
    epochs = 100
    batch_size = 32
    lossF = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001,weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    history = {'Test Loss': [], 'Test Accuracy': []}

    #开始训练
    model.to(device)
    for epoch in range(1,epochs+1):
        processBar = tqdm(train_loader,unit = 'step')
        model.train()
        for step,(train_mfccs,train_labels) in enumerate(processBar):
            model.zero_grad()
            outputs = model(train_mfccs)
            loss = lossF(outputs, train_labels)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(predictions == train_labels) / train_labels.shape[0]

            #反向传播
            loss.backward()
            optimizer.step()
            # scheduler.step()
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                       (epoch, epochs, loss.item(), accuracy.item()))

            if step == len(processBar) - 1:
                correct, totalLoss = 0, 0
                model.train(False)
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
                sys.stdout.write("\r\033[K")  # \r 光标回到行首，\033[K 清除整行
                sys.stdout.flush()
                tqdm.write("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                           (epoch, epochs, loss.item(), accuracy.item(), TestLoss.item(), testAccuracy.item()))
        processBar.close()
        scheduler.step()
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

    torch.save(model, 'SSM_model_8_4.pth')
    print("nice")