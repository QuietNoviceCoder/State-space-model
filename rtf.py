import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import SSM_function as sf
import matplotlib
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
data = torch.load('data/featurs_128.pt')
x_train_tensor = data['x_train'].to(device)
y_train_tensor = data['y_train'].to(device)
x_test_tensor = data['x_test'].to(device)
y_test_tensor = data['y_test'].to(device)
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)


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
        self.SSM1 = sf.SSMRTF_model(64,"tanh")
        self.SSM2 = sf.SSMRTF_model(64,"tanh")

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
        h1 = self.SSM1(u1, fft).permute(0, 2, 1)
        h1 = self.pool1(h1).permute(0, 2, 1)

        u2 = self.fc2(h1)
        u2 = self.dropout(u2)
        h2 = self.SSM2(u2, fft).permute(0, 2, 1)
        h2 = self.pool2(h2).permute(0, 2, 1)

        u4 = h2.flatten(1)
        y = self.fc4(u4)
        return y

model = SSMNet()
print(model)
model = torch.load('model/best_model_rtf.pth', weights_only=False)
#设置训练参数
epochs = 300
batch_size = 64
lossF = torch.nn.CrossEntropyLoss(label_smoothing=0)
#label_smoothing=0.1
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)
# train_sampler = DistributedSampler(train_dataset)#报错
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
history = {'Test Loss': [], 'Test Accuracy': []}
best_acc = 0.955966
#开始训练
model.to(device)
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
for epoch in range(1,epochs+1):
    train_loss, train_acc = 0, 0
    model.train()
    for train_mfccs,train_labels in train_loader:
        train_mfccs = train_mfccs
        train_labels = train_labels
        # if epoch > 0:
        #    noise = (2 * torch.randn_like(train_mfccs) - 1).to(device)
        #    train_mfccs = train_mfccs + noise * 0.001
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
    # scheduler.step()
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
        print(f'Epoch [{epoch + 1}/{epochs}], Train_Loss: {train_loss:.6f} , Train_Accuracy: {train_acc:.6f},'
              f'Test_Loss: {TestLoss.item():.6f} , Test_Accuracy: {testAccuracy.item():.6f},')

        if testAccuracy > best_acc:
            torch.save(model, 'model/best_model_rtf.pth')
            print('best_accuracy:', testAccuracy)
            best_acc = testAccuracy
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

torch.save(model, 'model/SSM_model_rtf.pth')
print("nice")