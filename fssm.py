import numpy as np
import torch
import torch.nn as nn
import SSM_function as sf


def Activation(activation=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softplus':
        return nn.Softplus()
#定义一个层间反馈层：
#结构为，反馈层，中间层，输出层
#输出层的输出通过反馈进入反馈层，这样可以保证中间层可以任意调整

#定义反馈层SSM
class input_ssm(nn.Module):
    def __init__(self,hidden_size, step, activation,len,channels):
        super().__init__()
        D_tensor = torch.tensor([0]).float()
        self.D = nn.Parameter(D_tensor, requires_grad=True)
        A,B,C,P,Q,diag =  sf.get_LegS(hidden_size,channels,DPLR=True)
        Ab,_,Cb = sf.discreatize(A, B, C, step, Discrete_method="B_trans")
        A_L = np.linalg.matrix_power(Ab,len)
        A_L = torch.from_numpy(A_L)
        B = torch.from_numpy(B)
        C = torch.from_numpy(Cb)
        P = torch.from_numpy(P)
        Q = torch.from_numpy(Q)
        diag = torch.from_numpy(diag)
        step = torch.tensor(step)
        self.A_L = nn.Parameter(A_L, requires_grad=False)
        self.B = nn.Parameter(B, requires_grad=True)
        self.C = nn.Parameter(C, requires_grad=True)
        self.P = nn.Parameter(P, requires_grad=True)
        self.Q = nn.Parameter(Q, requires_grad=True)
        self.diag = nn.Parameter(diag, requires_grad=False)
        self.step = nn.Parameter(step, requires_grad=True)
        self.activation = Activation(activation)

    def forward(self,r,feedback,fft=True):
        u = r - feedback
        K_c = sf.torch_get_K(self.A_L, self.B, self.C, self.P, self.Q, self.diag, self.step, x.shape[1],
                          DPLR=True)
        h1 = sf.torch_convolution(u, K_c, fft)
        y1 = (h1 + self.D * u)
        return self.activation(y1)

#定义中间层SSM
#就是普通的SSM，但是添加了传递函数的增益G、激活函数的增益H
class middle_ssm_e(nn.Module):
    def __init__(self,hidden_size, step, activation,len,channels):
        super().__init__()
        D_tensor = torch.tensor([0]).float()
        self.D = nn.Parameter(D_tensor, requires_grad=True)
        A,B,C,P,Q,diag =  sf.get_LegS(hidden_size,channels,DPLR=True)
        Ab,_,Cb = sf.discreatize(A, B, C, step, Discrete_method="B_trans")
        A_L = np.linalg.matrix_power(Ab,len)
        A_L = torch.from_numpy(A_L)
        B = torch.from_numpy(B)
        C = torch.from_numpy(Cb)
        P = torch.from_numpy(P)
        Q = torch.from_numpy(Q)
        diag = torch.from_numpy(diag)
        step = torch.tensor(step)
        self.A_L = nn.Parameter(A_L, requires_grad=False)
        self.B = nn.Parameter(B, requires_grad=True)
        self.C = nn.Parameter(C, requires_grad=True)
        self.P = nn.Parameter(P, requires_grad=True)
        self.Q = nn.Parameter(Q, requires_grad=True)
        self.diag = nn.Parameter(diag, requires_grad=False)
        self.step = nn.Parameter(step, requires_grad=True)
        if activation == "relu":
            self.activation = nn.ReLU()
            self.H = nn.Parameter(torch.tensor(1), requires_grad=False)
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
            self.H = nn.Parameter(torch.tensor(0.25), requires_grad=False)
        if activation == "tanh":
            self.activation = nn.Tanh()
            self.H = nn.Parameter(torch.tensor(1), requires_grad=False)

    def forward(self, r, fft=True):
        u = r
        K_c, K_h = sf.get_K_H(self.A_L, self.B, self.C, self.P, self.Q, self.diag, self.step, u.shape[1],
                               DPLR=True)
        h1 = sf.torch_convolution(u, K_c, fft)
        y1 = (h1 + self.D * u)
        return self.activation(y1), K_h * self.H

class middle_ssm(nn.Module):
    #参数设置
    def __init__(self,hidden_size, step, activation,len,channels,n_layers=1,DPLR=False):
        super().__init__()
        #中间可以叠加若干个线性层
        if n_layers>1:
            self.n = n_layers
            self.layers = nn.ModuleList()
            if activation == "relu": self.activation = nn.ReLU()
            if activation == "sigmoid":self.activation = nn.Sigmoid()
            if activation == "tanh":self.activation = nn.Tanh()
            for i in range(n_layers):
                self.layers.append(middle_ssm_e(hidden_size[i], step[i], activation,len[i],channels))
                self.layers.append(nn.Linear(channels,channels))
                self.layers.append(self.activation)
                self.layers.append(nn.LayerNorm(channels))

        else:
            self.n = 1
            self.layer = middle_ssm_e(hidden_size, step, activation,len,channels)
    def forward(self, r, fft=True):
        if self.n>1:
            for layer in self.layers:
                H = 1
                if isinstance(layer, middle_ssm_e):
                    r,h = layer(r)
                    H = H * h
                else:r= layer(r)
        else:r ,H= self.layer(r, fft)
        return r,H

#定义输出层SSM
#需要通过输出y获得反馈信号，反馈信号feedback=Fy，通过F映射为和最初的输入一个维度
class output_ssm(nn.Module):
    #feed_size指反馈回去的信号大小:batch * len * channel
    def __init__(self,hidden_size, step, activation,len,channels,input_size,feed_size):
        super().__init__()
        D_tensor = torch.tensor([0]).float()
        self.D = nn.Parameter(D_tensor, requires_grad=True)
        A,B,C,P,Q,diag =  sf.get_LegS(hidden_size,channels,DPLR=True)
        Ab,_,Cb = sf.discreatize(A, B, C, step, Discrete_method="B_trans")
        A_L = np.linalg.matrix_power(Ab,len)
        A_L = torch.from_numpy(A_L)
        B = torch.from_numpy(B)
        C = torch.from_numpy(Cb)
        P = torch.from_numpy(P)
        Q = torch.from_numpy(Q)
        diag = torch.from_numpy(diag)
        step = torch.tensor(step)
        self.A_L = nn.Parameter(A_L, requires_grad=False)
        self.B = nn.Parameter(B, requires_grad=True)
        self.C = nn.Parameter(C, requires_grad=True)
        self.P = nn.Parameter(P, requires_grad=True)
        self.Q = nn.Parameter(Q, requires_grad=True)
        self.diag = nn.Parameter(diag, requires_grad=False)
        self.step = nn.Parameter(step, requires_grad=True)
        if activation == "relu":
            self.activation = nn.ReLU()
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        if activation == "tanh":
            self.activation = nn.Tanh()

        self.H = nn.Parameter(torch.tensor(1,dtype=torch.float32), requires_grad=True)
        self.F1 = nn.Linear(input_size[1], feed_size[1])
        self.F2 = nn.Linear(input_size[2], feed_size[2])
    def forward(self, r, fft=True):
        u = r
        K_c, K_h = sf.get_K_H(self.A_L, self.B, self.C, self.P, self.Q, self.diag, self.step, u.shape[1],
                               DPLR=True)
        h1 = sf.torch_convolution(u, K_c, fft)
        #信号增益放在反馈之前,激活函数放在在反馈之后
        y1 = (h1 + self.D * u) * self.H
        feedback1 = self.F2(y1)
        feedback2 = self.F1(feedback1.transpose(-2,-1))
        return self.activation(y1), feedback2.transpose(-2,-1),K_h * self.H
class Feed_Block(nn.Module):
    def __init__(
            self,
            hidden_size,
            step,
            mult_activation,
            len,
            channels,
            mid_layers,
            output_size=None,
            feed_size=None,
            final_act='gelu',
            feed_act = None,
            skip = False,
            dropout=0.1,
            norm = False,
            DPLR=True):
        super().__init__()
        self.input_ssm = sf.SSM_Block(hidden_size, step, mult_activation, len, channels,final_act=final_act,skip=None,
                            dropout=dropout,norm=norm,DPLR=DPLR)
        self.output_ssm = sf.SSM_Block(hidden_size, step, mult_activation, len, channels,final_act=final_act,skip=skip,
                            dropout=dropout,norm=norm,DPLR=DPLR)
        self.mid_ssm = nn.ModuleList()
        self.mid_layers = mid_layers
        if mid_layers>1:
            for i in range(mid_layers):
                self.mid_ssm.append(sf.SSM_Block(hidden_size, step, mult_activation, len, channels,final_act=final_act,skip=skip,
                            dropout=dropout,norm=norm,DPLR=DPLR))
        else:self.mid_ssm.append(nn.Identity())
        self.fc = nn.Linear(channels,channels)
        if feed_act is not None:
            self.feed_act = Activation(feed_act)
        else:self.feed_act = nn.Identity()
    def forward(self,r):
        f = torch.zeros_like(r).to(r.device)
        for i in range(50):
            y1 = self.input_ssm(r-f)
            for layer in self.mid_ssm:
                y1= layer(y1)
            y2 = self.output_ssm(y1)
            F = self.fc(y2)
            e = torch.abs(F - f)
            # print('max_e = ',torch.max(e))
            # print('max_f = ',torch.max(torch.abs(F)))
            if torch.max(e) < torch.max(torch.abs(F))*0.01:
                # print('i=',i+1)
                break
            else:
                f = F.detach()
                f= self.feed_act(f)
        return y2

class FSSMBlock(nn.Module):
    def __init__(self, hidden_size, step, activation,len,input_size,channels,n_layers,feed_size):
        super(FSSMBlock, self).__init__()
        self.input_model = input_ssm(hidden_size, step, activation,len,channels)
        mid_hidden = [hidden_size]*n_layers
        mid_step = [step]*n_layers
        mid_len = [len]*n_layers
        self.hidden_model = middle_ssm(mid_hidden, mid_step,activation,mid_len,
                                          channels,n_layers=4)
        self.output_model = output_ssm(hidden_size, step, activation,len,channels,
                                          input_size,feed_size)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.bn3 = nn.BatchNorm1d(channels)

        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, channels)
        self.fc3 = nn.Linear(channels, channels)
        self.activation = nn.Tanh()

    def forward(self,x):
        f = torch.zeros(x.shape[0],x.shape[1],x.shape[2]).to(x.device)
        for i in range(10):
            y1 ,H1= self.input_model(x,f)
            y1 = self.fc1(y1)
            y1 = self.activation(y1)
            y1 = self.bn1(y1.transpose(1, 2)).transpose(1, 2)

            y2 ,H2= self.hidden_model(y1)
            y2 = self.bn2(y2.transpose(1, 2)).transpose(1, 2)

            y3 ,F ,H3= self.output_model(y2)

            e = torch.abs(F-f)
            if torch.max(e) < 1e-5:break
            else:f = F.detach()
        y3 = self.fc3(f)
        y3 = self.activation(y3)

        y = self.bn3(y3.transpose(1, 2)).transpose(1, 2)
        return y,H1*H2*H3

