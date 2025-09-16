import torch
import torch.nn as nn
import SSM_function as sf
from flashfft.flashfftconv import FlashFFTConv


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

#定义输入层，中间层，反馈层

class middle_fssm(nn.Module):
    def __init__(self,hidden_size, step, activation,len,channels):
        super().__init__()
        D_tensor = torch.tensor([0]).float()
        self.D = nn.Parameter(D_tensor, requires_grad=True)
        A,B,C,P,Q,diag =  sf.get_LegS(hidden_size,channels,DPLR=True)
        Ab,_,Cb = sf.discreatize(A, B, C, step, Discrete_method="B_trans")
        B = torch.from_numpy(B)
        C = torch.from_numpy(Cb)
        P = torch.from_numpy(P)
        Q = torch.from_numpy(Q)
        diag = torch.from_numpy(diag)
        step = torch.tensor(step)
        len = sf.return_L(len)
        len = torch.tensor(len)
        self.flashfftconv = FlashFFTConv(2*len, dtype=torch.float16)
        self.B = nn.Parameter(B, requires_grad=True)
        self.C = nn.Parameter(C, requires_grad=True)
        self.P = nn.Parameter(P, requires_grad=True)
        self.Q = nn.Parameter(Q, requires_grad=True)
        self.diag = nn.Parameter(diag, requires_grad=False)
        self.step = nn.Parameter(step, requires_grad=True)
        self.activation = Activation(activation)
    def forward(self,r):
        u = r.permute(0, 2, 1).half().contiguous()
        K_c = sf.torch_get_K_derta(self.B, self.C, self.P, self.Q, self.diag, self.step, u.shape[2],
                          DPLR=True)
        K = K_c.contiguous()
        h1 = self.flashfftconv(u, K).float()
        y1 = (h1 + self.D * u)
        y = self.activation(y1)
        return y.permute(0, 2, 1)

class FSSM_Block(nn.Module):
    def __init__(
            self,
            hidden_size,
            step,
            mult_activation,
            len,
            channels,
            model='input',
            final_act='gelu',
            skip = False,
            dropout=0.0,
            norm = False,
            input_size=None,
            feed_size=None,
            feed_act=None,
            ):
        super().__init__()
        if model == 'input': self.fssm = middle_fssm(hidden_size, step, mult_activation,len,channels)
        if model == 'middle':self.fssm = middle_fssm(hidden_size, step, mult_activation,len,channels)
        if model == 'output':
            self.fssm = middle_fssm(hidden_size, step, mult_activation,len,channels)
            self.fc1 = nn.Linear(input_size[1], feed_size[1])
            self.fc2 = nn.Linear(feed_size[2], feed_size[2])
            self.feedact = Activation(feed_act)
        self.model = model
        self.final_act = Activation(final_act)
        self.fc = nn.Linear(channels,channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = skip
        self.normlization = norm
        self.H = nn.Parameter(torch.tensor(1,dtype=torch.float32), requires_grad=True)
        if norm == 'BN':self.norm = nn.BatchNorm1d(channels)
        if norm == 'LN':self.norm = nn.LayerNorm(channels)
    def forward(self,x,feedback=None):
        if self.model == 'input':
            y1 = self.fssm(x-feedback)
        else:
            y1 = self.fssm(x)
        y2 = self.fc(y1)
        y2 = self.final_act(y2)
        if self.skip: y2 = y2 + x
        if self.normlization == 'BN' :y2 = self.norm(y2.transpose(1, 2)).transpose(1, 2)
        if self.normlization == 'LN' :y2 = self.norm(y2)
        y = self.dropout(y2)*self.H
        if self.model == 'output':
            feedback = self.fc2(y)
            feedback = self.fc1(feedback.permute(0,2,1)).permute(0,2,1)
            feedback = self.feedact(feedback)
            h = torch.norm(feedback, p=2, dim=1) / (torch.norm(x, p=2, dim=1) + 1e-6) * self.H
            return y, feedback,h
        else:
            h = torch.norm(y, p=2, dim=1) / (torch.norm(x, p=2, dim=1) + 1e-6) * self.H
            return y, h

class FSSM_model(nn.Module):
    def __init__(
            self,
            hidden_size,
            step,
            mult_activation,
            len,
            channels,
            mid_layers=0,
            final_act='gelu',
            skip = False,
            dropout=0.0,
            norm = False,
            input_size=None,
            feed_size=None,
            feed_act=None,
            ):
        super().__init__()
        self.input = FSSM_Block(hidden_size,step,mult_activation,len,channels,'input',final_act,skip,dropout,norm,
                                input_size=None)
        self.mid = nn.ModuleList()
        if mid_layers > 0:
            for i in range(mid_layers):self.mid.append(
                FSSM_Block(hidden_size, step, mult_activation, len, channels, 'middle', final_act, skip, dropout, norm)
            )
        else:self.mid.append(nn.Identity())
        self.midlayers = mid_layers
        self.output = FSSM_Block(hidden_size,step,mult_activation,len,channels,'output',final_act,skip,dropout,norm,
                                input_size, feed_size, feed_act)
    def forward(self,x):
        feed = torch.zeros_like(x)
        for i in range(30):
            y1, h1 = self.input(x, feed)
            h2 = torch.ones_like(h1)
            y2 = y1
            if self.midlayers > 0:
                for layer in self.mid:
                    y2,h2_ = layer(y2)
                    h2 = h2 * h2_
            y3,feedback,h3 = self.output(y2)
            e = torch.abs((feedback - feed) / (torch.abs(feedback) + 1e-8) * 100)
            if torch.mean(e) < 5:
                # print('tiao_i=', i)
                break
            else:
                feed = (feed + 0.6 * (feedback - feed)).detach()
        h = h1*h2*h3
        return  y3,h

def loss_h(h,Target):
    return torch.relu(h - Target).mean()