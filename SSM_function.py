import torch
import torch.nn as nn
import numpy as np
from flashfft.flashfftconv import FlashFFTConv
import warnings
warnings.filterwarnings("ignore",category=  UserWarning,message="ComplexHalf support is experimental.*")
#定义hippo矩阵和离散方法
def get_LegT(N,slide_window):
    A = np.zeros((N, N), dtype=float)
    B = np.zeros((N, 1))
    for i in range(N):
        for j in range(N):
            base = -1/slide_window*(np.sqrt((2*i+1)*(2*j+1)))
            if i > j : A[i,j] = base
            else : A[i,j] = base*((-1)**(i-j))
        B[i,0] = 1/slide_window*(np.sqrt(2*i+1))
    C = np.ones((1, N))
    return A,B,C
#求输入的对角阵和转换矩阵
def eig_matrix(input):
    value, vectors = np.linalg.eig(input)
    eig_value = np.imag(value) * 1j
    for i in range(len(value)):
        vectors[:, i] = vectors[:, i] / np.linalg.norm(vectors[:, i])
    U = vectors
    return eig_value , U
def conj_round(input):
    threshold = 1e-6
    real_part = input.real
    imag_part = input.imag
    real_part[torch.abs(real_part) < threshold] = 0
    imag_part[torch.abs(imag_part) < threshold] = 0
    return real_part+imag_part*1j
def get_LegS(N,DPLR = False):
    if DPLR==False:
        A = np.zeros((N, N), dtype=float)
        B = np.zeros((N, 1))
        for i in range(N):
            for j in range(N):
                if i > j : A[i,j] = -(np.sqrt((2*i+1)*(2*j+1)))
                elif i == j: A[i,j] = -(i+1)
            B[i,0] = np.sqrt(2*i+1)
        C = np.ones((1, N))
        return A,B,C
    if DPLR==True:
        S = np.zeros((N, N), dtype=float)
        B = np.zeros((N, 1))
        C = np.ones((1, N))
        P = np.zeros((N, 1))
        Q = np.zeros((N, 1))
        for i in range(N):
            for j in range(N):
                if i == j : S[i,j] = 0
                elif i > j : S[i,j] = -1/2*np.sqrt((2*i+1)*(2*j+1))
                elif i < j : S[i,j] = 1/2*np.sqrt((2*i+1)*(2*j+1))
            B[i, 0] = np.sqrt(2 * i + 1)
            P[i, 0] = np.sqrt(2 * i + 1)
            Q[i, 0] = np.sqrt(2 * i + 1)
        eig_value , U = eig_matrix(S)
        eig_value = eig_value - 1/2
        diag = np.diag(eig_value)
        diag = diag - np.eye(N)*1/2
        P = U.conj().T @ P * np.sqrt(2) / 2
        Q = U.conj().T @ Q * np.sqrt(2) / 2
        B = U.conj().T @ B
        C = C @ U
        A = diag -  P @ Q.conj().T
        return A,B,C,P,Q,eig_value
def get_RTF(N,ini="zeros"):
    C = np.ones((1, N + 1))
    if ini == "zeros":
        A = np.zeros((1, N+1))
        A[0, 0] = 1
    if ini == "roots":
        sroots = -1 * np.arange(1, N + 1)
        zroots = np.exp(sroots)
        A = np.poly(zroots).reshape(1, N+1)
    return A,C

def discreatize(A,B,C,step,Discrete_method="B_trans"):
    I = np.eye(A.shape[0])
    if Discrete_method == "F_trans":
        Ab = I + step * A
        Bb = step * B
        return Ab,Bb,C
    if Discrete_method == "Back_trans":
        Ab = np.linalg.inv(I - step* A)
        Bb = step * Ab @ B
        return Ab, Bb, C
    if Discrete_method == "B_trans":
        BL = np.linalg.inv(I - (step / 2.0) * A)
        Ab = BL @ (I + (step / 2.0) * A)
        Bb = step * BL @ B
        return Ab, Bb, C

#定义RNN过程
def scan_SSM(Ab,Bb,Cb,u,x0):
    x1 = Ab @ x0 + Bb*u
    y = Cb @ x1
    return x1,y

def run_SSM(Ab,Bb,Cb,u):
    L = u.shape[0]
    N = Ab.shape[0]
    x0 = np.zeros((N,1))
    y = np.zeros((1,L))
    for i in range(L):
        x0,y[0,i] = scan_SSM(Ab,Bb,Cb,u[i],x0)
    return y

#定义卷积过程
#卷积核
def get_K(A,B,C,n_times):
    for i in range(n_times):
        if i > 0 :
            raw_date = A @ raw_date
            K = np.hstack([K,raw_date])
        elif i == 0:
            raw_date = B
            K = B
    return C @ K
def cauchy(QP,w,lamda):
    QP = QP.flatten()
    w = w.reshape(len(w),1)
    out = (QP/(w - lamda)).sum(dim = 1)
    return out.reshape(1,len(w))


def torch_get_K(*args,DPLR=False):
    if DPLR == False :
        A,B,C,n_times = args
        for i in range(n_times):
            if i > 0 :
                raw_date = torch.mm(A,raw_date)
                K = torch.cat((K,raw_date),dim=1)
            elif i == 0:
                raw_date = B
                K = B
        return torch.mm(C,K)
    if DPLR == True:
        A_L,B,C,P,Q,eig_value,derta,n_times = args
        I = torch.eye(A_L.shape[0]).to(A_L.device)
        z = torch.exp((torch.pi * -2j) * torch.arange(n_times) / n_times).to(A_L.device)
        w = 2 / derta * (1-z)/(1+z)
        #C波浪
        C_ = C @ (I - A_L)
        k00 = cauchy(C_ * B.T,w,eig_value)
        k01 = cauchy(C_ * P.T,w,eig_value)
        k10 = cauchy(Q.conj().T * B.T,w,eig_value)
        k11 = cauchy(Q.conj().T * P.T,w,eig_value)
        K_w = 2/(1+z)*(k00 - k01 / (1+k11) * k10)
        K = torch.fft.irfft(K_w,n_times)
        return K

def torch_get_RTF(A,C,len):
    out = torch.zeros(1,len).to(A.device)
    num = C.clone()
    for i in range(len):
        k_ = num[0,0]
        out[0,i] = k_
        num_new = torch.sub(num,A*k_)
        num = torch.roll(num_new,-1,1)
    return out

#定义卷积函数
def convolution(u,K,fft):
    K = K.reshape(u.shape[0])
    u = u.T.reshape(u.shape[0])
    if fft == False:
        y = np.convolve(u, K)[:u.shape[0]]
        return y
    if fft == True:
        Kd = np.fft.rfft(np.pad(K,(0,u.shape[0])))
        ud = np.fft.rfft(np.pad(u,(0,K.shape[0])))
        out = np.fft.irfft(Kd*ud,u.shape[0])
        return out


def torch_convolution(u,K,fft):
    if len(u.shape)==2:
        u = u.reshape(u.shape[0],-1)
        K = K.flatten()
        if fft == False:
            out = torch.nn.functional.conv1d(u, K)[:u.shape[1]]
            return out
        if fft == True:
            u_pad = torch.nn.functional.pad(u,(0,K.shape[0]))
            k_pad = torch.nn.functional.pad(K,(0,u.shape[1]))
            K_fft = torch.fft.rfft(k_pad)
            u_fft = torch.fft.rfft(u_pad)
            out_fft = K_fft * u_fft
            out = torch.fft.irfft(out_fft,u.shape[1]).float()
        return out
    elif len(u.shape)==3:
        out = torch.zeros([u.shape[0],u.shape[1],u.shape[2]],device=u.device)
        if fft == False:
            out = torch.nn.functional.conv1d(u, K)[:u.shape[0]]
            return out
        if fft == True:
            u_pad = torch.nn.functional.pad(u,(0,0,0,K.shape[1],0,0))
            k_pad = torch.nn.functional.pad(K,(0,u.shape[1]))
            for i in range(u.shape[2]):
                K_fft = torch.fft.rfft(k_pad)
                u_fft = torch.fft.rfft(u_pad[:,:,i])
                out_fft = K_fft * u_fft
                out[:,:,i] = torch.fft.irfft(out_fft,u.shape[1])
        return out
def torch_flashfftconv(u,K,L):
    #输入u的形状是（B，H,L）,K的形状是（H,L）
    #L必须是256-4,194,304之间的2的幂，若大于32768，必须是16的倍数
    #u的长度可以小于L，但是必须是2的倍数，L的大小必须是4的倍数
    u = u.permute(0, 2, 1).half().contiguous()
    K = K.half().contiguous()
    flash_conv = FlashFFTConv(L).to(u.device)
    output = flash_conv(u, K)
    return output.permute(0, 2, 1).float()
#定义SSM线性层
class SSM_model(nn.Module):
    def __init__(self,*args,DPLR=False):
        super().__init__()
        D_tensor = torch.tensor([0]).float()
        self.D = nn.Parameter(D_tensor, requires_grad=True)
        if DPLR == False:
            hidden_size, step, activation = args
            A, B, C = get_LegS(hidden_size)
            A, B, C = discreatize(A, B, C, step, Discrete_method="B_trans")
            A_tensor = torch.from_numpy(A).float()
            B_tensor = torch.from_numpy(B).float()
            C_tensor = torch.from_numpy(C).float()
            self.A = nn.Parameter(A_tensor, requires_grad=False)
            self.B = nn.Parameter(B_tensor, requires_grad=True)
            self.C = nn.Parameter(C_tensor, requires_grad=True)
        if DPLR == True:
            hidden_size, step, activation,len = args
            A,B,C,P,Q,diag =  get_LegS(hidden_size,DPLR=True)
            Ab,_,Cb =discreatize(A, B, C, step, Discrete_method="B_trans")
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
            self.P = nn.Parameter(P, requires_grad=False)
            self.Q = nn.Parameter(Q, requires_grad=False)
            self.diag = nn.Parameter(diag, requires_grad=False)
            self.step = nn.Parameter(step, requires_grad=False)
        if activation == "relu":
            self.activation = nn.ReLU()
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        if activation == "tanh":
            self.activation = nn.Tanh()
    def forward(self,x,fft=True,DPLR=False):
        if DPLR == False:
            K_c = torch_get_K(self.A, self.B, self.C, x.shape[1])
            h1 = torch_convolution(x,K_c,fft)
            y1 = h1 + self.D * x
        if DPLR == True:
            K_c = torch_get_K(self.A_L, self.B, self.C, self.P, self.Q, self.diag,self.step, x.shape[1], DPLR=True)
            h1 = torch_convolution(x, K_c, fft)
            y1 = h1 + self.D * x
        return self.activation(y1)

class SSMRTF_model(nn.Module):
    def __init__(self,hidden_size,activation,L):
        super().__init__()
        self.L = L
        A, C = get_RTF(hidden_size,ini="zeros")
        A_tensor = torch.from_numpy(A).float()
        C_tensor = torch.from_numpy(C).float()
        self.A = nn.Parameter(A_tensor,requires_grad=True)
        self.C = nn.Parameter(C_tensor,requires_grad=True)
        if activation == "relu":
            self.activation = nn.ReLU()
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        if activation == "tanh":
            self.activation = nn.Tanh()
    def forward(self,x):
        K_c = torch_get_RTF(self.A, self.C,x.shape[1])
        K = K_c.repeat(x.shape[2], 1)
        h1 = torch_flashfftconv(x,K,self.L)
        h2 = self.activation(h1)
        return h2