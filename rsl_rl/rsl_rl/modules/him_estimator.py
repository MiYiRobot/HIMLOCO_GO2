import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import Normal, Categorical


class HIMEstimator(nn.Module):
    def __init__(self,
                 temporal_steps,
                 num_one_step_obs,
                 enc_hidden_dims=[128, 64, 16],
                 tar_hidden_dims=[128, 64],
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 num_prototype=32,
                 temperature=3.0,
                 **kwargs):
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(HIMEstimator, self).__init__()
        activation = get_activation(activation) #获取激活函数实例

        self.temporal_steps = temporal_steps  #历史长度6
        self.num_one_step_obs = num_one_step_obs #每步步长45
        self.num_latent = enc_hidden_dims[-1]  #环境特性维度16
        self.max_grad_norm = max_grad_norm #梯度裁剪阈值，控制estimator更新时梯度的最大范围，预防梯度爆炸导致不稳点
        self.temperature = temperature  #softmax 温度系数（T越小越极端，越大越平滑）

        # Encoder
        enc_input_dim = self.temporal_steps * self.num_one_step_obs #学生网络维度  6*45             
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
        self.encoder = nn.Sequential(*enc_layers)

        # Target
        tar_input_dim = self.num_one_step_obs  #教师网络维度45
        tar_layers = []
        for l in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation]
            tar_input_dim = tar_hidden_dims[l]
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
        self.target = nn.Sequential(*tar_layers)

        # Prototype
        self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1])

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_latent(self, obs_history):
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def forward(self, obs_history):
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel.detach(), z.detach()

    def encode(self, obs_history):
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel, z

    def update(self, obs_history, next_critic_obs, lr=None):
        if lr is not None: #调整学习率
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
        #特权网络中的基座线速度        
        vel = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs+3].detach()
        #特权网络中的其他数据
        next_obs = next_critic_obs.detach()[:, 3:self.num_one_step_obs+3] 

        #学生网络的前向传播 输入：观测历史 输出：预测的环境特性
        z_s = self.encoder(obs_history)
        #教师网络的前向传播 输入：特性观测 输出：事实得出的环境特性
        z_t = self.target(next_obs)
        pred_vel, z_s = z_s[..., :3], z_s[..., 3:]  #拆分学生特性 拆分成预测的线速度和环境特性

        #特征归一化
        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = F.normalize(z_t, dim=-1, p=2)

        #原形归一化
        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        #相似度计算
        score_s = z_s @ self.proto.weight.T   # @为矩阵叉乘
        score_t = z_t @ self.proto.weight.T

        #将相似度转为一个平衡的软分配矩阵
        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        #将相似度转为对数概率
        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        #交叉预测损失
        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()  #关于隐变量即环境特征的损失
        #监督损失：速度预测误差
        estimation_loss = F.mse_loss(pred_vel, vel)  #均方差损失
        losses = estimation_loss + swap_loss  #总损失

        self.optimizer.zero_grad()  #梯度清零
        losses.backward()   #反向传播
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm) #梯度裁剪
        self.optimizer.step()       #更新参数

        return estimation_loss.item(), swap_loss.item()


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    Q = torch.exp(out / eps).T  #对相似度矩阵进行缩放并取指数，得到一个非负矩阵Q，T为转置，使得行表示样本，列表示原型,out/eps为温度缩放
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()  #归一化，

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)  #行归一化
        Q /= K #使每行的和变成1/K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)  #列归一化
        Q /= B  #使每列的和变成1/B
    return (Q * B).T


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None