# from common.common import *
from algorithm.value_base.DQN import DQN
import torch.nn as nn
import torch.nn.functional as func
import torch

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")
"""use CPU or GPU"""


class DuelingNeuralNetwork(nn.Module):
    def __init__(self, _input: int, _output: int):
        """
        :brief:             神经网络初始化
        :param _input:      输入维度
        :param _output:     输出维度
        """
        super(DuelingNeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(_input, 64)  # input -> hidden1
        self.hidden2 = nn.Linear(64, 64)  # hidden1 -> hidden2
        # self.out = nn.Linear(64, _output)  # hidden2 -> output
        self.value = nn.Linear(64, _output)
        self.advantage = nn.Linear(64, _output)
        # self.init()
        self.init_default()

    def init(self):
        torch.nn.init.orthogonal_(self.hidden1.weight, gain=1)
        torch.nn.init.uniform_(self.hidden1.bias, 0, 1)
        torch.nn.init.orthogonal_(self.hidden2.weight, gain=1)
        torch.nn.init.uniform_(self.hidden2.bias, 0, 1)
        torch.nn.init.orthogonal_(self.out.weight, gain=1)
        torch.nn.init.uniform_(self.out.bias, 0, 1)
        torch.nn.init.orthogonal_(self.value.weight, gain=1)
        torch.nn.init.uniform_(self.value.bias, 0, 1)
        torch.nn.init.orthogonal_(self.advantage.weight, gain=1)
        torch.nn.init.uniform_(self.advantage.bias, 0, 1)

    def init_default(self):
        self.hidden1.reset_parameters()
        self.hidden2.reset_parameters()
        self.value.reset_parameters()
        self.advantage.reset_parameters()

    def forward(self, _x):
        """
        :brief:         神经网络前向传播
        :param _x:      输入网络层的张量
        :return:        网络的输出
        """
        x = _x
        x = self.hidden1(x)
        x = func.relu(x)
        x = self.hidden2(x)
        x = func.relu(x)

        x1 = self.value(x)
        x1 = func.relu(x1)

        x2 = self.advantage(x)
        x2 = func.relu(x2)

        state_action_value = x1 + (x2 - x2.mean())
        return state_action_value


class Dueling_DQN(DQN):
    def __init__(self,
                 gamma,
                 epsilon,
                 learning_rate,
                 memory_capacity,
                 batch_size,
                 target_replace_iter,
                 modelFileXML):
        super(Dueling_DQN, self).__init__(gamma, epsilon, learning_rate, memory_capacity, batch_size, target_replace_iter, modelFileXML)
        '''Re-define NN'''
        self.eval_net = DuelingNeuralNetwork(_input=self.state_dim_nn, _output=self.action_dim_nn).to(device)
        self.target_net = DuelingNeuralNetwork(_input=self.state_dim_nn, _output=self.action_dim_nn).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()
        '''Re-define NN'''

    def DuelingDQN_info(self):
        print('This is Dueling DQN:')
        self.DQN_info()
