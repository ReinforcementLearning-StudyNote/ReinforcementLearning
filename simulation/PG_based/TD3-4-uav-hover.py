import math
import os
import sys
import datetime
import time
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
# import copy
from environment.envs.UAV.uav_hover import UAV_Hover
from algorithm.actor_critic.Twin_Delayed_DDPG import Twin_Delayed_DDPG as TD3
from common.common_func import *
from common.common_cls import *


cfgPath = '../../environment/config/'
cfgFile = 'UGV_Forward_Obstacle_Continuous.xml'
optPath = '../../datasave/network/'


class CriticNetWork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, name, chkpt_dir):
        super(CriticNetWork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

        self.fc1 = nn.Linear(self.state_dim, 128)  # state -> hidden1
        self.batch_norm1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 64)  # hidden1 -> hidden2
        self.batch_norm2 = nn.LayerNorm(64)

        self.action_value = nn.Linear(self.action_dim, 64)  # action -> hidden2
        self.q = nn.Linear(64, 1)  # hidden2 -> output action value

        # self.initialization()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, _action):
        state_value = self.fc1(state)  # forward
        state_value = self.batch_norm1(state_value)  # batch normalization
        state_value = func.relu(state_value)  # relu

        state_value = self.fc2(state_value)
        state_value = self.batch_norm2(state_value)

        action_value = func.relu(self.action_value(_action))
        state_action_value = func.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def initialization_default(self):
        self.fc1.reset_parameters()
        self.batch_norm1.reset_parameters()
        self.fc2.reset_parameters()
        self.batch_norm2.reset_parameters()

        self.action_value.reset_parameters()
        self.q.reset_parameters()

    def initialization(self):
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3 = 0.003
        nn.init.uniform_(self.q.weight.data, -f3, f3)
        nn.init.uniform_(self.q.bias.data, -f3, f3)

    def save_checkpoint(self, name=None, path='', num=None):
        print('...saving checkpoint...')
        if name is None:
            torch.save(self.state_dict(), self.checkpoint_file)
        else:
            if num is None:
                torch.save(self.state_dict(), path + name)
            else:
                torch.save(self.state_dict(), path + name + str(num))

    def save_all_net(self):
        print('...saving all net...')
        torch.save(self, self.checkpoint_file_whole_net)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim1, state_dim2, action_dim, name, chkpt_dir):
        super(ActorNetwork, self).__init__()
        self.state_dim1 = state_dim1
        self.state_dim2 = state_dim2
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

        self.linear11 = nn.Linear(self.state_dim1, 128)  # 第一部分网络第一层
        self.batch_norm11 = nn.LayerNorm(128)
        self.linear12 = nn.Linear(128, 64)  # 第一部分网络第二层
        self.batch_norm12 = nn.LayerNorm(64)
        self.linear13 = nn.Linear(64, 64)
        self.batch_norm13 = nn.LayerNorm(64)  # 第一部分网络第三层

        self.linear21 = nn.Linear(self.state_dim2, 128)  # 第二部分网络第一层
        self.batch_norm21 = nn.LayerNorm(128)
        self.linear22 = nn.Linear(128, 64)  # 第二部分网络第二层
        self.batch_norm22 = nn.LayerNorm(64)
        self.linear23 = nn.Linear(64, 32)
        self.batch_norm23 = nn.LayerNorm(32)  # 第二部分网络第三层

        self.mu = nn.Linear(64 + 32, self.action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def initialization_default(self):
        self.linear11.reset_parameters()
        self.batch_norm11.reset_parameters()
        self.linear12.reset_parameters()
        self.batch_norm12.reset_parameters()
        self.linear13.reset_parameters()
        self.batch_norm13.reset_parameters()

        self.linear21.reset_parameters()
        self.batch_norm21.reset_parameters()
        self.linear22.reset_parameters()
        self.batch_norm22.reset_parameters()
        self.linear23.reset_parameters()
        self.batch_norm23.reset_parameters()

        # self.combine.reset_parameters()
        self.mu.reset_parameters()

    def forward(self, state):
        """
        :param state:
        :return:            output of the net
        """
        if state.dim() == 1:
            split_state = torch.split(state, [self.state_dim1, self.state_dim2], dim=0)
        else:
            split_state = torch.split(state, [self.state_dim1, self.state_dim2], dim=1)
        x1 = self.linear11(split_state[0])
        x1 = self.batch_norm11(x1)
        x1 = func.relu(x1)

        x1 = self.linear12(x1)
        x1 = self.batch_norm12(x1)
        x1 = func.relu(x1)

        x1 = self.linear13(x1)
        x1 = self.batch_norm13(x1)
        x1 = func.relu(x1)  # 该合并了

        x2 = self.linear21(split_state[1])
        x2 = self.batch_norm21(x2)
        x2 = func.relu(x2)

        x2 = self.linear22(x2)
        x2 = self.batch_norm22(x2)
        x2 = func.relu(x2)

        x2 = self.linear23(x2)
        x2 = self.batch_norm23(x2)
        x2 = func.relu(x2)  # 该合并了

        x = torch.cat((x1, x2)) if x1.dim() == 1 else torch.cat((x1, x2), dim=1)
        # print(x1.size(), x2.size(), x.size())
        # x = self.combine(x)
        # x = func.relu(x)

        x = torch.tanh(self.mu(x))
        return x

    def save_checkpoint(self, name=None, path='', num=None):
        print('...saving checkpoint...')
        if name is None:
            torch.save(self.state_dict(), self.checkpoint_file)
        else:
            if num is None:
                torch.save(self.state_dict(), path + name)
            else:
                torch.save(self.state_dict(), path + name + str(num))

    def save_all_net(self):
        print('...saving all net...')
        torch.save(self, self.checkpoint_file_whole_net)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))