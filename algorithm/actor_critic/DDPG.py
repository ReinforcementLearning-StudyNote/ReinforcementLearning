import random
import torch.nn as nn
import torch.nn.functional as func
import torch
import numpy as np
from environment.config.xml_write import xml_cfg
from common.common import *
import pandas as pd

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = True
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""

class Actor_NetWork(nn.Module):
    def __init__(self, _input, _output):
        super(Actor_NetWork, self).__init__()


class Critic_NetWork(nn.Module):
    def __init__(self, _input, _output):
        super(Critic_NetWork, self).__init__()


class DDPG:
    def __init__(self,
                 gamma: float = 0.9,
                 actor_learning_rate: float = 1e-3,
                 critic_learning_rate: float = 1e-3,
                 actor_soft_update: float = 1e-2,
                 critic_soft_update: float = 1e-2,
                 memory_capacity: int = 5000,
                 batch_size: int = 64,
                 target_replace_iter: int = 100,
                 modelFileXML: str = ''
                 ):
        """
        :param gamma:                   discount factor
        :param actor_learning_rate:     learning rate of actor net
        :param critic_learning_rate:    learning rate of critic net
        :param actor_soft_update:       soft update rate of actor
        :param critic_soft_update:      soft update rate of critic
        :param memory_capacity:         capacity of the replay memory
        :param batch_size:              batch size
        :param target_replace_iter:
        :param modelFileXML:            model file
        """
        '''DDPG'''
        self.gamma = gamma
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.actor_tau = actor_soft_update
        self.critic_tau = critic_soft_update
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.target_replace_iter = target_replace_iter
        '''DDPG'''

        '''From rl_base'''
        # DDPG 要求智能体状态必须是连续的，动作必须连续的
        self.agentName, self.state_dim_nn, self.action_dim_nn, self.action_dim_physical, self.action_num = \
            self.get_RLBase_from_XML(modelFileXML)
        # agentName:            the name of the agent
        # state_dim_nn:         the dimension of the neural network input
        # action_dim_nn:        the dimension of the neural network output
        # action_dim_physical:  the dimension of the physical action
        '''From rl_base'''

    def get_RLBase_from_XML(self, filename):
        rl_base, agentName = self.load_rl_basefromXML(filename=filename)
        state_dim_nn = int(rl_base['state_dim'])  # input dimension of NN
        action_space = str2list(rl_base['action_space'])  #
        action_dim_nn = 1
        action_dim_physical = len(action_space)
        action_num = []
        for item in action_space:
            action_num.append(len(item))
            action_dim_nn *= len(item)
        return agentName, state_dim_nn, action_dim_nn, action_space, action_dim_physical, action_num

    @staticmethod
    def load_rl_basefromXML(filename: str) -> (dict, str):
        """
        :brief:             从模型文件中加载数据到DQN中
        :param filename:    模型文件
        :return:            数据字典
        """
        root = xml_cfg().XML_Load(filename)
        return xml_cfg().XML_GetTagValue(node=xml_cfg().XML_FindNode(nodename='RL_Base', root=root)), root.attrib['name']
