# import math
import random
# import re
import numpy as np
# from numpy import linalg
import torch.nn as nn
import torch.nn.functional as func
import torch
# import scipy.spatial as spt


class ReplayBuffer:
    def __init__(self, max_size, batch_size, state_dim, action_dim):
        # print(state_dim, action_dim)
        self.mem_size = max_size
        self.mem_counter = 0
        self.batch_size = batch_size
        self.s_mem = np.zeros((self.mem_size, state_dim))
        self._s_mem = np.zeros((self.mem_size, state_dim))
        self.a_mem = np.zeros((self.mem_size, action_dim))
        self.r_mem = np.zeros(self.mem_size)
        self.end_mem = np.zeros(self.mem_size, dtype=np.float)
        self.sorted_index = []
        self.resort_count = 0

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size
        self.s_mem[index] = state
        self.a_mem[index] = action
        self.r_mem[index] = reward
        self._s_mem[index] = state_
        self.end_mem[index] = 1 - done
        self.mem_counter += 1

    def get_reward_sort(self):
        """
        :return:        根据奖励大小得到所有数据的索引值，升序，即从小到大
        """
        print('...sorting...')
        self.sorted_index = sorted(range(min(self.mem_counter, self.mem_size)), key=lambda k: self.r_mem[k], reverse=False)

    def store_transition_per_episode(self, states, actions, rewards, states_, dones):
        self.resort_count += 1
        num = len(states)
        for i in range(num):
            self.store_transition(states[i], actions[i], rewards[i], states_[i], dones[i])

    def sample_buffer(self, is_reward_ascent=True):
        max_mem = min(self.mem_counter, self.mem_size)
        if is_reward_ascent:
            batchNum = min(int(0.25 * max_mem), self.batch_size)
            batch = random.sample(self.sorted_index[-int(0.25 * max_mem):], batchNum)
        else:
            batch = np.random.choice(max_mem, self.batch_size)
        states = self.s_mem[batch]
        actions = self.a_mem[batch]
        rewards = self.r_mem[batch]
        actions_ = self._s_mem[batch]
        terminals = self.end_mem[batch]

        return states, actions, rewards, actions_, terminals


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.x0 = x0
        self.dt = dt
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        self.reset()

    def __call__(self):
        # noise = OUActionNoise()
        # noise()
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class GaussianNoise(object):
    def __init__(self, mu):
        self.mu = mu
        self.sigma = 1 / 3

    def __call__(self, sigma=1 / 3):
        return np.random.normal(self.mu, sigma, self.mu.shape)


class CriticNetWork(nn.Module):
    def __init__(self, beta=1e-3, state_dim=1, action_dim=1, name='CriticNetWork', chkpt_dir=''):
        super(CriticNetWork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.beta = beta
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

    def forward(self, state, action):
        state_action_value = func.relu(torch.add(state, action))

        return state_action_value

    def initialization(self):
        pass

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
    def __init__(self, alpha=1e-4, state_dim=1, action_dim=1, name='ActorNetwork', chkpt_dir=''):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

    def initialization(self):
        pass

    def forward(self, state):
        # x = torch.tanh(state)  # bound the output to [-1, 1]
        # return x
        pass

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


class Quaternion:
    def __init__(self):
        pass

    @staticmethod
    def q_add(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        :func:          计算两个四元数相加
        :param q1:      q1
        :param q2:      q2
        :return:        new q
        """
        return q1 + q2

    # def q_im_anti_sym(self) -> np.ndarray:
    #     """
    #     :func:          计算四元数方向部分对应的反对称矩阵
    #     :return:        反对称矩阵
    #     """
    #     m = np.array([[0, -self.q[3], self.q[2]],
    #                   [self.q[3], 0, -self.q[1]],
    #                   [-self.q[2], self.q[1], 0]])
    #     return m

    @staticmethod
    def q_im_cross(q1_im: np.ndarray, q2_im: np.ndarray) -> np.ndarray:
        """
        :func:          四元数虚部叉乘
        :param q1_im:   q1虚部
        :param q2_im:   q2虚部
        :return:        叉乘结果
        """
        return np.cross(q1_im, q2_im)

    @staticmethod
    def q_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        :func:          四元数乘法
        :param q1:      q1
        :param q2:      q2
        :return:        q1 圈乘 q2
        """
        re1, im1 = q1[0], q1[1:4]
        re2, im2 = q2[0], q2[1:4]

        re = re1 * re2 - np.dot(im1, im2)
        im = np.cross(im1, im2) + re1 * im2 + re2 * im1

        return np.array([re, im[0], im[1], im[2]])

    @staticmethod
    def q_conj(q: np.ndarray) -> np.ndarray:
        q[1:4] *= -1
        return q

    @staticmethod
    def q_norm(q: np.ndarray) -> float:
        return np.linalg.norm(q)

    def q_inv(self, q: np.ndarray) -> np.ndarray:
        return self.q_conj(q) / self.q_norm(q)

