import math
import re
import numpy as np


def deg2rad(deg: float) -> float:
    """
    :brief:         omit
    :param deg:     degree
    :return:        radian
    """
    return deg * math.pi / 180.0


def rad2deg(rad: float) -> float:
    """
    :brief:         omit
    :param rad:     radian
    :return:        degree
    """
    return rad * 180.8 / math.pi


def str2list(string: str) -> list:
    """
    :brief:         transfer a string to list，必须是具备特定格式的
    :param string:  string
    :return:        the list
    """
    res = re.split(r'[\[\]]', string.strip())
    inner = []
    outer = []
    for item in res:
        item.strip()
    while '' in res:
        res.remove('')
    while ', ' in res:
        res.remove(', ')
    while ',' in res:
        res.remove(',')
    while ' ' in res:
        res.remove(' ')
    for _res in res:
        _res_spilt = re.split(r',', _res)
        for item in _res_spilt:
            inner.append(float(item))
        outer.append(inner.copy())
        inner.clear()
    return outer


def sind(theta):
    """
    :param theta:   degree, not rad
    :return:
    """
    return math.sin(theta / 180.0 * math.pi)


def cosd(theta):
    """
    :param theta:   degree, not rad
    :return:
    """
    return math.cos(theta / 180.0 * math.pi)


class ReplayBuffer:
    def __init__(self, max_size, batch_size, state_dim, action_dim):
        self.mem_size = max_size
        self.mem_counter = 0
        self.batch_size = batch_size
        self.s_mem = np.zeros((self.mem_size, state_dim))
        self._s_mem = np.zeros((self.mem_size, state_dim))
        self.a_mem = np.zeros((self.mem_size, action_dim))
        self.r_mem = np.zeros(self.mem_size)
        self.end_mem = np.zeros(self.mem_size, dtype=np.float)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size
        self.s_mem[index] = state
        self.a_mem[index] = action
        self.r_mem[index] = reward
        self._s_mem[index] = state_
        self.end_mem[index] = 1 - done
        self.mem_counter += 1

    def store_transition_per_episode(self, states, actions, rewards, states_, dones):
        num = len(states)
        for i in range(num):
            self.store_transition(states[i], actions[i], rewards[i], states_[i], dones[i])

    def sample_buffer(self):
        max_mem = min(self.mem_counter, self.mem_size)
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
