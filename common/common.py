import math
import random
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


def points_rotate(pts, theta):
    """
    :param pts:
    :param theta:   rad, counter-clockwise
    :return:        new position
    """
    if type(pts[0]) == list:
        return [[math.cos(theta) * pt[0] - math.sin(theta) * pt[1], math.sin(theta) * pt[0] + math.cos(theta) * pt[1]] for pt in pts]
    else:
        return [math.cos(theta) * pts[0] - math.sin(theta) * pts[1], math.sin(theta) * pts[0] + math.cos(theta) * pts[1]]


def points_move(pts, dis):
    if type(pts[0]) == list:
        return [[pt[0] + dis[0], pt[1] + dis[1]] for pt in pts]
    else:
        return [pts[0] + dis[0], pts[1] + dis[1]]


def cal_vector_degree(v1, v2):
    """
    :brief:         calculate the rad between two vectors
    :param v1:      vector1
    :param v2:      vector2
    :return:        the rad
    """
    # print(v1, v2)
    if np.linalg.norm(v2) < 1e-4 or np.linalg.norm(v1) < 1e-4:
        return 0
    cosTheta = min(max(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1), 1)
    return math.acos(cosTheta)

class ReplayBuffer:
    def __init__(self, max_size, batch_size, state_dim, action_dim):
        print(state_dim, action_dim)
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
        self.sorted_index = sorted(range(min(self.mem_counter, self.mem_size)), key=lambda k: self.r_mem[k], reverse=False)

    def get_reward_resort(self, per):
        if self.resort_count > per:
            print('...resorting...')
            self.resort_count = 0
            self.get_reward_sort()

    def store_transition_per_episode(self, states, actions, rewards, states_, dones):
        self.resort_count += 1
        num = len(states)
        for i in range(num):
            self.store_transition(states[i], actions[i], rewards[i], states_[i], dones[i])

    def sample_buffer(self, is_reward_ascent=True):
        max_mem = min(self.mem_counter, self.mem_size)
        if is_reward_ascent:
            '''倒着数是最好的，从10倍倒着数的数据中随机取出一倍的数据量作为batch'''
            batch = random.sample(self.sorted_index[-self.batch_size * 10:], self.batch_size)
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
