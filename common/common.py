import math
import random
import re
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
import torch


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


def cal_vector_rad(v1, v2):
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


def cross_product(vec1: list, vec2: list) -> float:
    """
    :brief:         cross product of two vectors
    :param vec1:    vector1
    :param vec2:    vector2
    :return:        cross product
    """
    return vec1[0] * vec2[1] - vec2[0] * vec1[1]


def dis_two_points(point1: list, point2: list) -> float:
    """
    :brief:         euclidean distance between two points
    :param point1:  point1
    :param point2:  point2
    :return:        euclidean distance
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def point_is_in_circle(center: list, r: float, point: list) -> bool:
    """
    :brief:         if a point is in a circle
    :param center:  center of the circle
    :param r:       radius of the circle
    :param point:   point
    :return:        if the point is in the circle
    """
    sub = [center[i] - point[i] for i in [0, 1]]
    return np.linalg.norm(sub) <= r


def point_is_in_ellipse(long: float, short: float, rotate_angle: float, center: list, point: list) -> bool:
    """
    :brief:                     判断点是否在椭圆内部
    :param long:                长轴
    :param short:               短轴
    :param rotate_angle:        椭圆自身的旋转角度
    :param center:              中心点
    :param point:               待测点
    :return:                    bool
    """
    sub = np.array([point[i] - center[i] for i in [0, 1]])
    trans = np.array([[cosd(-rotate_angle), -sind(-rotate_angle)], [sind(-rotate_angle), cosd(-rotate_angle)]])
    [x, y] = list(np.dot(trans, sub))
    return (x / long) ** 2 + (y / short) ** 2 <= 1


def point_is_in_poly(center, r, points: list, point: list) -> bool:
    """
    :brief:                     if a point is in a polygon
    :param center:              center of the circumcircle of the polygon
    :param r:                   radius of the circumcircle of the polygon
    :param points:              points of the polygon
    :param point:               the point to be tested
    :return:                    if the point is in the polygon
    """
    if center and r:
        if point_is_in_circle(center, r, point) is False:
            return False
    '''若在多边形对应的外接圆内，再进行下一步判断'''
    l_pts = len(points)
    res = False
    j = l_pts - 1
    for i in range(l_pts):
        if ((points[i][1] > point[1]) != (points[j][1] > point[1])) and \
                (point[0] < (points[j][0] - points[i][0]) * (point[1] - points[i][1]) / (
                        points[j][1] - points[i][1]) + points[i][0]):
            res = not res
        j = i
    if res is True:
        return True


def line_is_in_ellipse(long: float, short: float, rotate_angle: float, center: list, point1: list,
                       point2: list) -> bool:
    """
    :brief:                     判断线段与椭圆是否有交点
    :param long:                长轴
    :param short:               短轴
    :param rotate_angle:        椭圆自身的旋转角度
    :param center:              中心点
    :param point1:              待测点1
    :param point2:              待测点2
    :return:                    bool
    """
    if point_is_in_ellipse(long, short, rotate_angle, center, point1):
        return True
    if point_is_in_ellipse(long, short, rotate_angle, center, point2):
        return True
    pt1 = [point1[i] - center[i] for i in [0, 1]]
    pt2 = [point2[j] - center[j] for j in [0, 1]]  # 平移至原点

    pptt1 = [pt1[0] * cosd(-rotate_angle) - pt1[1] * sind(-rotate_angle),
             pt1[0] * sind(-rotate_angle) + pt1[1] * cosd(-rotate_angle)]
    pptt2 = [pt2[0] * cosd(-rotate_angle) - pt2[1] * sind(-rotate_angle),
             pt2[0] * sind(-rotate_angle) + pt2[1] * cosd(-rotate_angle)]

    if pptt1[0] == pptt2[0]:
        if short ** 2 * (1 - pptt1[0] ** 2 / long ** 2) < 0:
            return False
        else:
            y_cross = math.sqrt(short ** 2 * (1 - pptt1[0] ** 2 / long ** 2))
            if max(pptt1[1], pptt2[1]) >= y_cross >= -y_cross >= min(pptt1[1], pptt2[1]):
                return True
            else:
                return False
    else:
        k = (pptt2[1] - pptt1[1]) / (pptt2[0] - pptt1[0])
        b = pptt1[1] - k * pptt1[0]
        ddelta = (long * short) ** 2 * (short ** 2 + long ** 2 * k ** 2 - b ** 2)
        if ddelta < 0:
            return False
        else:
            x_medium = -(k * b * long ** 2) / (short ** 2 + long ** 2 * k ** 2)
            if max(pptt1[0], pptt2[0]) >= x_medium >= min(pptt1[0], pptt2[0]):
                return True
            else:
                return False


def line_is_in_circle(center: list, r: float, point1: list, point2: list) -> bool:
    """
    :brief:             if a circle and a line segment have an intersection
    :param center:      center of the circle
    :param r:           radius of the circle
    :param point1:      point1 of the line segment
    :param point2:      point2 of t he line segment
    :return:            if the circle and the line segment have an intersection
    """
    return line_is_in_ellipse(r, r, 0, center, point1, point2)


def line_is_in_poly(center: list, r: float, points: list, point1: list, point2: list) -> bool:
    """
    :brief:             if a polygon and a line segment have an intersection
    :param center:      center of the circumcircle of the polygon
    :param r:           radius of the circumcircle of the polygon
    :param points:      points of the polygon
    :param point1:      the first point of the line segment
    :param point2:      the second point of the line segment
    :return:            if the polygon and the line segment have an intersection
    """
    if point_is_in_poly(center, r, points, point1):
        # print('Something wrong happened...')
        return True
    if point_is_in_poly(center, r, points, point2):
        # print('Something wrong happened...')
        return True
    length = len(points)
    for i in range(length):
        a = points[i % length]
        b = points[(i + 1) % length]
        c = point1.copy()
        d = point2.copy()
        '''通过坐标变换将a点变到原点'''
        b = [b[i] - a[i] for i in [0, 1]]
        c = [c[i] - a[i] for i in [0, 1]]
        d = [d[i] - a[i] for i in [0, 1]]
        a = [a[i] - a[i] for i in [0, 1]]
        '''通过坐标变换将a点变到原点'''

        '''通过坐标旋转将b点变到与X重合'''
        l_ab = dis_two_points(a, b)  # length of ab
        cos = b[0] / l_ab
        sin = b[1] / l_ab
        bb = [cos * b[0] + sin * b[1], -sin * b[0] + cos * b[1]]
        cc = [cos * c[0] + sin * c[1], -sin * c[0] + cos * c[1]]
        dd = [cos * d[0] + sin * d[1], -sin * d[0] + cos * d[1]]
        '''通过坐标旋转将b点变到与X重合'''

        if cc[1] * dd[1] > 0:
            '''如果变换后的cd纵坐标在x轴的同侧'''
            # return False
            continue
        else:
            '''如果变换后的cd纵坐标在x轴的异侧(包括X轴)'''
            if cc[0] == dd[0]:
                '''k == inf'''
                if min(bb) <= cc[0] <= max(bb):
                    return True
                else:
                    continue
            else:
                '''k != inf'''
                k_cd = (dd[1] - cc[1]) / (dd[0] - cc[0])
                b_cd = cc[1] - k_cd * cc[0]
                if k_cd != 0:
                    x_cross = -b_cd / k_cd
                    if min(bb) <= x_cross <= max(bb):
                        return True
                    else:
                        continue
                else:
                    '''k_cd == 0'''
                    if (min(bb) <= cc[0] <= max(bb)) or (min(bb) <= dd[0] <= max(bb)):
                        return True
                    else:
                        continue
    return False


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
