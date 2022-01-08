import math
import random

from environment.envs.pathplanning.samplingmap import samplingmap
from environment.config.xml_write import *
from algorithm.rl_base.rl_base import rl_base
import pandas as pd
import cv2 as cv
from environment.Color import Color


class Nav_EmptyWorld_Continuous(samplingmap, rl_base):
    def __init__(self, samplingMap_dict: dict, vRange: list, aRange: list, save_cfg: bool):
        super(Nav_EmptyWorld_Continuous, self).__init__(samplingMap_dict['width'],
                                                        samplingMap_dict['height'],
                                                        samplingMap_dict['x_size'],
                                                        samplingMap_dict['y_size'],
                                                        samplingMap_dict['image_name'],
                                                        samplingMap_dict['start'],
                                                        samplingMap_dict['terminal'],
                                                        samplingMap_dict['obs'],
                                                        samplingMap_dict['draw'])

        '''physical parameters'''
        self.initP = self.start.copy()
        self.initV = [0., 0.]
        self.initA = [0., 0.]
        self.terminalP = self.terminal.copy()
        self.vRange = vRange
        self.aRange = aRange
        self.p = self.initP.copy()
        self.v = self.initV.copy()
        self.a = self.initA.copy()
        self.dt = 0.01
        self.time = 0.0
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功
        self.tMax = 6
        self.miss = 0.4
        '''physical parameters'''

        '''rl_base'''
        self.static_gain = 10.0
        self.state_dim = 4
        self.state_num = [math.inf, math.inf, math.inf, math.inf]
        self.state_step = [None, None, None, None]
        self.state_space = [None, None, None, None]
        self.state_range = [[-self.static_gain, self.static_gain],
                            [-self.static_gain, self.static_gain],
                            self.vRange.copy(),
                            self.vRange.copy()]
        self.isStateContinuous = [True, True, True, True, True, True]

        self.action_dim = 2
        self.action_num = [math.inf, math.inf]
        self.action_step = [None, None]
        self.action_space = [None, None]
        self.isActionContinuous = [True, True]
        self.action_range = [self.aRange.copy(), self.aRange.copy()]

        self.initial_state = [(self.terminalP[0] - self.p[0]) / self.x_size * self.static_gain,
                              (self.terminalP[1] - self.p[1]) / self.y_size * self.static_gain,
                              self.initV[0],
                              self.initV[1]]
        self.initial_action = self.initA.copy()
        self.current_state = self.initial_state.copy()
        self.next_state = self.current_state.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        '''rl_base'''

        '''visualization_opencv'''
        # Inherited in samplingmap.
        self.save = self.image.copy()
        '''visualization_opencv'''

        '''data_save'''
        self.Px = [self.p[0]]
        self.Py = [self.p[1]]
        self.Vx = [self.v[0]]
        self.Vy = [self.v[1]]
        self.Ax = [self.a[0]]
        self.Ay = [self.a[1]]
        self.Time = [self.time]
        '''data_save'''
        if save_cfg:
            self.saveModel2XML()

    def action_saturation(self, action: list):
        for i in range(self.action_dim):
            action[i] = max(min(action[i], self.action_range[i][1]), self.action_range[i][0])
        self.current_action = action.copy()
        return action

    def state_saturation(self):
        self.p[0] = min(max(self.p[0], 0.0), self.x_size)
        self.p[1] = min(max(self.p[1], 0.0), self.y_size)
        self.v[0] = min(max(self.v[0], self.vRange[0]), self.vRange[1])
        self.v[1] = min(max(self.v[1], self.vRange[0]), self.vRange[1])
        self.a[0] = min(max(self.a[0], self.aRange[0]), self.aRange[1])
        self.a[1] = min(max(self.a[1], self.aRange[0]), self.aRange[1])

    def show_dynamic_image(self, isWait=False):
        self.image = self.image_temp.copy()
        self.map_draw_boundary()
        cv.circle(self.image, self.dis2pixel(self.p), 5, Color().Red, -1)
        cv.circle(self.image, self.dis2pixel(self.terminal), 5, Color().Blue, -1)
        cv.putText(self.image, 'time:' + str(round(self.time, 3)) + 's', (10, 20), cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Red, 1)
        self.save = self.image.copy()
        cv.imshow(self.name4image, self.image)
        cv.waitKey(0) if isWait else cv.waitKey(1)

    def is_Terminal(self):
        if (self.p[0] <= 0.0) or (self.p[0] >= self.x_size):
            self.terminal_flag = 1
            print('...out...')
            return True
        if (self.p[1] <= 0.0) or (self.p[1] >= self.y_size):
            self.terminal_flag = 1
            print('...out...')
            return True
        if self.time > self.tMax:
            self.terminal_flag = 2
            print('...time out...')
            return True
        if self.dis_two_points(self.p, self.terminalP) < self.miss:
            self.terminal_flag = 3
            print('...success...')
            return True
        self.terminal_flag = 0
        return False

    def step_update(self, action: list):
        self.a = self.action_saturation(action).copy()  # 动作饱和处理
        self.current_state = [(self.terminalP[0] - self.p[0]) / self.x_size * self.static_gain,
                              (self.terminalP[1] - self.p[1]) / self.y_size * self.static_gain,
                              self.v[0],
                              self.v[1]]

        '''step update'''
        self.p = [self.p[i] + self.v[i] * self.dt + 1 / 2 * self.a[i] * self.dt ** 2 for i in range(2)]
        self.v = [self.v[i] + self.a[i] * self.dt for i in range(2)]
        self.time = self.time + self.dt
        '''这个操作不会影响结果，只是为了设置奖励函数合理'''
        # if self.p[0] >= self.x_size:         # x方向出界，vx=0，vy变为原来0.6倍
        #     self.p[0] = self.x_size
        #     self.v[0] = -self.v[0]
        #     self.v[1] = 0.8 * self.v[1]
        # if self.p[0] <= 0:                  # x方向出界，vx=0，vy变为原来0.6倍
        #     self.p[0] = 0
        #     self.v[0] = -self.v[0]
        #     self.v[1] = 0.8 * self.v[1]
        # if self.p[1] > self.y_size:
        #     self.p[1] = self.y_size
        #     self.v[0] = 0.8 * self.v[0]
        #     self.v[1] = -self.v[1]
        # if self.p[1] <= 0:
        #     self.p[1] = 0
        #     self.v[0] = 0.8 * self.v[0]
        #     self.v[1] = -self.v[1]
        '''这个操作不会影响结果，只是为了设置奖励函数合理'''
        self.is_terminal = self.is_Terminal()  # 刚刚更新完状态，就需要立即进行会和终止检测
        self.next_state = [(self.terminalP[0] - self.p[0]) / self.x_size * self.static_gain,
                           (self.terminalP[1] - self.p[1]) / self.y_size * self.static_gain,
                           self.v[0],
                           self.v[1]]
        if not self.is_terminal:
            self.state_saturation()  # 如果不是回合终止，那么做状态饱和处理，否则做了也没有意义
        '''step update'''

        '''reward function'''
        '''success 奖励'''
        r3 = 0
        # if self.terminal_flag == 3:
        #     r3 = 2000
        '''success 奖励'''

        '''out奖励'''
        r2 = 0
        # if self.terminal_flag == 1:
        #     r2 = -2000
        '''out奖励'''

        '''距离误差奖励'''
        dis = self.dis_two_points(self.p, self.terminalP)
        gain = 0.1
        # r1 = -gain * dis ** 2
        r1 = -math.fabs(dis)
        '''距离误差奖励'''
        self.reward = r1 + r2 + r3
        '''reward function'''

        self.saveData()

        return self.current_state, action, self.reward, self.next_state, self.is_terminal

    def reset(self):
        """
        :return:    None
        """
        '''physical parameters'''
        self.initP = self.start.copy()
        self.initV = [0., 0.]
        self.initA = [0., 0.]
        self.p = self.initP.copy()
        self.v = self.initV.copy()
        self.a = self.initA.copy()
        self.terminalP = self.terminal.copy()
        self.time = 0.0
        self.terminal_flag = 0
        '''physical parameters'''

        '''RL_BASE'''
        self.initial_state = [(self.terminalP[0] - self.p[0]) / self.x_size * self.static_gain,
                              (self.terminalP[1] - self.p[1]) / self.y_size * self.static_gain,
                              self.initV[0],
                              self.initV[1]]
        self.initial_action = self.initA.copy()
        self.current_state = self.initial_state.copy()
        self.next_state = self.current_state.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        '''RL_BASE'''

        '''data_save'''
        self.Px = [self.p[0]]
        self.Py = [self.p[1]]
        self.Vx = [self.v[0]]
        self.Vy = [self.v[1]]
        self.Ax = [self.a[0]]
        self.Ay = [self.a[1]]
        self.Time = [self.time]
        '''data_save'''

    def reset_random(self):
        """
        :return:    None
        """
        '''physical parameters'''
        # self.set_start([random.uniform(0, self.x_size), random.uniform(0, self.y_size)])  # 随机重置地图的 start
        self.set_terminal([random.uniform(0, self.x_size), random.uniform(0, self.y_size)])  # 随机重置地图的 terminal
        self.initP = self.start.copy()
        self.initV = [0., 0.]
        self.initA = [0., 0.]
        self.p = self.initP.copy()
        self.v = self.initV.copy()
        self.a = self.initA.copy()
        self.terminalP = self.terminal.copy()
        self.time = 0.0
        self.terminal_flag = 0
        print('...init position...:', self.initP)
        '''physical parameters'''

        '''RL_BASE'''
        self.initial_state = [(self.terminalP[0] - self.p[0]) / self.x_size * self.static_gain,
                              (self.terminalP[1] - self.p[1]) / self.y_size * self.static_gain,
                              self.initV[0],
                              self.initV[1]]
        self.current_state = self.initial_state.copy()
        self.next_state = self.current_state.copy()
        self.initial_action = self.initA.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        '''RL_BASE'''

        '''data_save'''
        self.Px = [self.p[0]]
        self.Py = [self.p[1]]
        self.Vx = [self.v[0]]
        self.Vy = [self.v[1]]
        self.Ax = [self.a[0]]
        self.Ay = [self.a[1]]
        self.Time = [self.time]
        '''data_save'''

    def saveModel2XML(self, filename='Nav_EmptyWorld_Continuous.xml', filepath='../config/'):
        rootMsg = {
            'name': 'Flight_Attitude_Simulator_Continuous',
            'author': 'Yefeng YANG',
            'date': '2022.01.05',
            'E-mail': 'yefeng.yang@connect.polyu.hk; 18B904013@stu.hit.edu.cn'
        }
        rl_baseMsg = {
            'state_dim': self.state_dim,
            'state_num': self.state_num,
            'state_step': self.state_step,
            'state_space': self.state_space,
            'state_range': self.state_range,
            'isStateContinuous': self.isStateContinuous,
            'initial_state': self.initial_state,
            'current_state': self.current_state,
            'next_state': self.next_state,
            'action_dim': self.action_dim,
            'action_step': self.action_step,
            'action_range': self.action_range,
            'action_num': self.action_num,
            'action_space': self.action_space,
            'isActionContinuous': self.isActionContinuous,
            'initial_action': self.initial_action,
            'current_action': self.current_action,
            'is_terminal': self.is_terminal
        }
        physicalMsg = {
            'x_size': self.x_size,
            'y_size': self.y_size,
            'initP': self.initP,
            'initV': self.initV,
            'initA': self.initA,
            'terminalP': self.terminalP,
            'vRange': self.vRange,
            'aRange': self.aRange,
            'p': self.p,
            'v': self.v,
            'a': self.a,
            'dt': self.dt,
            'tMax': self.tMax,
            'miss': self.miss
        }
        imageMsg = {
            'width': self.width,
            'height': self.height,
            'name4image': self.name4image,
            'x_offset': self.x_offset,
            'y_offset': self.y_offset,
            'pixel_per_meter': self.pixel_per_meter
        }
        xml_cfg().XML_Create(filename=filepath + filename,
                             rootname='Plant',
                             rootmsg=rootMsg)
        xml_cfg().XML_InsertNode(filename=filepath + filename,
                                 nodename='RL_Base',
                                 nodemsg=rl_baseMsg)
        xml_cfg().XML_InsertNode(filename=filepath + filename,
                                 nodename='Physical',
                                 nodemsg=physicalMsg)
        xml_cfg().XML_InsertNode(filename=filepath + filename,
                                 nodename='Image',
                                 nodemsg=imageMsg)
        xml_cfg().XML_Pretty_All(filepath + filename)

    def saveData(self, is2file=False, filename='2D_Nav_EmptyWorld_Continuous.csv', filepath=''):
        self.Px.append(self.p[0])
        self.Py.append(self.p[1])
        self.Vx.append(self.v[0])
        self.Vy.append(self.v[1])
        self.Ax.append(self.a[0])
        self.Ay.append(self.a[1])
        self.Time.append(self.time)
        if is2file:
            data = pd.DataFrame({
                'Px:': self.Px,
                'Py:': self.Py,
                'Vx': self.Vx,
                'Vy': self.Vy,
                'Ax': self.Ax,
                'Ay': self.Ay,
                'Time': self.Time
            })
            data.to_csv(filepath + filename, index=False, sep=',')
