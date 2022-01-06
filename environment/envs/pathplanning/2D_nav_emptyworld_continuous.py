import math
from environment.envs.pathplanning.samplingmap import samplingmap
from environment.config.xml_write import *
from algorithm.rl_base.rl_base import rl_base
import pandas as pd


class twoD_Nav_Empty(samplingmap, rl_base):
    def __init__(self, samplingMap_dict: dict, vRange: list, aRange: list, jRange: list, save_cfg: bool):
        super(twoD_Nav_Empty, self).__init__(samplingMap_dict['width'],
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
        self.initJ = [0., 0.]
        self.terminalP = self.terminal.copy()
        self.vRange = vRange
        self.aRange = aRange
        self.jRange = jRange
        self.p = self.initP.copy()
        self.v = self.initV.copy()
        self.a = self.initA.copy()
        self.j = self.initJ.copy()
        self.dt = 0.01
        self.time = 0.0
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时
        self.tMax = 2.0 * max(math.pow(6.0*self.x_size/self.jRange[0], 1/3), math.pow(6.0*self.y_size/self.jRange[1], 1/3))
        '''physical parameters'''

        '''rl_base'''
        self.state_dim = 6
        self.state_num = [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf]
        self.state_step = [None, None, None, None, None, None]
        self.state_space = [None, None, None, None, None, None]
        self.state_range = [[-1.0, 1.0], [-1.0, 1.0], self.vRange.copy(), self.vRange.copy(), self.aRange.copy(), self.aRange.copy()]
        self.isStateContinuous = [True, True, True, True, True, True]

        self.action_dim = 2
        self.action_num = [math.inf, math.inf]
        self.action_step = [None, None]
        self.action_space = [None, None]
        self.isActionContinuous = [True, True]
        self.action_range = [self.jRange.copy(), self.jRange.copy()]

        self.initial_state = [(self.terminalP[0] - self.p[0]) / self.x_size,
                              (self.terminalP[1] - self.p[1]) / self.y_size,
                              self.initV[0],
                              self.initV[1],
                              self.initA[0],
                              self.initA[1]]
        self.initial_action = self.initJ.copy()
        self.current_state = self.initial_state.copy()
        self.next_state = self.current_state.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        '''rl_base'''

        '''visualization_opencv'''
        # Inherited in samplingmap.
        '''visualization_opencv'''

        '''data_save'''
        self.Px = [self.p[0]]
        self.Py = [self.p[1]]
        self.Vx = [self.v[0]]
        self.Vy = [self.v[1]]
        self.Ax = [self.a[0]]
        self.Ay = [self.a[1]]
        self.Jx = [self.j[0]]
        self.Jy = [self.j[1]]
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
        self.v[0] = min(max(self.p[1], self.vRange[0]), self.vRange[1])
        self.v[1] = min(max(self.p[1], self.vRange[0]), self.vRange[1])
        self.a[0] = min(max(self.p[1], self.aRange[0]), self.aRange[1])
        self.a[1] = min(max(self.p[1], self.aRange[0]), self.aRange[1])

    def is_Terminal(self):
        if (self.p[0] <= 0.0) or (self.p[0] >= self.x_size):
            self.terminal_flag = 1
            return True
        if (self.p[1] <= 0.0) or (self.p[1] >= self.y_size):
            self.terminal_flag = 1
            return True
        if self.time > self.tMax:
            self.terminal_flag = 2
            return True
        self.terminal_flag = 0
        return False

    def step_update(self, action: list):
        self.j = self.action_saturation(action).copy()      # 动作饱和处理
        # self.state_saturation()                             # 状态饱和处理
        self.current_state = [(self.terminalP[0] - self.p[0]) / self.x_size,
                              (self.terminalP[1] - self.p[1]) / self.y_size,
                              self.initV[0],
                              self.initV[1],
                              self.initA[0],
                              self.initA[1]]

        '''step update'''
        self.p = [self.p[i] + self.v[i] * self.dt + 1 / 2 * self.a[i] * self.dt ** 2 + 1 / 6 * self.j[i] * self.dt ** 3 for i in range(2)]
        self.v = [self.v[i] + self.a[i] * self.dt + 1 / 2 * self.j[i] * self.dt ** 2 for i in range(2)]
        self.a = [self.a[i] + self.dt * self.j[i] for i in range(2)]
        self.is_terminal = self.is_Terminal()       # 刚刚更新完状态，就需要立即进行会和终止检测
        self.time = self.time + self.dt
        self.next_state = [(self.terminalP[0] - self.p[0]) / self.x_size,
                           (self.terminalP[1] - self.p[1]) / self.y_size,
                           self.initV[0],
                           self.initV[1],
                           self.initA[0],
                           self.initA[1]]
        '''step update'''

        '''reward function'''
        self.reward = 0
        '''reward function'''

        self.saveData()

        return self.current_state, action, self.reward, self.next_state, self.is_terminal

    def saveModel2XML(self, filename='2D_Nav_EmptyWorld_Continuous.xml', filepath='../../config/'):
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
            'initJ': self.initJ,
            'terminalP': self.terminalP,
            'vRange': self.vRange,
            'aRange': self.aRange,
            'jRange': self.jRange,
            'p': self.p,
            'v': self.v,
            'a': self.a,
            'j': self.j,
            'dt': self.dt,
            'tMax': self.tMax
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
        self.Jx.append(self.j[0])
        self.Jy.append(self.j[1])
        self.Time.append(self.time)
        if is2file:
            data = pd.DataFrame({
                'Px:': self.Px,
                'Py:': self.Py,
                'Vx': self.Vx,
                'Vy': self.Vy,
                'Ax': self.Ax,
                'Ay': self.Ay,
                'Jx': self.Jx,
                'Jy': self.Jy,
                'Time': self.Time
            })
            data.to_csv(filepath + filename, index=False, sep=',')
