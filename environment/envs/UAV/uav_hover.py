from environment.config.xml_write import *
from environment.envs.UAV.uav import UAV
from environment.envs.UAV.uav_visualization import UAV_Visualization
from algorithm.rl_base.rl_base import rl_base
from common.common_func import *
import pandas as pd


class UAV_Hover(rl_base, UAV):
    def __init__(self,
                 m: float = 0.8,
                 g: float = 9.8,
                 Jxx: float = 4.212e-3,
                 Jyy: float = 4.212e-3,
                 Jzz: float = 8.255e-3,
                 d: float = 0.12,
                 CT: float = 2.168e-6,
                 CM: float = 2.136e-8,
                 J0: float = 1.01e-5,
                 pos0=None,
                 vel0=None,
                 angle0=None,
                 omega0_inertial=None,
                 omega0_body=None,
                 target_pos=None,
                 save_cfg=True):
        super(UAV_Hover, self).__init__(m=m, g=g, Jxx=Jxx, Jyy=Jyy, Jzz=Jzz, d=d, CT=CT, CM=CM, J0=J0,
                                        pos0=pos0, vel0=vel0, angle0=angle0, omega0_inertial=omega0_inertial, omega0_body=omega0_body)

        self.init_pos = np.array(pos0)  # initial position
        self.init_vel = np.array(vel0)  # initial velocity
        self.init_angle = np.array(angle0)  # initial attitude
        self.init_omega0_inertial = np.array(omega0_inertial)  # initial omega in S_i
        self.init_omega0_body = np.array(omega0_body)  # initial omega in S_b
        self.target_pos = np.array(target_pos) if target_pos else self.init_pos
        self.error_pos = self.target_pos - self.pos  # 位置跟踪误差
        self.staticGain = 2

        '''rl_base'''
        self.state_dim = 19  # ex ey ez x y z vx vy vz phi theta psi dphi dtheta dpsi f1 f2 f3 f4
        self.state_num = [math.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.state_range = [[-self.staticGain, self.staticGain] for _ in range(self.state_dim)]
        self.isStateContinuous = [True for _ in range(self.state_dim)]

        self.initial_state = self.state_norm()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.action_dim = 4
        self.action_step = [None for _ in range(self.action_dim)]
        self.action_range = [[self.fmin, self.fmax] for _ in range(self.action_dim)]
        self.action_num = [math.inf for _ in range(self.action_dim)]
        self.action_space = [None for _ in range(self.action_dim)]
        self.isActionContinuous = [True for _ in range(self.action_dim)]
        self.initial_action = [0.0 for _ in range(self.action_dim)]
        self.current_action = self.initial_action.copy()

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功 4-碰撞障碍物
        '''rl_base'''

        '''visualization_opencv'''
        self.uav_vis = UAV_Visualization(xbound=np.array([self.pos_min[0], self.pos_max[0]]),
                                         ybound=np.array([self.pos_min[1], self.pos_max[1]]),
                                         zbound=np.array([self.pos_min[2], self.pos_max[2]]),
                                         origin=self.init_pos)  # visualization
        '''visualization_opencv'''

        '''datasave'''
        self.save_pos = np.atleast_2d(self.pos)
        self.save_vel = np.atleast_2d(self.vel)
        self.save_angle = np.atleast_2d(self.angle)
        self.save_omega_inertial = np.atleast_2d(self.omega_inertial)
        self.save_omega_body = np.atleast_2d(self.omega_body)
        self.save_f = np.atleast_2d(self.force)
        self.save_t = np.array([self.time])
        '''datasave'''
        if save_cfg:
            self.saveModel2XML()

    def state_norm(self) -> list:
        norm_error = self.error_pos / (self.pos_max - self.pos_min) * self.staticGain
        norm_pos = (2 * self.pos - self.pos_max - self.pos_min) / (self.pos_max - self.pos_min) * self.staticGain
        norm_vel = (2 * self.vel - self.vel_max - self.vel_min) / (self.vel_max - self.vel_min) * self.staticGain
        norm_angle = (2 * self.angle - self.angle_max - self.angle_min) / (self.angle_max - self.angle_min) * self.staticGain
        norm_dangle = (2 * self.omega_inertial - self.dangle_max - self.dangle_min) / (self.dangle_max - self.dangle_min) * self.staticGain
        norm_f = (2 * self.force - self.fmax - self.fmin) / (self.fmax - self.fmin) * self.staticGain

        norm_state = np.concatenate((norm_error, norm_pos, norm_vel, norm_angle, norm_dangle, norm_f)).tolist()

        return list(norm_state)

    def saveModel2XML(self, filename='uav_hover.xml', filepath='../config/'):
        rootMsg = {
            'name': 'uav_hover',
            'author': 'Yefeng YANG',
            'date': '2023.02.27',
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
            'm': self.m,
            'g': self.g,
            'Jxx': self.Jxx,
            'Jyy': self.Jyy,
            'Jzz': self.Jzz,
            'd': self.d,
            'CT': self.CT,
            'CM': self.CM,
            'J0': self.J0,
            'pos': self.pos,
            'vel': self.vel,
            'angle': self.angle,
            'omega_iner': self.omega_inertial,
            'omega_body': self.omega_body,
            'pos_max': self.pos_max,
            'pos_min': self.pos_min,
            'vel_max': self.vel_max,
            'vel_min': self.vel_min,
            'angle_max': self.angle_max,
            'angle_min': self.angle_min,
            'dangle_max': self.dangle_max,
            'dangle_min': self.dangle_min,
            'force': self.force,
            'fmax': self.fmax,
            'fmin': self.fmin,
            'w_rotor': self.w_rotor,
            'dt': self.dt,
            'time': self.time,
            'tmax': self.tmax
        }
        imageMsg = {
            'figsize': [9, 9],
            'xbound': self.uav_vis.xbound,
            'ybound': self.uav_vis.ybound,
            'zbound': self.uav_vis.zbound,
            'length_per_n': self.uav_vis.length_per_n
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

    def saveData(self, is2file=False, filename='uav.csv', filepath=''):
        if is2file:
            data = pd.DataFrame({
                'x:': self.save_pos[:, 0],
                'y': self.save_pos[:, 1],
                'z': self.save_pos[:, 2],
                'vx': self.save_vel[:, 0],
                'vy': self.save_vel[:, 1],
                'xz': self.save_vel[:, 2],
                'phi': self.save_angle[:, 0],
                'theta': self.save_angle[:, 1],
                'psi': self.save_angle[:, 2],
                'dphi': self.save_omega_inertial[:, 0],
                'dtheta': self.save_omega_inertial[:, 1],
                'dpsi': self.save_omega_inertial[:, 2],
                'p': self.save_omega_body[:, 0],
                'q': self.save_omega_body[:, 1],
                'r': self.save_omega_body[:, 2],
                'f1': self.save_f[:, 0],
                'f2': self.save_f[:, 1],
                'f3': self.save_f[:, 2],
                'f4': self.save_f[:, 3],
                'time': self.save_t
            })
            data.to_csv(filepath + filename, index=False, sep=',')
        else:
            self.save_pos = np.vstack((self.save_pos, self.pos))
            self.save_vel = np.vstack((self.save_vel, self.vel))
            self.save_angle = np.vstack((self.save_angle, self.angle))
            self.save_omega_inertial = np.vstack((self.save_omega_inertial, self.omega_inertial))
            self.save_omega_body = np.vstack((self.save_omega_body, self.omega_body))
            self.save_f = np.vstack((self.save_f, self.force))
            self.save_t = np.array((self.save_t, self.time))

    def get_reward(self, param):
        pass

    def step_update(self, action):
        pass

    def reset(self):
        pass

    def reset_random(self):
        pass
