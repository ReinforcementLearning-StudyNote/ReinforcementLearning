import numpy as np

from common.common_func import *
from environment.envs import *
import math


class UAV(rl_base):
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
                 # domega0_body=None,
                 save_cfg: bool = True):
        super(UAV).__init__()
        # if domega0_body is None:
        #     domega0_body = [0, 0, 0]
        if omega0_body is None:
            omega0_body = [0, 0, 0]
        if omega0_inertial is None:
            omega0_inertial = [0, 0, 0]
        if angle0 is None:
            angle0 = [0, 0, 0]
        if vel0 is None:
            vel0 = [0, 0, 0]
        if pos0 is None:
            pos0 = [0, 0, 0]
        '''physical parameters'''
        self.m = m  # 无人机质量
        self.g = g  # 重力加速度
        self.Jxx = Jxx  # X方向转动惯量
        self.Jyy = Jyy  # Y方向转动惯量
        self.Jzz = Jzz  # Z方向转动惯量
        self.d = d  # 机臂长度 'X'构型
        self.CT = CT  # 螺旋桨升力系数
        self.CM = CM  # 螺旋桨力矩系数
        self.J0 = J0  # 电机和螺旋桨的转动惯量

        [self.x, self.y, self.z] = pos0  # 无人机在世界坐标系下的初始位置
        [self.vx, self.vy, self.vz] = vel0  # 无人机在世界坐标系下的初始速度
        [self.phi, self.theta, self.psi] = angle0  # 无人机在世界坐标系下的初始角度
        [self.dphi, self.dtheta, self.dpsi] = omega0_inertial  # 无人机在世界坐标系下的初始角速度
        [self.p, self.q, self.r] = omega0_body  # 无人机在机体坐标系下的初始角速度
        # [self.dp, self.dq, self.dr] = domega0_body              # 无人机在机体坐标系下的初始角加速度

        self.dt = 0.02  # 控制频率，50Hz
        self.time = 0.  # 当前时间
        self.tmax = 20  # 每回合最大时间
        self.staticGain = 4

        self.control_state = np.array([self.x, self.y, self.z,
                                       self.vx, self.vy, self.vz,
                                       self.phi, self.theta, self.psi,
                                       self.dphi, self.dtheta, self.dpsi])  # 控制系统的状态，不是强化学习的状态

        'state limitation'
        self.xmax, self.ymax, self.zmax = 10, 10, 5
        self.xmin, self.ymin, self.zmin = -10, -10, 0
        self.vxmax, self.vymax, self.vzmax = 6, 6, 3
        self.vxmin, self.vymin, self.vzmin = -6, -6, -3

        self.phimax, self.thetamax, self.psimax = deg2rad(80), deg2rad(80), deg2rad(180)
        self.phimin, self.thetamin, self.psimin = -deg2rad(80), -deg2rad(80), -deg2rad(180)

        self.dphimax, self.dthetamax, self.dpsimax = deg2rad(180), deg2rad(180), deg2rad(100)
        self.dphimin, self.dthetamin, self.dpsimin = -deg2rad(180), -deg2rad(180), -deg2rad(100)
        'state limitation'

        'control'
        '''
        这里直接使用力作为输入，范围小，方便学习。
        解算微分方程时，直接利用升力系数将力换算成电机角速度即可
        '''
        self.f1 = 0
        self.f2 = 0
        self.f3 = 0
        self.f4 = 0
        self.fmin = 0
        self.fmax = 10
        self.w1 = math.sqrt(self.f1 / self.CT)
        self.w2 = math.sqrt(self.f2 / self.CT)
        self.w3 = math.sqrt(self.f3 / self.CT)
        self.w4 = math.sqrt(self.f4 / self.CT)
        'control'

        '''rl_base'''
        self.state_dim = 22  # [evx evy evz ex ey ez x y z vx vy vz phi theta psi dphi dtheta dpsi f1 f2 f3 f4]
        self.state_num = [math.inf for _ in range(self.state_dim)]  # 连续系统，状态数量无穷
        self.state_step = [None for _ in range(self.state_dim)]  # 连续系统，没有步长
        self.state_space = [None for _ in range(self.state_dim)]  # 连续系统，没有状态空间
        self.state_range = [[-self.staticGain, self.staticGain],  # 归一化的 vx 的误差 范围
                            [-self.staticGain, self.staticGain],  # 归一化的 vy 的误差 范围
                            [-self.staticGain, self.staticGain],  # 归一化的 vz 的误差 范围
                            [-self.staticGain, self.staticGain],  # 归一化的  x 的误差 范围
                            [-self.staticGain, self.staticGain],  # 归一化的  y 的误差 范围
                            [-self.staticGain, self.staticGain],  # 归一化的  z 的误差 范围
                            [-self.staticGain, self.staticGain],  # 归一化的 x 范围
                            [-self.staticGain, self.staticGain],  # 归一化的 y 范围
                            [-self.staticGain, self.staticGain],  # 归一化的 z 范围
                            [-self.staticGain, self.staticGain],  # 归一化的 vx 范围
                            [-self.staticGain, self.staticGain],  # 归一化的 vy 范围
                            [-self.staticGain, self.staticGain],  # 归一化的 vz 范围
                            [-self.staticGain, self.staticGain],  # 归一化的  phi   范围
                            [-self.staticGain, self.staticGain],  # 归一化的 theta  范围
                            [-self.staticGain, self.staticGain],  # 归一化的  psi   范围
                            [-self.staticGain, self.staticGain],  # 归一化的  dphi  范围
                            [-self.staticGain, self.staticGain],  # 归一化的 dtheta 范围
                            [-self.staticGain, self.staticGain],  # 归一化的  dpsi  范围
                            [-self.staticGain, self.staticGain],  # 归一化的 f1 范围
                            [-self.staticGain, self.staticGain],  # 归一化的 f2 范围
                            [-self.staticGain, self.staticGain],  # 归一化的 f3 范围
                            [-self.staticGain, self.staticGain]  # 归一化的 f4 范围
                            ]
        self.isStateContinuous = [True for _ in range(self.state_dim)]
        self.initial_state = self.state_norm2rl_base(vel_d=[0, 0, 0], pos_d=[0, 0, 0])
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

        self.Q = np.identity(6)  # 没有经过调试，使用时需要修改，正定对角阵
        self.R = np.identity(self.action_dim)  # 没有经过调试，使用时需要修改，正定对角阵

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 轨迹跟踪或者控制属于无限时间任务，只不过将最大时间限制在tmax而已
        '''rl_base'''

        '''physical parameters'''

        '''visualization_opencv'''
        self.show_dynamic_image(isWait=False)
        '''visualization_opencv'''

        '''datasave'''
        self.save_x = [self.x]
        self.save_y = [self.y]
        self.save_z = [self.z]

        self.save_vx = [self.vx]
        self.save_vy = [self.vy]
        self.save_vz = [self.vz]

        self.save_phi = [self.phi]
        self.save_theta = [self.theta]
        self.save_psi = [self.psi]

        self.save_dphi = [self.dphi]
        self.save_dtheta = [self.dtheta]
        self.save_dpsi = [self.dpsi]

        self.save_p = [self.p]
        self.save_q = [self.q]
        self.save_r = [self.r]

        self.save_dp = [self.dp]
        self.save_dq = [self.dq]
        self.save_dr = [self.dr]

        self.save_f1 = [self.f1]
        self.save_f2 = [self.f2]
        self.save_f3 = [self.f3]
        self.save_f4 = [self.f4]

        self.save_t = [self.time]
        '''datasave'''

        if save_cfg:
            self.saveModel2XML()

    def state_norm2rl_base(self, vel_d: list, pos_d: list) -> list:
        """
        :return:    归一化的状态
        :tips:      仅仅归一化与RL有关的状态，但是并不会影响这个物理状态本身的数值
        """
        '''[evx evy evz ex ey ez x y z vx vy vz phi theta psi dphi dtheta dpsi f1 f2 f3 f4]'''
        evx = (self.vx - vel_d[0]) / (self.vxmax - self.vxmin) * self.staticGain
        evy = (self.vy - vel_d[1]) / (self.vymax - self.vymin) * self.staticGain
        evz = (self.vz - vel_d[2]) / (self.vzmax - self.vzmin) * self.staticGain

        ex = (self.x - pos_d[0]) / (self.xmax - self.xmin) * self.staticGain
        ey = (self.y - pos_d[1]) / (self.ymax - self.ymin) * self.staticGain
        ez = (self.z - pos_d[2]) / (self.zmax - self.zmin) * self.staticGain

        x = (2 * self.x - self.xmin - self.xmax) / (self.xmax - self.xmin) * self.staticGain
        y = (2 * self.y - self.ymin - self.ymax) / (self.ymax - self.ymin) * self.staticGain
        z = (2 * self.z - self.zmin - self.zmax) / (self.zmax - self.zmin) * self.staticGain

        vx = (2 * self.vx - self.vxmin - self.vxmax) / (self.vxmax - self.vxmin) * self.staticGain
        vy = (2 * self.vy - self.vymin - self.vymax) / (self.vymax - self.vymin) * self.staticGain
        vz = (2 * self.vz - self.vzmin - self.vzmax) / (self.vzmax - self.vzmin) * self.staticGain

        phi = (2 * self.phi - self.phimin - self.phimax) / (self.phimax - self.phimin) * self.staticGain
        theta = (2 * self.theta - self.thetamin - self.thetamax) / (self.thetamax - self.thetamin) * self.staticGain
        psi = (2 * self.psi - self.psimin - self.psimax) / (self.psimax - self.psimin) * self.staticGain

        dphi = (2 * self.dphi - self.dphimin - self.dphimax) / (self.dphimax - self.dphimin) * self.staticGain
        dtheta = (2 * self.dtheta - self.dthetamin - self.dthetamax) / (self.dthetamax - self.dthetamin) * self.staticGain
        dpsi = (2 * self.dpsi - self.dpsimin - self.dpsimax) / (self.dpsimax - self.dpsimin) * self.staticGain

        f1 = (2 * self.f1 - self.fmin - self.fmax) / (self.fmax - self.fmin) * self.staticGain
        f2 = (2 * self.f2 - self.fmin - self.fmax) / (self.fmax - self.fmin) * self.staticGain
        f3 = (2 * self.f3 - self.fmin - self.fmax) / (self.fmax - self.fmin) * self.staticGain
        f4 = (2 * self.f4 - self.fmin - self.fmax) / (self.fmax - self.fmin) * self.staticGain

        return [evx, evy, evz, ex, ey, ez, x, y, z, vx, vy, vz, phi, theta, psi, dphi, dtheta, dpsi, f1, f2, f3, f4]

    def is_out(self):
        """
        :return:
        """
        '''简化处理，只判断中心的大圆有没有出界就好'''
        is_omg_out = (self.dphi > self.dphimax) or (self.dphi < self.dphimin) or \
                     (self.dtheta > self.dthetamax) or (self.dtheta < self.dthetamin) or \
                     (self.dpsi > self.dpsimax) or (self.dpsi < self.dpsimin)
        if is_omg_out:
            print('Omega out...')
            return True

        is_ang_out = (self.phi > self.phimax) or (self.phi < self.phimin) or \
                     (self.theta > self.thetamax) or (self.theta < self.thetamin) or \
                     (self.psi > self.psimax) or (self.psi < self.psimin)
        if is_ang_out:
            print('Attitude out...')
            return True

        is_vel_out = (self.vx > self.vxmax) or (self.vx < self.vxmin) or \
                     (self.vy > self.vymax) or (self.vy < self.vymin) or \
                     (self.vz > self.vzmax) or (self.vz < self.vzmin)
        if is_vel_out:
            print('Velocity out...')
            return True

        is_pos_out = (self.x > self.xmax) or (self.x < self.xmin) or \
                     (self.y > self.ymax) or (self.y < self.ymin) or \
                     (self.z > self.zmax) or (self.z < self.zmin)
        if is_pos_out:
            print('Position out...')
            return True

        return False

    def is_Terminal(self, param=None):
        self.terminal_flag = 0
        if self.time > self.tmax:
            print('Time out...')
            self.terminal_flag = 2
            return True

        if self.is_out():
            print('Out...')
            self.terminal_flag = 1
            return True

        return False

    def get_reward(self, param=None):
        xx = self.control_state[0:6]
        uu = np.array([self.f1, self.f2, self.f3, self.f4])
        r1 = np.matmul(np.matmul(xx, self.Q), xx)
        r2 = np.matmul(np.matmul(uu, self.R), uu)
        # RL 中奖励应该是最大化才对
        # 但是这里 r1 和 r2 实际上是惩罚
        # 因此要最小化惩罚才行，或者最大化 ”负的惩罚“
        self.reward = -(r1 + r2)

    def f2omega(self):
        self.w1 = math.sqrt(self.f1 / self.CT)
        self.w2 = math.sqrt(self.f2 / self.CT)
        self.w3 = math.sqrt(self.f3 / self.CT)
        self.w4 = math.sqrt(self.f4 / self.CT)

    def f(self):
        """
        :func:          微分方程
        :param u:       控制输入
        :return:        状态的导数，也就是微分方程的左端
        """
        '''
        在微分方程里面的状态 X = [x y z vx vy vz phi theta psi p q r] 一共12个
        定义惯性系到机体系的旋转矩阵
        R_i_b1 = np.array([[math.cos(self.psi), math.sin(self.psi), 0],
                           [-math.sin(self.psi), math.cos(self.psi), 0],
                           [0, 0, 1]])  # 从惯性系到b1系，旋转偏航角psi
        R_b1_b2 = np.array([[math.cos(self.theta), 0, -math.sin(self.theta)],
                            [0, 1, 0],
                            [math.sin(self.theta), 0, math.cos(self.theta)]])  # 从b1系到b2系，旋转俯仰角theta
        R_b2_b = np.array([[1, 0, 0],
                           [0, math.cos(self.phi), math.sin(self.phi)],
                           [0, -math.sin(self.phi), math.cos(self.phi)]])  # 从b2系到b系，旋转滚转角phi
        R_i_b = np.matmul(R_b2_b, np.matmul(R_b1_b2, R_i_b1))  # 从惯性系到机体系的转换矩阵
        R_b_i = R_i_b.T  # 从机体系到惯性系的转换矩阵
        e3_i = np.array([0, 0, 1])  # 惯性系下的Z轴基向量
        # dx = v
        # dv = g*e3_i + f/m*
        '''
        self.f2omega()      # 根据力，计算出四个电机的转速
        f = self.f1 + self.f2 + self.f3 + self.f4       # 总推力
        square_w = np.array([self.w1 ** 2, self.w2 ** 2, self.w3 ** 2, self.w4 ** 2])
        '''1. 无人机绕机体系旋转的角速度p q r 的微分方程'''
        dp = (self.CT * self.d / math.sqrt(2) * np.dot(square_w, [1, -1, -1, 1]) +
              (self.Jyy - self.Jzz) * self.q * self.r +
              self.J0 * self.q * (self.w1 - self.w2 + self.w3 - self.w4)) / self.Jxx
        dq = (self.CT * self.d / math.sqrt(2) * np.dot(square_w, [1, 1, -1, -1]) +
              (self.Jzz - self.Jxx) * self.p * self.r +
              self.J0 * self.p * (-self.w1 + self.w2 - self.w3 + self.w4)) / self.Jyy
        dr = (self.CM * np.dot(square_w, [1, -1, 1, -1]) + (self.Jyy - self.Jxx) * self.p * self.q) / self.Jzz
        '''1. 无人机绕机体系旋转的角速度 p q r 的微分方程'''

        '''2. 无人机在惯性系下的姿态角 phi theta psi 的微分方程'''
        R_pqr2diner = np.array([[1, math.tan(self.theta) * math.sin(self.phi), math.tan(self.theta)*math.cos(self.phi)],
                                [0, math.cos(self.phi), -math.sin(self.phi)],
                                [0, math.sin(self.phi)/math.cos(self.theta), math.cos(self.phi)/math.cos(self.theta)]])
        [dphi, dtheta, dpsi] = np.dot(R_pqr2diner, [self.p, self.q, self.r]).tolist()
        '''2. 无人机在惯性系下的姿态角 phi theta psi 的微分方程'''

        '''3. 无人机在惯性系下的位置 x y z 和速度 vx vy vz 的微分方程'''
        [dx, dy, dz] = [self.vx, self.vy, self.vz]
        dvx = -f / self.m * (math.cos(self.psi) * math.sin(self.theta) * math.cos(self.phi) + math.sin(self.psi) * math.sin(self.phi))
        dvy = -f / self.m * (math.sin(self.psi) * math.sin(self.theta) * math.cos(self.phi) - math.cos(self.psi) * math.sin(self.phi))
        dvz = self.g - f / self.m * math.cos(self.phi) * math.cos(self.theta)
        '''3. 无人机在惯性系下的位置 x y z 和速度 vx vy vz 的微分方程'''

        return [dx, dy, dz, dvx, dvy, dvz, dphi, dtheta, dpsi, dp, dq, dr]

    def step_update(self, action: list):
        [self.f1, self.f2, self.f3, self.f4] = action
        self.current_action = action.copy()
        self.current_state = 