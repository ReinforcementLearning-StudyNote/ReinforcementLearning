import numpy as np
from common.common_func import *
from environment.envs import *
import math

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d


class UAV:
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
                 # domega0_body=None
                 ):
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
        self.xmax, self.ymax, self.zmax = 10, 10, 0
        self.xmin, self.ymin, self.zmin = -10, -10, -5
        self.vxmax, self.vymax, self.vzmax = 10, 10, 10
        self.vxmin, self.vymin, self.vzmin = -10, -10, -10

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

        '''physical parameters'''

        '''visualization_opencv'''

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

        self.save_f1 = [self.f1]
        self.save_f2 = [self.f2]
        self.save_f3 = [self.f3]
        self.save_f4 = [self.f4]

        self.save_t = [self.time]
        '''datasave'''

    def is_out(self):
        """
        :return:
        """
        '''简化处理，只判断中心的大圆有没有出界就好'''
        is_omg_out = (self.dphi > self.dphimax + 1e-1) or (self.dphi < self.dphimin - 1e-1) or \
                     (self.dtheta > self.dthetamax + 1e-1) or (self.dtheta < self.dthetamin - 1e-1) or \
                     (self.dpsi > self.dpsimax + 1e-1) or (self.dpsi < self.dpsimin - 1e-1)
        if is_omg_out:
            print('Omega out...')
            return True

        is_ang_out = (self.phi > self.phimax + 1e-2) or (self.phi < self.phimin - 1e-2) or \
                     (self.theta > self.thetamax + 1e-2) or (self.theta < self.thetamin - 1e-2) or \
                     (self.psi > self.psimax + 1e-2) or (self.psi < self.psimin - 1e-2)
        if is_ang_out:
            print('Attitude out...')
            return True

        is_vel_out = (self.vx > self.vxmax + 1e-1) or (self.vx < self.vxmin - 1e-1) or \
                     (self.vy > self.vymax + 1e-1) or (self.vy < self.vymin - 1e-1) or \
                     (self.vz > self.vzmax + 1e-1) or (self.vz < self.vzmin - 1e-1)
        if is_vel_out:
            print('Velocity out...')
            return True

        is_pos_out = (self.x > self.xmax + 1e-2) or (self.x < self.xmin - 1e-2) or \
                     (self.y > self.ymax + 1e-2) or (self.y < self.ymin - 1e-2) or \
                     (self.z > self.zmax + 1e-2) or (self.z < self.zmin - 1e-2)
        if is_pos_out:
            print('Position out...')
            return True

        return False

    def is_episode_Terminal(self):
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

    def f2omega(self):
        self.w1 = math.sqrt(self.f1 / self.CT)
        self.w2 = math.sqrt(self.f2 / self.CT)
        self.w3 = math.sqrt(self.f3 / self.CT)
        self.w4 = math.sqrt(self.f4 / self.CT)

    def f(self, xx: np.ndarray):
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
        [_x, _y, _z] = xx[0:3]
        [_vx, _vy, _vz] = xx[3:6]
        [_phi, _theta, _psi] = xx[6:9]
        [_p, _q, _r] = xx[9:12]
        self.f2omega()  # 根据力，计算出四个电机的转速
        f = self.f1 + self.f2 + self.f3 + self.f4  # 总推力
        square_w = np.array([self.w1 ** 2, self.w2 ** 2, self.w3 ** 2, self.w4 ** 2])
        '''1. 无人机绕机体系旋转的角速度p q r 的微分方程'''
        dp = (self.CT * self.d / math.sqrt(2) * np.dot(square_w, [1, -1, -1, 1]) +
              (self.Jyy - self.Jzz) * _q * _r +
              self.J0 * _q * (self.w1 - self.w2 + self.w3 - self.w4)) / self.Jxx
        dq = (self.CT * self.d / math.sqrt(2) * np.dot(square_w, [1, 1, -1, -1]) +
              (self.Jzz - self.Jxx) * _p * _r +
              self.J0 * _p * (-self.w1 + self.w2 - self.w3 + self.w4)) / self.Jyy
        dr = (self.CM * np.dot(square_w, [1, -1, 1, -1]) + (self.Jyy - self.Jxx) * _p * _q) / self.Jzz
        '''1. 无人机绕机体系旋转的角速度 p q r 的微分方程'''

        '''2. 无人机在惯性系下的姿态角 phi theta psi 的微分方程'''
        _R_pqr2diner = np.array([[1, math.tan(_theta) * math.sin(_phi), math.tan(_theta) * math.cos(_phi)],
                                 [0, math.cos(_phi), -math.sin(_phi)],
                                 [0, math.sin(_phi) / math.cos(_theta), math.cos(_phi) / math.cos(_theta)]])
        [dphi, dtheta, dpsi] = np.dot(_R_pqr2diner, [_p, _q, _r]).tolist()
        '''2. 无人机在惯性系下的姿态角 phi theta psi 的微分方程'''

        '''3. 无人机在惯性系下的位置 x y z 和速度 vx vy vz 的微分方程'''
        [dx, dy, dz] = [_vx, _vy, _vz]
        dvx = -f / self.m * (math.cos(_psi) * math.sin(_theta) * math.cos(_phi) + math.sin(_psi) * math.sin(_phi))
        dvy = -f / self.m * (math.sin(_psi) * math.sin(_theta) * math.cos(_phi) - math.cos(_psi) * math.sin(_phi))
        dvz = self.g - f / self.m * math.cos(_phi) * math.cos(_theta)
        '''3. 无人机在惯性系下的位置 x y z 和速度 vx vy vz 的微分方程'''

        return np.array([dx, dy, dz, dvx, dvy, dvz, dphi, dtheta, dpsi, dp, dq, dr])

    def rk44(self, action: list):
        [self.f1, self.f2, self.f3, self.f4] = action
        h = self.dt / 1  # RK-44 解算步长
        tt = self.time + self.dt
        while self.time < tt:
            xx_old = np.array([self.x, self.y, self.z,
                               self.vx, self.vy, self.vz,
                               self.phi, self.theta, self.psi,
                               self.p, self.q, self.r])
            K1 = h * self.f(xx_old)
            K2 = h * self.f(xx_old + K1 / 2)
            K3 = h * self.f(xx_old + K2 / 2)
            K4 = h * self.f(xx_old + K3)
            xx_new = xx_old + (K1 + 2 * K2 + 2 * K3 + K4) / 6
            xx_new = xx_new.tolist()
            [self.x, self.y, self.z, self.vx, self.vy, self.vz, self.phi, self.theta, self.psi, self.p, self.q, self.r] = xx_new.copy()
            self.time += h
        R_pqr2diner = np.array([[1, math.tan(self.theta) * math.sin(self.phi), math.tan(self.theta) * math.cos(self.phi)],
                                [0, math.cos(self.phi), -math.sin(self.phi)],
                                [0, math.sin(self.phi) / math.cos(self.theta), math.cos(self.phi) / math.cos(self.theta)]])
        [self.dphi, self.dtheta, self.dpsi] = np.dot(R_pqr2diner, [self.p, self.q, self.r]).tolist()

    def saveData(self, is2file=False, filename='UGV_Forward_Continuous.csv', filepath=''):
        if is2file:
            data = pd.DataFrame({
                'x:': self.save_x,
                'y': self.save_y,
                'z': self.save_z,
                'vx': self.save_vx,
                'vy': self.save_vy,
                'xz': self.save_vz,
                'phi': self.save_phi,
                'theta': self.save_theta,
                'psi': self.save_psi,
                'dphi': self.save_dphi,
                'dtheta': self.save_dtheta,
                'dpsi': self.save_dpsi,
                'p': self.save_p,
                'q': self.save_q,
                'r': self.save_r,
                'f1': self.save_f1,
                'f2': self.save_f2,
                'f3': self.save_f3,
                'f4': self.save_f4,
                'time': self.save_t
            })
            data.to_csv(filepath + filename, index=False, sep=',')
        else:
            self.save_x.append(self.x)
            self.save_y.append(self.y)
            self.save_z.append(self.z)

            self.save_vx.append(self.vx)
            self.save_vy.append(self.vy)
            self.save_vz.append(self.vz)

            self.save_phi.append(self.phi)
            self.save_theta.append(self.theta)
            self.save_psi.append(self.psi)

            self.save_dphi.append(self.dphi)
            self.save_dtheta.append(self.dtheta)
            self.save_dpsi.append(self.dpsi)

            self.save_p.append(self.p)
            self.save_q.append(self.q)
            self.save_r.append(self.r)

            self.save_f1.append(self.f1)
            self.save_f2.append(self.f2)
            self.save_f3.append(self.f3)
            self.save_f4.append(self.f4)

            self.save_t.append(self.time)

    def show_uav_linear_state(self, with_time=True):
        if with_time:
            s = 'Time: %.3fs  Pos: [%.3f, %.3f, %.3f]m  Vel: [%.3f, %.3f, %.3f]m/s' % (self.time, self.x, self.y, self.z, self.vx, self.vy, self.vz)
        else:
            s = 'Pos: [%.3f, %.3f, %.3f]m  Vel: [%.3f, %.3f, %.3f]m/s' % (self.x, self.y, self.z, self.vx, self.vy, self.vz)
        print(s)

    def show_uav_angular_state(self, with_time=True):
        if with_time:
            s = 'Time: %.3fs  Ang: [%.3f, %.3f, %.3f]rad  dAng: [%.3f, %.3f, %.3f]rad/s  Omg: [%.3f, %.3f, %.3f]rad/s' %\
                (self.time, self.phi, self.theta, self.psi, self.dphi, self.dtheta, self.dpsi, self.p, self.q, self.r)
        else:
            s = 'Ang: [%.3f, %.3f, %.3f]rad  dAng: [%.3f, %.3f, %.3f]rad/s  Omg: [%.3f, %.3f, %.3f]rad/s' % \
                (self.phi, self.theta, self.psi, self.dphi, self.dtheta, self.dpsi, self.p, self.q, self.r)
        print(s)

    def reset(self, pos0, vel0, angle0, omega0_inertial, omega0_body):
        [self.x, self.y, self.z] = pos0
        [self.vx, self.vy, self.vz] = vel0
        [self.phi, self.theta, self.psi] = angle0
        [self.dphi, self.dtheta, self.dpsi] = omega0_inertial
        [self.p, self.q, self.r] = omega0_body
        self.time = 0
        self.control_state = np.array([self.x, self.y, self.z,
                                       self.vx, self.vy, self.vz,
                                       self.phi, self.theta, self.psi,
                                       self.dphi, self.dtheta, self.dpsi])
        self.f1 = 0
        self.f2 = 0
        self.f3 = 0
        self.f4 = 0
        self.w1 = math.sqrt(self.f1 / self.CT)
        self.w2 = math.sqrt(self.f2 / self.CT)
        self.w3 = math.sqrt(self.f3 / self.CT)
        self.w4 = math.sqrt(self.f4 / self.CT)

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

        self.save_f1 = [self.f1]
        self.save_f2 = [self.f2]
        self.save_f3 = [self.f3]
        self.save_f4 = [self.f4]

        self.save_t = [self.time]
        '''datasave'''


