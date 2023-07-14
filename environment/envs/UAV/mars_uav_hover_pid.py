import math

import numpy as np

from common.common_func import *
from mars_uav_hover import MARS_UAV_Hover
import matplotlib.pyplot as plt
from environment.envs.PIDControl.pid import PID

if __name__ == '__main__':
    env = MARS_UAV_Hover(target_pos=[5, 5, 0], save_cfg=False)
    env.uav_vis.arm_scale = 2  # 显示放大的尺度，自己设置即可

    pid_x = PID(kp=0.25, ki=0., kd=80)  # controller of x
    pid_y = PID(kp=0.25, ki=0., kd=80)  # controller of y
    pid_z = PID(kp=0.6, ki=0., kd=150)  # controller of z

    pid_phi = PID(kp=0.05, ki=0., kd=5)  # controller of roll along X in world
    pid_theta = PID(kp=0.05, ki=0., kd=5)  # controller of pitch along Y in world
    pid_psi = PID(kp=0.01, ki=0., kd=1)  # controller of yaw along Z in world

    num = 0

    plt.ion()
    while num < 5:
        env.reset_random()
        # env.reset()
        # env.reset_target_random()
        while not env.is_terminal:
            env.show_dynamic_image(per_show=10)
            plt.pause(0.00000001)
            '''control'''
            '''位置PID输出'''
            pid_x.set_e(env.error_pos[0])
            pid_y.set_e(env.error_pos[1])
            pid_z.set_e(env.error_pos[2])
            ux, uy, uz = pid_x.out(), pid_y.out(), pid_z.out()
            '''位置PID输出'''

            '''计算期望姿态'''
            U1 = env.m * math.sqrt(ux ** 2 + uy ** 2 + (uz + env.g) ** 2)
            psi_ref = deg2rad(1)
            phi_ref = np.arcsin(env.m * (ux * np.sin(psi_ref) - uy * np.cos(psi_ref)) / U1)
            theta_ref = np.arcsin(env.m * (ux * np.cos(psi_ref) + uy * np.sin(psi_ref)) / (U1 * np.cos(phi_ref)))
            '''计算期望姿态'''

            '''姿态PID输出'''
            e_phi, e_theta, e_psi = phi_ref - env.angle[0], theta_ref - env.angle[1], psi_ref - env.angle[2]
            pid_phi.set_e(e_phi)
            pid_theta.set_e(e_theta)
            pid_psi.set_e(e_psi)
            U2, U3, U4 = pid_phi.out(), pid_theta.out(), pid_psi.out()
            '''姿态PID输出'''

            '''control'''
            # f = [random.uniform(env.fmin, env.fmax) for _ in range(4)]
            env.step_update(action=[U1, U2, U3, U4])
            # print('Pos_e: {}'.format(env.error_pos))
        num += 1
    plt.ioff()
