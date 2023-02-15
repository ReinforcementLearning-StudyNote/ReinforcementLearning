import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib.pyplot import MultipleLocator
# from matplotlib.axes import Axes
import numpy as np
import math


class UAV_Visualization:
    def __init__(self, xbound: np.ndarray, ybound: np.ndarray, zbound: np.ndarray, origin: np.ndarray):
        """
        :param xbound:      观察系的
        :param ybound:      观察系的
        :param zbound:      观察系的
        :param origin:      观察系的
        """
        self.fig = plt.figure(figsize=(9, 9))
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.o = origin
        self.ax = axes3d.Axes3D(self.fig)
        self.ax.set_aspect('auto')      # 只能auto
        self.is_dynamic_axis = max(np.max(np.fabs(xbound)), np.max(np.fabs(ybound)), np.max(np.fabs(zbound))) >= math.inf
        if not self.is_dynamic_axis:
            self.ax.set_xlim3d(self.xbound)
            self.ax.set_ylim3d(self.ybound)
            self.ax.set_zlim3d(self.zbound)
        else:
            self.ax.set_xlim3d([self.o[0] - 10, self.o[0] + 10])
            self.ax.set_ylim3d([self.o[1] - 10, self.o[1] + 10])
            self.ax.set_zlim3d([self.o[2] - 10, self.o[2] + 10])
        self.ax.xaxis.set_major_locator(MultipleLocator(2))
        self.ax.yaxis.set_major_locator(MultipleLocator(2))
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('QuadrotorFly Simulation', fontsize='13')

        self.traj_data = [[self.o[0]], [self.o[1]], [self.o[2]]]
        self.traj_ref_data = [[], [], []]
        self.traj_count = 600
        self.sim_index = 0
        self.length_per_n = 0.6
        self.color = ['r', 'b', 'g', 'k', 'm', 'y', 'k']

        self.has_ref = False

        '''UAV相关部件'''
        self.origin_point, = self.ax.plot([], [], [], marker='o', color='black', markersize=6, antialiased=False)       # 画原点
        self.label = self.ax.text(0., 0., 0., 'quad', fontsize='11')                                                    # 写名字
        self.center, = self.ax.plot([], [], [], marker='o', color='blue', markersize=10, antialiased=False)  # 中心
        self.bar1, = self.ax.plot([], [], [], color='red', linewidth=4, antialiased=False)                        # 机臂1
        self.bar2, = self.ax.plot([], [], [], color='orange', linewidth=4, antialiased=False)                        # 机臂2
        self.bar3, = self.ax.plot([], [], [], color='blue', linewidth=4, antialiased=False)                       # 机臂3
        self.bar4, = self.ax.plot([], [], [], color='black', linewidth=4, antialiased=False)                       # 机臂4
        self.head, = self.ax.plot([], [], [], marker='o', color='green', markersize=6, antialiased=False)               # 机头点
        self.head_bar, = self.ax.plot([], [], [], color='green', markersize=6, antialiased=False)                       # 朝向臂
        self.traj, = self.ax.plot([], [], [], color='red', linewidth=1.5)
        self.traj_ref, = self.ax.plot([], [], [], color='blue', linewidth=1.5)
        self.grav = self.ax.quiver(self.o[0], self.o[1], self.o[2], 0, 0, -1, length=0.8*9.8*self.length_per_n, color='red')
        self.f1 = self.ax.quiver(self.o[0], self.o[1], self.o[2], 0, 0, 0, length=1, color='red')
        self.f2 = self.ax.quiver(self.o[0], self.o[1], self.o[2], 0, 0, 0, length=1, color='orange')
        self.f3 = self.ax.quiver(self.o[0], self.o[1], self.o[2], 0, 0, 0, length=1, color='blue')
        self.f4 = self.ax.quiver(self.o[0], self.o[1], self.o[2], 0, 0, 0, length=1, color='black')

        self.quadGui = {
            'origin_point': self.origin_point,
            'label': self.label,
            'center': self.center,
            'bar1': self.bar1,
            'bar2': self.bar2,
            'bar3': self.bar3,
            'bar4': self.bar4,
            'head': self.head,
            'head_bar': self.head_bar,
            'traj': self.traj,
            'traj_ref': self.traj_ref,
        }
        cx = np.mean(self.xbound)
        cy = np.mean(self.ybound)
        cz = np.mean(self.zbound)
        self.ax.scatter3D([cx, self.xbound[0], self.xbound[1], cx, cx, cx, cx],
                          [cy, cy, cy, self.ybound[0], self.ybound[1], cy, cy],
                          [cz, cz, cz, cz, cz, self.zbound[0], self.zbound[1]],
                          s=30, c='red')
        '''UAV相关部件'''

        self.figure = self.ax.plot([], [])
        self.draw_bound()

    def draw_bound(self):
        """
        :return:        绘制外框
        """
        for _z in self.zbound:
            for ik in [0, 1]:
                posx = self.xbound
                posy = [self.ybound[ik], self.ybound[ik]]
                posz = [_z, _z]
                self.figure = self.ax.plot(posx, posy, posz, 'y')

                posx = [self.xbound[ik], self.xbound[ik]]
                posy = self.ybound
                posz = [_z, _z]
                self.figure = self.ax.plot(posx, posy, posz, 'y')
        for _x in self.xbound:
            for _y in self.ybound:
                self.figure = self.ax.plot([_x, _x], [_y, _y], self.zbound, 'y')

    @staticmethod
    def rotate_matrix(attitude: np.ndarray):
        [phi, theta, psi] = attitude
        _R_i_b1 = np.array([[math.cos(psi), math.sin(psi), 0],
                            [-math.sin(psi), math.cos(psi), 0],
                            [0, 0, 1]])  # 从惯性系到b1系，旋转偏航角psi
        _R_b1_b2 = np.array([[math.cos(theta), 0, -math.sin(theta)],
                             [0, 1, 0],
                             [math.sin(theta), 0, math.cos(theta)]])  # 从b1系到b2系，旋转俯仰角theta
        _R_b2_b = np.array([[1, 0, 0],
                            [0, math.cos(phi), math.sin(phi)],
                            [0, -math.sin(phi), math.cos(phi)]])  # 从b2系到b系，旋转滚转角phi
        _R_i_b = np.matmul(_R_b2_b, np.matmul(_R_b1_b2, _R_i_b1))  # 从惯性系到机体系的转换矩阵
        _R_b_i = _R_i_b.T  # 从机体系到惯性系的转换矩阵
        return _R_b_i

    def render(self,
               p: np.ndarray,
               ref_p: np.ndarray,
               a: np.ndarray,
               f: np.ndarray,
               d: float,
               win: int):
        """
        :param p:   无人机在 观察系 下的位置
        :param a:   无人机在 观察系 下的姿态
        :param f:   无人机螺旋桨的升力
        :param d:   无人机机臂长度
        :param win: 时间窗口长度
        :return:
        """
        '''轨迹数据存储'''
        if self.sim_index < self.traj_count:
            self.traj_data[0].append(p[0])
            self.traj_data[1].append(p[1])
            self.traj_data[2].append(p[2])
            if self.has_ref:
                self.traj_ref_data[0].append(ref_p[0])
                self.traj_ref_data[1].append(ref_p[1])
                self.traj_ref_data[2].append(ref_p[2])
        else:
            self.traj_data[0].pop(0)
            self.traj_data[1].pop(0)
            self.traj_data[2].pop(0)
            if self.has_ref:
                self.traj_ref_data[0].pop(0)
                self.traj_ref_data[1].pop(0)
                self.traj_ref_data[2].pop(0)
            self.traj_data[0].append(p[0])
            self.traj_data[1].append(p[1])
            self.traj_data[2].append(p[2])
            if self.has_ref:
                self.traj_ref_data[0].append(ref_p[0])
                self.traj_ref_data[1].append(ref_p[1])
                self.traj_ref_data[2].append(ref_p[2])
        '''轨迹数据存储'''

        if self.sim_index % win == 0:
            R_b_i = self.rotate_matrix(a)
            d0 = d / math.sqrt(2)
            bar1 = np.dot(R_b_i, [d0, d0, 0]) + p
            bar2 = np.dot(R_b_i, [d0, -d0, 0]) + p
            bar3 = np.dot(R_b_i, [-d0, -d0, 0]) + p
            bar4 = np.dot(R_b_i, [-d0, +d0, 0]) + p
            head = np.dot(R_b_i, [2 * d0, 0, 0]) + p
            # f1_tail = np.dot(R_b_i, [d0, d0, f[0] * self.length_per_n]) + p
            # f2_tail = np.dot(R_b_i, [d0, -d0, f[1] * self.length_per_n]) + p
            # f3_tail = np.dot(R_b_i, [-d0, -d0, f[2] * self.length_per_n]) + p
            # f4_tail = np.dot(R_b_i, [-d0, +d0, f[3] * self.length_per_n]) + p
            dir_f1 = np.dot(R_b_i, [0, 0, 1])
            dir_f2 = np.dot(R_b_i, [0, 0, 1])
            dir_f3 = np.dot(R_b_i, [0, 0, 1])
            dir_f4 = np.dot(R_b_i, [0, 0, 1])

            if self.is_dynamic_axis:
                self.ax.set_xlim3d([p[0] - 10, p[0] + 10])
                self.ax.set_ylim3d([p[1] - 10, p[1] + 10])
                self.ax.set_zlim3d([p[2] - 10, p[2] + 10])

            '''初始位置'''
            self.quadGui['origin_point'].set_data([self.o[0], self.o[0]], [self.o[1], self.o[1]])
            self.quadGui['origin_point'].set_3d_properties([self.o[2], self.o[2]])
            '''初始位置'''

            '''无人机中心位置'''
            self.quadGui['center'].set_data([p[0], p[0]], [p[1], p[1]])
            self.quadGui['center'].set_3d_properties([p[2], p[2]])
            '''无人机中心位置'''

            '''四个机臂位置'''
            self.quadGui['bar1'].set_data([bar1[0], p[0]], [bar1[1], p[1]])
            self.quadGui['bar1'].set_3d_properties([bar1[2], p[2]])
            self.quadGui['bar2'].set_data([bar2[0], p[0]], [bar2[1], p[1]])
            self.quadGui['bar2'].set_3d_properties([bar2[2], p[2]])
            self.quadGui['bar3'].set_data([bar3[0], p[0]], [bar3[1], p[1]])
            self.quadGui['bar3'].set_3d_properties([bar3[2], p[2]])
            self.quadGui['bar4'].set_data([bar4[0], p[0]], [bar4[1], p[1]])
            self.quadGui['bar4'].set_3d_properties([bar4[2], p[2]])
            '''四个机臂位置'''

            '''机头'''
            self.quadGui['head_bar'].set_data([head[0], p[0]], [head[1], p[1]])
            self.quadGui['head_bar'].set_3d_properties([head[2], p[2]])
            self.quadGui['head'].set_data([head[0], head[0]], [head[1], head[1]])
            self.quadGui['head'].set_3d_properties([head[2], head[2]])
            '''机头'''

            '''轨迹'''
            self.quadGui['traj'].set_data(self.traj_data[0], self.traj_data[1])
            self.quadGui['traj'].set_3d_properties(self.traj_data[2])
            '''轨迹'''

            '''参考轨迹 (如果有)'''
            if self.has_ref:
                self.quadGui['traj_ref'].set_data(self.traj_ref_data[0], self.traj_ref_data[1])
                self.quadGui['traj_ref'].set_3d_properties(self.traj_ref_data[2])
            '''参考轨迹 (如果有)'''

            '''重力加速度'''
            self.grav.remove()
            self.grav = self.ax.quiver(p[0], p[1], p[2], 0, 0, -1, length=0.8*9.8*self.length_per_n, color='black')
            '''重力加速度'''

            '''升力'''
            self.f1.remove()
            self.f2.remove()
            self.f3.remove()
            self.f4.remove()
            self.f1 = self.ax.quiver(bar1[0], bar1[1], bar1[2],
                                     dir_f1[0], dir_f1[1], dir_f1[2],
                                     length=f[0]*self.length_per_n, color='red')
            self.f2 = self.ax.quiver(bar2[0], bar2[1], bar2[2],
                                     dir_f2[0], dir_f2[1], dir_f2[2],
                                     length=f[1]*self.length_per_n, color='orange')
            self.f3 = self.ax.quiver(bar3[0], bar3[1], bar3[2],
                                     dir_f3[0], dir_f3[1], dir_f3[2],
                                     length=f[2]*self.length_per_n, color='blue')
            self.f4 = self.ax.quiver(bar4[0], bar4[1], bar4[2],
                                     dir_f4[0], dir_f4[1], dir_f4[2],
                                     length=f[3]*self.length_per_n, color='black')
            '''升力'''

        self.sim_index += 1
