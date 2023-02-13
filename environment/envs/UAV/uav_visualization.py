import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib.pyplot import MultipleLocator
from matplotlib.axes import Axes
import numpy as np
import math


class UAV_Visualization:
    def __init__(self, xbound: list, ybound: list, zbound: list, origin: list):
        self.fig = plt.figure(figsize=(12, 9))
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = [-zbound[1], -zbound[0]]
        self.origin = origin
        self.ax = axes3d.Axes3D(self.fig)
        self.ax.set_aspect('auto')      # 只能auto
        self.ax.set_xlim3d(self.xbound)
        self.ax.set_ylim3d(self.ybound)
        self.ax.set_zlim3d(self.zbound)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.xaxis.set_major_locator(MultipleLocator(1))
        self.ax.yaxis.set_major_locator(MultipleLocator(1))
        self.ax.set_title('QuadrotorFly Simulation', fontsize='13')

        self.trajectory = [[], [], []]
        self.color = ['r', 'b', 'g', 'k', 'm', 'y', 'k']

        '''UAV相关部件'''
        self.origin_point, = self.ax.plot([], [], [], marker='o', color='black', markersize=6, antialiased=False)       # 画原点
        self.label = self.ax.text(0., 0., 0., 'quad', fontsize='11')                                                    # 写名字
        self.center, = self.ax.plot([], [], [], marker='o', color='blue', markersize=15, antialiased=False)  # 中心
        self.bar1, = self.ax.plot([], [], [], color='red', linewidth=4, antialiased=False)                        # 机臂1
        self.bar2, = self.ax.plot([], [], [], color='red', linewidth=4, antialiased=False)                        # 机臂2
        self.bar3, = self.ax.plot([], [], [], color='black', linewidth=4, antialiased=False)                       # 机臂3
        self.bar4, = self.ax.plot([], [], [], color='black', linewidth=4, antialiased=False)                       # 机臂4
        self.head, = self.ax.plot([], [], [], marker='o', color='green', markersize=6, antialiased=False)               # 机头点
        self.head_bar, = self.ax.plot([], [], [], color='green', markersize=6, antialiased=False)                       # 朝向臂
        self.quadGui = {
            'origin_point': self.origin_point,
            'label': self.label,
            'center': self.center,
            'bar1': self.bar1,
            'bar2': self.bar2,
            'bar3': self.bar3,
            'bar4': self.bar4,
            'head': self.head,
            'head_bar': self.head_bar
        }
        '''UAV相关部件'''

        self.figure = self.ax.plot([], [])
        self.draw_grid()

    def draw_grid(self):
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
        # plt.show()
        # plt.pause(0.0000000001)

    def rotate_matrix(self, attitude: np.ndarray):
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

    def render(self, position: np.ndarray, attitude: np.ndarray, d: float):
        """
        :param position:        无人机在惯性系下的位置
        :param attitude:        无人机在惯性系下的姿态
        :param d:               机臂长
        :return:
        """
        R_b_i = self.rotate_matrix(attitude)
        d0 = d / math.sqrt(2)
        bar1 = np.dot(R_b_i, [d0, -d0, 0]) + position
        bar2 = np.dot(R_b_i, [d0, d0, 0]) + position
        bar3 = np.dot(R_b_i, [-d0, d0, 0]) + position
        bar4 = np.dot(R_b_i, [-d0, -d0, 0]) + position
        head = np.dot(R_b_i, [2 * d0, 0, 0]) + position
        # print(R_b_i, position, attitude)

        '''初始位置'''
        self.quadGui['origin_point'].set_data([self.origin[0], self.origin[0]], [self.origin[1], self.origin[1]])
        self.quadGui['origin_point'].set_3d_properties([self.origin[2], self.origin[2]])
        '''初始位置'''

        '''无人机中心位置'''
        self.quadGui['center'].set_data([position[0], position[0]], [position[1], position[1]])
        self.quadGui['center'].set_3d_properties([-position[2], -position[2]])
        '''无人机中心位置'''

        '''四个机臂位置'''
        self.quadGui['bar1'].set_data([bar1[0], position[0]], [bar1[1], position[1]])
        self.quadGui['bar1'].set_3d_properties([-bar1[2], -position[2]])
        self.quadGui['bar2'].set_data([bar2[0], position[0]], [bar2[1], position[1]])
        self.quadGui['bar2'].set_3d_properties([-bar2[2], -position[2]])
        self.quadGui['bar3'].set_data([bar3[0], position[0]], [bar3[1], position[1]])
        self.quadGui['bar3'].set_3d_properties([-bar3[2], -position[2]])
        self.quadGui['bar4'].set_data([bar4[0], position[0]], [bar4[1], position[1]])
        self.quadGui['bar4'].set_3d_properties([-bar4[2], -position[2]])
        '''四个机臂位置'''

        '''机头'''
        self.quadGui['head_bar'].set_data([head[0], position[0]], [head[1], position[1]])
        self.quadGui['head_bar'].set_3d_properties([-head[2], -position[2]])
        self.quadGui['head'].set_data([head[0], head[0]], [head[1], head[1]])
        self.quadGui['head'].set_3d_properties([-head[2], -head[2]])
        '''机头'''
