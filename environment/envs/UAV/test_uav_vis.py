from uav import UAV
from uav_visualization import UAV_Visualization
import numpy as np
from common.common_func import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    quad = UAV(pos0=[0, 0, -5], angle0=[deg2rad(0), deg2rad(0), deg2rad(0)])
    xbound = [quad.xmin, quad.xmax]
    ybound = [quad.ymin, quad.ymax]
    zbound = [quad.zmin, quad.zmax]

    quad_vis = UAV_Visualization(xbound=xbound, ybound=ybound, zbound=zbound, origin=[quad.x, quad.y, quad.z])   # 初始化显示界面
    index = 0
    plt.ion()
    while not quad.is_episode_Terminal():
        position = np.array([quad.x, quad.y, quad.z])
        attitude = np.array([quad.phi, quad.theta, quad.psi])
        d = 10 * quad.d
        if index % 10 ==0:
            quad_vis.render(position, attitude, d)
            quad.show_uav_linear_state(with_time=True)
            plt.show()
            plt.pause(0.0000000001)
        action = [1.8, 1.8, 1.8, 1.8]       # 给无人机四个螺旋桨的力
        quad.rk44(action=action)
    plt.ioff()
