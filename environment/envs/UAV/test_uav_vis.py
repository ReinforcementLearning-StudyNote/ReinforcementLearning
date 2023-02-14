from uav import UAV
from uav_visualization import UAV_Visualization
from common.common_func import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    pos0 = [0, 0, 0]                               # 给的都是惯性系 东北天
    angle0 = [deg2rad(0), deg2rad(0), deg2rad(0)]    # 给的都是惯性系 东北天

    quad = UAV(pos0=pos0, angle0=angle0)

    xbound = np.array([quad.xmin, quad.xmax])
    ybound = np.array([quad.ymin, quad.ymax])
    zbound = np.array([quad.zmin, quad.zmax])
    origin = np.array([quad.x, quad.y, quad.z])

    quad_vis = UAV_Visualization(xbound=xbound, ybound=ybound, zbound=zbound, origin=origin)   # 初始化显示界面
    index = 0
    plt.ion()
    while not quad.is_episode_Terminal():
        position = np.array([quad.x, quad.y, quad.z])
        attitude = np.array([quad.phi, quad.theta, quad.psi])
        d = 10 * quad.d
        if index % 10 == 0:
            quad_vis.render(position, attitude, d)
            quad.show_uav_linear_state(with_time=True)
            quad.show_uav_angular_state(with_time=True)
            plt.show()
            plt.pause(0.0000000001)
            
        f1 = [9.8 / 5 - 0.02, 9.8 / 5 - 0.02, 9.8 / 5 + 0.02, 9.8 / 5 + 0.02]   # 转
        f2 = [9.8 / 5, 9.8 / 5, 9.8 / 5, 9.8 / 5]                           # 平衡
        f3 = [0, 0, 0, 0]                                                   # 零
        f4 = [9.8 / 5 - 0.1, 9.8 / 5 - 0.1, 9.8 / 5 - 0.1, 9.8 / 5 - 0.1]   # 下降
        f5 = [9.8 / 5 + 0.1, 9.8 / 5 + 0.1, 9.8 / 5 + 0.1, 9.8 / 5 + 0.1]   # 上升
        eq_f = quad.m * quad.g / math.cos(math.fabs(quad.phi)) / math.cos(math.fabs(quad.theta)) / 4
        f6 = [eq_f for _ in range(4)]
        action = f1  # 给无人机四个螺旋桨的力
        quad.rk44(action=action)
        index += 1
    plt.ioff()
    plt.show()
