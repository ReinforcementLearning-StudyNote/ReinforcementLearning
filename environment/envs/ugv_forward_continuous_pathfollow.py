import cv2 as cv
import numpy as np

from common.common import *
from environment.envs import *
from environment.envs.pathplanning.bezier import Bezier
from environment.envs.ugv_forward_continuous import UGV_Forward_Continuous as UGV


class UGV_Forward_Continuous_Path_Follow(UGV):
    def __init__(self,
                 initPhi: float,
                 save_cfg: bool,
                 x_size: float,
                 y_size: float,
                 start: list,
                 terminal: list):
        """
        :param initPhi:         initial heading angle
        :param save_cfg:        save to model file or not
        :param x_size:          map size X
        :param y_size:          map size Y
        :param start:           start position
        :param terminal:        terminal position
        """
        super(UGV_Forward_Continuous_Path_Follow, self).__init__(initPhi, save_cfg, x_size, y_size, start, terminal)
        '''physical parameters'''
        # 继承自UGV
        self.miss = self.rBody * 2  # 因为是跟踪路径，所以可以适当大一些
        self.refPoints = self.points_generator()  # 贝塞尔曲线参考点
        self.bezier = Bezier(self.refPoints)  # 贝塞尔曲线
        self.curve = self.bezier.Curve()
        self.samplePoints = self.get_sample_auto(threshold=1.5)
        self.sampleNum = len(self.samplePoints)  # 采样点数量
        self.successfulFlag = [False for _ in range(self.sampleNum)]
        self.lookForward = 1  # 一共将lookForward这么多点的状态加入state中
        self.index = 1  # 当前点的索引，0是start
        self.timeMax = 20.0

        self.trajectory = [self.start]
        self.traj_length = 0
        self.trajMax = 50
        self.traj_per = 5
        self.traj_index = 0
        self.randomInitFLag = 0
        self.wMax = 10
        '''physical parameters'''

        '''rl_base'''
        self.state_dim = self.lookForward * 2 + 6
        # [ex1/sizeX, ey1/sizeY, ex2/sizeX, ey2/sizeY, ex3/sizeX, ey3/sizeY, x/sizeX, y/sizeY, phi, dx, dy, dphi]
        self.state_num = [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf]
        self.state_step = [None, None, None, None, None, None]
        self.state_space = [None, None, None, None, None, None]
        self.state_range = [[0, self.staticGain],
                            [0, self.staticGain],
                            [-math.pi, math.pi],
                            [-self.r * self.wMax, self.r * self.wMax],
                            [-self.r * self.wMax, self.r * self.wMax],
                            [-self.r / self.L * 2 * self.r * self.wMax, self.r / self.L * 2 * self.r * self.wMax]]
        self.isStateContinuous = [True, True, True, True, True, True]
        for _ in range(self.lookForward * 2):
            self.state_num.insert(0, math.inf)
            self.state_step.insert(0, None)
            self.state_space.insert(0, None)
            self.state_range.insert(0, [-self.staticGain, self.staticGain])
            self.isStateContinuous.insert(0, True)

        self.initial_state = self.get_points_state() + [self.x / self.x_size * self.staticGain,
                                                        self.y / self.y_size * self.staticGain,
                                                        self.phi, self.dx, self.dy, self.dphi]
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.action_dim = 2
        self.action_step = [None, None]
        self.action_range = [[0, self.wMax], [0, self.wMax]]  # only forward
        self.action_num = [math.inf, math.inf]
        self.action_space = [None, None]
        self.isActionContinuous = [True, True]
        self.initial_action = [0.0, 0.0]
        self.current_action = self.initial_action.copy()

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-子目标成功 4-碰撞障碍物 5-最终成功
        '''rl_base'''

        '''visualization_opencv'''
        self.show_dynamic_imagePathFollow(isWait=False)
        '''visualization_opencv'''

        '''datasave'''
        self.saveX = [self.x]
        self.saveY = [self.y]
        self.savePhi = [self.phi]
        self.savedX = [self.dx]
        self.savedY = [self.dy]
        self.savedPhi = [self.dphi]
        self.savewLeft = [self.wLeft]
        self.savewRight = [self.wRight]
        self.saveTime = [self.time]
        '''datasave'''
        if save_cfg:
            self.saveModel2XML2()

    '''DRAW'''

    def draw_bezier_curve(self):
        # for pt in self.refPoints[1:-1]:
        #     cv.circle(self.image, self.dis2pixel(pt), 5, Color().DarkGreen, -1)
        for i in range(len(self.curve) - 1):
            cv.line(self.image, self.dis2pixel(self.curve[i]), self.dis2pixel(self.curve[i + 1]), Color().Blue, 2)
        for i in range(self.sampleNum - 1):
            if self.successfulFlag[i]:
                cv.circle(self.image, self.dis2pixel(self.samplePoints[i]), 5, Color().Red, -1)
            else:
                cv.circle(self.image, self.dis2pixel(self.samplePoints[i]), 5, Color().DarkGreen, -1)

    def draw_car_with_traj(self, withTraj=False):
        cv.circle(self.image, self.dis2pixel([self.x, self.y]), self.length2pixel(self.rBody), Color().Orange, -1)  # 主体
        '''两个车轮'''
        # left
        pts_left = [[self.r, self.rBody], [self.r, self.rBody - self.l_wheel], [-self.r, self.rBody - self.l_wheel], [-self.r, self.rBody]]
        pts_left = points_rotate(pts_left, self.phi)
        pts_left = points_move(pts_left, [self.x, self.y])
        cv.fillConvexPoly(self.image, points=np.array([list(self.dis2pixel(pt)) for pt in pts_left]), color=Color().Red)
        # right
        pts_right = [[self.r, -self.rBody], [self.r, -self.rBody + self.l_wheel], [-self.r, -self.rBody + self.l_wheel], [-self.r, -self.rBody]]
        pts_right = points_rotate(pts_right, self.phi)
        pts_right = points_move(pts_right, [self.x, self.y])
        cv.fillConvexPoly(self.image, points=np.array([list(self.dis2pixel(pt)) for pt in pts_right]), color=Color().Red)
        '''两个车轮'''
        # 额外画一个圆形，标志头
        head = [self.r - 0.02, 0]
        head = points_rotate(head, self.phi)
        head = points_move(head, [self.x, self.y])
        cv.circle(self.image, self.dis2pixel(head), self.length2pixel(0.04), Color().Black, -1)  # 主体
        if withTraj:
            p1 = self.trajectory[0]
            for p2 in self.trajectory[1:]:
                cv.line(self.image, self.dis2pixel(p1), self.dis2pixel(p2), Color().Purple, 2)
                p1 = p2.copy()

    def draw_unit__direction_vector(self):
        if self.index < self.sampleNum - 1:
            cv.line(self.image, self.dis2pixel(self.samplePoints[self.index]), self.dis2pixel(self.samplePoints[self.index+1]), Color().Thistle, 2)

    def show_dynamic_imagePathFollow(self, isWait):
        self.image = self.image_temp.copy()
        self.draw_region_grid(xNUm=3, yNum=3)
        self.map_draw_obs()
        self.map_draw_boundary()
        self.draw_bezier_curve()
        self.draw_unit__direction_vector()
        self.draw_car_with_traj(withTraj=True)
        self.draw_terminal()
        cv.putText(self.image, str(round(self.time, 3)), (0, 15), cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Purple, 1)
        cv.putText(self.image,
                   'dis: ' + str(round(dis_two_points([self.x, self.y], self.terminal), 3)), (120, 15), cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Purple, 1)
        cv.imshow(self.name4image, self.image)
        cv.waitKey(0) if isWait else cv.waitKey(1)
        self.save = self.image.copy()

    '''DRAW'''

    '''PATH FOLLOWING'''

    def get_points_state(self):
        pts = []
        # print(self.samplePoints)
        for i in range(self.lookForward):
            _index = min(self.sampleNum - 1, self.index + i)
            node = self.samplePoints[_index]
            pts.append((node[0] - self.x) / self.x_size * self.staticGain)
            pts.append((node[1] - self.y) / self.y_size * self.staticGain)
        return pts

    def points_generator(self):
        """
        :return:
        """
        k = cal_vector_rad([1, 0], [self.terminal[0] - self.start[0], self.terminal[1] - self.start[1]]) * np.sign(self.terminal[1] - self.start[1])
        b = self.start[1] - k * self.start[0]
        dx = np.fabs(self.terminal[0] - self.start[0])
        dy = np.fabs(self.terminal[1] - self.start[1])
        maxd = max(dx, dy)  #
        nodeNum = int(maxd / self.L / 2) + 2
        '''根据偏移大的一个采样'''
        if dx >= dy:
            x = np.linspace(self.start[0], self.terminal[0], nodeNum)
            y = k * x + b
            bias = np.random.uniform(low=-maxd / 3, high=maxd / 3, size=nodeNum)
            xx = x
            yy = np.clip(y + bias / np.sqrt(k ** 2 + 1) * np.sign(bias), 1.0, self.y_size - 1.0)
        else:
            y = np.linspace(self.start[1], self.terminal[1], nodeNum)
            x = (y - b) / k
            bias = np.random.uniform(low=-maxd / 2, high=maxd / 2, size=nodeNum)
            yy = y
            xx = np.clip(x - bias / k, 1.0, self.x_size - 1.0)
        '''根据偏移大的一个采样'''
        bezier_nodes = np.array([xx, yy]).T  # 已经为贝塞尔曲线准备好点
        bezier_nodes[0] = self.start
        bezier_nodes[-1] = self.terminal
        return bezier_nodes

    def get_sample_auto(self, threshold):
        if dis_two_points(self.start, self.terminal) <= 1.5 * threshold:
            return [self.start, self.terminal]
        if 1.5 * threshold < dis_two_points(self.start, self.terminal) <= 2 * threshold:
            return [self.start,
                    [(self.start[0] + self.terminal[0]) / 2, (self.start[1] + self.terminal[1]) / 2],
                    self.terminal]
        samples = [self.start]
        index = 0
        for pt in self.curve[1: -1]:
            if dis_two_points(samples[index], pt) > threshold:  # 增加点
                samples.append(pt)
                index += 1
            if dis_two_points(pt, self.terminal) <= threshold:  # 截止
                break
        samples.append(self.terminal)
        return samples

    '''PATH FOLLOWING'''

    def is_out(self):
        """
        :return:
        """
        '''简化处理，只判断中心的大圆有没有出界就好'''
        if (self.x + self.rBody > self.x_size) or (self.x - self.rBody < 0) or (self.y + self.rBody > self.y_size) or (self.y - self.rBody < 0):
            return True
        return False

    def is_Terminal(self, param=None):
        self.terminal_flag = 0
        if self.time > self.timeMax:
            print('...time out...')
            self.terminal_flag = 2
            return True
        if self.delta_phi_absolute > 4 * math.pi + deg2rad(0) and dis_two_points([self.x, self.y], [self.initX, self.initY]) <= 1.0:
            print('...转的角度太大了...')
            self.terminal_flag = 1
            return True
        if dis_two_points([self.x, self.y], self.samplePoints[self.index]) <= self.miss:
            if self.index == len(self.samplePoints) - 1:
                print('...成功，回合结束...')
                self.terminal_flag = 5
                self.successfulFlag[self.index] = True
                return True
            else:
                # print('...第' + str(self.index) + '个目标成功...')
                self.successfulFlag[self.index] = True
                self.terminal_flag = 3              # sub-terminal successful
                self.index += 1
                # self.dx = 0.  # 每到达一个节点，就按照第一阶段学习的结果初始化
                # self.dy = 0.
                self.dphi = 0.
                # print('...重置速度，角速度...')
                return False
        if self.is_out():
            # print('...out...')
            # self.terminal_flag = 1
            return False
        return False

    def get_reward(self, param=None):
        # cex = self.current_state[0] * self.x_size / self.staticGain
        # cey = self.current_state[1] * self.y_size / self.staticGain
        # nex = self.next_state[0] * self.x_size / self.staticGain
        # ney = self.next_state[1] * self.y_size / self.staticGain
        # currentError = math.sqrt(cex ** 2 + cey ** 2)
        # nextError = math.sqrt(nex ** 2 + ney ** 2)
        cex, cey, nex, ney, currentError, nextError = [], [], [], [], [], []
        for i in range(self.lookForward):
            cex.append(self.current_state[2 * i] * self.x_size / self.staticGain)
            cey.append(self.current_state[2 * i + 1] * self.y_size / self.staticGain)
            nex.append(self.next_state[2 * i] * self.x_size / self.staticGain)
            ney.append(self.next_state[2 * i + 1] * self.y_size / self.staticGain)
            currentError.append(math.sqrt(cex[i] ** 2 + cey[i] ** 2))  # 当前时刻距离每一个点的误差
            nextError.append(math.sqrt(nex[i] ** 2 + ney[i] ** 2))  # next时刻距离每一个点的误差

        '''r1 是每运行一步的常值惩罚'''
        r1 = -1  # 常值误差，每运行一步，就 -1
        '''r1 是每运行一步的常值惩罚'''

        '''r2 是位置误差变化的奖励'''
        if currentError[0] > nextError[0] + 1e-3:
            r2 = 5
        elif 1e-3 + currentError[0] < nextError[0]:
            r2 = -5
            # if self.index == self.sampleNum - 1:
            # print(currentError[0], nextError[0])
            # if self.terminal_flag != 3:
            #     self.is_terminal = True       # TODO 直接终止
        else:
            r2 = 0
        '''r2 是位置误差变化的奖励'''

        '''r3 是角度误差变化的奖励'''
        currentPhi = self.current_state[2 * self.lookForward + 2]
        nextPhi = self.next_state[2 * self.lookForward + 2]
        if self.index == len(self.samplePoints) - 1:
            currentThetaError = cal_vector_rad([cex[0], cey[0]], [math.cos(currentPhi), math.sin(currentPhi)])
            nextThetaError = cal_vector_rad([nex[0], nex[0]], [math.cos(nextPhi), math.sin(nextPhi)])
        else:
            if self.lookForward > 1:
                currentThetaError = cal_vector_rad([cex[1] - cex[0], cey[1] - cey[0]], [cex[0], cey[0]])
                nextThetaError = cal_vector_rad([nex[1] - nex[0], ney[1] - ney[0]], [nex[0], ney[0]])
            else:
                currentThetaError = cal_vector_rad([cex[0], cey[0]], [math.cos(currentPhi), math.sin(currentPhi)])
                nextThetaError = cal_vector_rad([nex[0], nex[0]], [math.cos(nextPhi), math.sin(nextPhi)])
        if currentThetaError > nextThetaError + 1e-2:
            r3 = 2
        elif 1e-2 + currentThetaError < nextThetaError:
            r3 = -2
        else:
            r3 = 0
        '''r3 是角度误差变化的奖励'''

        '''4. 其他'''  # 0-正常 1-转角太大 2-超时 3-子目标成功 4-碰撞障碍物 5-最终成功
        if self.terminal_flag == 3:
            r4 = 50
        elif self.terminal_flag == 5:
            r4 = 500
        elif self.terminal_flag == 1:  # 转的角度太大
            r4 = -2
        else:
            r4 = 0
        '''4. 其他'''
        # print('r1=', r1, 'r2=', r2, 'r3=', r3, 'r4=', r4)
        self.reward = r1 + r2 + r3 + r4

    def f(self, _phi):
        _dx = self.r / 2 * (self.wLeft + self.wRight) * math.cos(_phi)
        _dy = self.r / 2 * (self.wLeft + self.wRight) * math.sin(_phi)
        _dphi = self.r / self.rBody * (self.wRight - self.wLeft)
        return np.array([_dx, _dy, _dphi])

    def step_update(self, action: list):
        self.wLeft = max(min(action[0], self.wMax), 0)
        self.wRight = max(min(action[1], self.wMax), 0)
        self.current_action = action.copy()
        self.current_state = self.get_points_state() + [self.x / self.x_size * self.staticGain,
                                                        self.y / self.y_size * self.staticGain,
                                                        self.phi, self.dx, self.dy, self.dphi]

        '''RK-44'''
        h = self.dt / 10
        t_sim = 0
        state = np.array([self.x, self.y, self.phi])
        while t_sim <= self.dt:
            K1 = self.f(state[2])
            K2 = self.f(state[2] + h * K1[2] / 2)
            K3 = self.f(state[2] + h * K2[2] / 2)
            K4 = self.f(state[2] + h * K3[2])
            state = state + h * (K1 + 2 * K2 + 2 * K3 + K4) / 6
            t_sim += h
        '''RK-44'''
        '''动力学系统状态更新'''
        [self.x, self.y, self.phi] = list(state)
        self.dx = self.r / 2 * (self.wLeft + self.wRight) * math.cos(self.phi)
        self.dy = self.r / 2 * (self.wLeft + self.wRight) * math.sin(self.phi)
        self.dphi = self.r / self.rBody * (self.wRight - self.wLeft)
        self.time += self.dt
        '''动力学系统状态更新'''
        self.delta_phi_absolute += math.fabs(self.phi - self.current_state[2 * self.lookForward + 2])
        '''角度处理'''
        if self.phi > math.pi:
            self.phi -= 2 * math.pi
        if self.phi < -math.pi:
            self.phi += 2 * math.pi
        '''角度处理'''
        self.is_terminal = self.is_Terminal()  # 负责判断回合是否结束，改变标志位，改变当前子目标
        '''出界处理'''
        if self.x + self.rBody > self.x_size:  # Xout
            self.x = self.x_size - self.rBody
            self.dx = 0
        if self.x - self.rBody < 0:
            self.x = self.rBody
            self.dx = 0
        if self.y + self.rBody > self.y_size:  # Yout
            self.y = self.y_size - self.rBody
            self.dy = 0
        if self.y - self.rBody < 0:
            self.y = self.rBody
            self.dy = 0
        '''出界处理'''

        self.next_state = self.get_points_state() + [self.x / self.x_size * self.staticGain,
                                                     self.y / self.y_size * self.staticGain,
                                                     self.phi, self.dx, self.dy, self.dphi]

        '''处理轨迹绘制'''
        if self.traj_index % self.traj_per == 0:
            if self.traj_length >= self.trajMax:
                self.trajectory.pop(0)
                self.trajectory.append([self.x, self.y])
            else:
                self.traj_length += 1
                self.trajectory.append([self.x, self.y])
            self.traj_index = 0
        self.traj_index += 1
        '''处理轨迹绘制'''

        self.get_reward()
        self.saveData()

        return self.current_state, action, self.reward, self.next_state, self.is_terminal

    def reset(self):
        """
        :brief:     reset
        :return:    None
        """
        '''physical parameters'''
        self.x = self.initX  # X
        self.y = self.initY  # Y
        self.phi = self.initPhi  # 车的转角
        self.dx = 0
        self.dy = 0
        self.dphi = 0
        self.wLeft = 0.
        self.wRight = 0.
        self.time = 0.  # time
        self.delta_phi_absolute = 0.
        self.index = 0  # 当前点的索引
        self.trajectory = [self.start]
        self.traj_length = 0
        self.traj_index = 0
        self.index = 1
        self.successfulFlag = [False for _ in range(self.sampleNum)]
        '''physical parameters'''

        '''RL_BASE'''
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        '''RL_BASE'''

        '''data_save'''
        self.saveX = [self.x]
        self.saveY = [self.y]
        self.savePhi = [self.phi]
        self.savedX = [self.dx]
        self.savedY = [self.dy]
        self.savedPhi = [self.dphi]
        self.savewLeft = [self.wLeft]
        self.savewRight = [self.wRight]
        '''data_save'''

    def reset_random(self, uniform=False):
        """
        :return:
        """
        '''physical parameters'''
        if not uniform:
            self.set_start([random.uniform(0, self.x_size), random.uniform(2 * self.y_size / 3, self.y_size)])
            self.set_terminal([random.uniform(0, self.x_size), random.uniform(2 * self.y_size / 3, self.y_size)])
            self.start_clip([3 * self.rBody, 3 * self.rBody], [self.x_size - 3 * self.rBody, self.y_size - 3 * self.rBody])
            self.terminal_clip([3 * self.rBody, 3 * self.rBody], [self.x_size - 3 * self.rBody, self.y_size - 3 * self.rBody])
        else:
            self.randomInitFLag = self.randomInitFLag % 81
            start = self.randomInitFLag % 9         # start的区域编号
            terminal = self.randomInitFLag // 9      # terminal的区域编号
            stepX = self.x_size / 3
            stepY = self.y_size / 3
            s_X = [stepX * start % 3, stepX * (start % 3 + 1)]
            s_Y = [stepY * start // 3, stepY * (start // 3 + 1)]
            t_X = [stepX * terminal % 3, stepX * (terminal % 3 + 1)]
            t_Y = [stepY * terminal % 3, stepY * (terminal % 3 + 1)]
            self.randomInitFLag += 1
            self.set_start([random.uniform(s_X[0], s_X[1]), random.uniform(s_Y[0], s_Y[1])])
            self.set_terminal([random.uniform(t_X[0], t_X[1]), random.uniform(t_Y[0], t_Y[1])])
            self.start_clip(self.rBody, self.x_size - self.rBody)
            self.terminal_clip(self.rBody, self.y_size - self.rBody)
        self.x = self.start[0]  # X
        self.y = self.start[1]  # Y
        self.initX = self.start[0]
        self.initY = self.start[1]

        self.index = 1
        self.refPoints = self.points_generator()  # 贝塞尔曲线参考点
        self.bezier = Bezier(self.refPoints)  # 贝塞尔曲线
        self.curve = self.bezier.Curve()
        self.samplePoints = self.get_sample_auto(threshold=1.5)
        self.sampleNum = len(self.samplePoints)  # 采样点数量
        self.trajectory = [self.start]
        self.traj_length = 0
        self.traj_index = 0
        self.successfulFlag = [False for _ in range(self.sampleNum)]

        phi0 = cal_vector_rad([self.samplePoints[1][0] - self.x, self.samplePoints[1][1] - self.y], [1, 0]) * np.sign(self.samplePoints[1][1] - self.y)
        # print(rad2deg(phi0))
        self.phi = random.uniform(phi0 - deg2rad(45), phi0 + deg2rad(45))  # 将初始化的角度放在初始对准目标的90度范围内
        '''角度处理'''
        if self.phi > math.pi:
            self.phi -= 2 * math.pi
        if self.phi < -math.pi:
            self.phi += 2 * math.pi
        '''角度处理'''
        self.initPhi = self.phi

        self.dx = 0
        self.dy = 0
        self.dphi = 0
        self.wLeft = 0.
        self.wRight = 0.
        self.time = 0.  # time
        self.delta_phi_absolute = 0.
        '''physical parameters'''

        '''RL_BASE'''
        self.initial_state = self.get_points_state() + [self.x / self.x_size * self.staticGain,
                                                        self.y / self.y_size * self.staticGain,
                                                        self.phi, self.dx, self.dy, self.dphi]
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        '''RL_BASE'''

        '''data_save'''
        self.saveX = [self.x]
        self.saveY = [self.y]
        self.savePhi = [self.phi]
        self.savedX = [self.dx]
        self.savedY = [self.dy]
        self.savedPhi = [self.dphi]
        self.savewLeft = [self.wLeft]
        self.savewRight = [self.wRight]
        '''data_save'''

    def saveModel2XML2(self, filename='UGV_Forward_Continuous_Path_Follow.xml', filepath='../config/'):
        rootMsg = {
            'name': 'UGV_Forward_Continuous',
            'author': 'Yefeng YANG',
            'date': '2022.01.17',
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
            'initX': self.initX,
            'initY': self.initY,
            'initPhi': self.initPhi,
            'x': self.x,
            'y': self.y,
            'phi': self.phi,
            'dx': self.dx,
            'dy': self.dy,
            'dphi': self.dphi,
            'wLeft': self.wLeft,
            'wRight': self.wRight,
            'wMax': self.wMax,
            'r': self.r,
            'l_wheel': self.l_wheel,
            'rBody': self.rBody,
            'L': self.L,
            'dt': self.dt,
            'time': self.time,
            'x_size': self.x_size,
            'y_size': self.y_size,
            'miss': self.miss,
            'timeMax': self.timeMax
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

    def saveData(self, is2file=False, filename='UGV_Forward_Continuous_Path_Follow.csv', filepath=''):
        if is2file:
            data = pd.DataFrame({
                'x:': self.saveX,
                'y': self.saveY,
                'phi': self.savePhi,
                'dx': self.savedX,
                'dy': self.savedY,
                'dphi': self.savedPhi,
                'wLeft': self.savewLeft,
                'wRight': self.savewRight,
                'time': self.saveTime
            })
            data.to_csv(filepath + filename, index=False, sep=',')
        else:
            self.saveX = [self.x]
            self.saveY = [self.y]
            self.savePhi = [self.phi]
            self.savedX = [self.dx]
            self.savedY = [self.dy]
            self.savedPhi = [self.dphi]
            self.savewLeft = [self.wLeft]
            self.savewRight = [self.wRight]
            self.saveTime = [self.time]
