import math

from common.common import *
from environment.envs import *
from environment.envs.pathplanning.rasterizedmap import rasterizedmap


class UGV_Forward_Obstacle_Continuous(rasterizedmap, rl_base):
    def __init__(self,
                 width: int = 500,
                 height: int = 500,
                 x_size: float = 5.0,
                 y_size: float = 5.0,
                 image_name: str = 'ugv_forward_obstacle',
                 start=None,
                 terminal: list = None,
                 obs: list = None,
                 draw: bool = False,
                 initPhi: float = 0.,
                 save_cfg: bool = True,
                 x_grid: int = 50,
                 y_grid: int = 50,
                 dataBasePath: str = './pathplanning/5X5-50X50-DataBase-Random/'):
        """
        :param width:               width of the figure
        :param height:              width of the figure
        :param x_size:              map size X
        :param y_size:              map size Y
        :param image_name:          naem of the figure
        :param start:               start position
        :param terminal:            terminal position
        :param obs:                 obstacles
        :param draw:                draw the map or not
        :param initPhi:             initial heading angle
        :param save_cfg:            save to model file or not
        :param x_grid:              number of the grids in X
        :param y_grid:              number of the grids in Y
        :param dataBasePath:        path of the database
        """
        super(UGV_Forward_Obstacle_Continuous, self).__init__(width, height, x_size, y_size, image_name, start, terminal, obs, draw, x_grid, y_grid)
        '''physical parameters'''
        self.initX = self.start[0]
        self.initY = self.start[1]
        self.initPhi = initPhi
        self.x = self.initX  # X
        self.y = self.initY  # Y
        self.phi = self.initPhi  # 车的转角
        self.dx = 0
        self.dy = 0
        self.dphi = 0
        self.wLeft = 0.
        self.wRight = 0.

        self.wMax = 10  # 车轮最大角速度rad/s
        self.r = 0.1  # 车轮半径
        self.l_wheel = 0.06  # 车轮厚度
        self.rBody = 0.15  # 车主体半径
        self.L = 2 * self.rBody  # 车主体直径
        self.dt = 0.02  # 50Hz
        self.time = 0.  # time
        self.miss = self.rBody + 0.05
        self.staticGain = 4
        self.delta_phi_absolute = 0.
        self.laser_dis = 5  # 假设车的中心所在grid沿着某一方向延申5个grid为探测中心
        self.laser_range = 4  # 以探测中心位中心，向 ‘两侧’各扩展4个格子
        self.laser_state = 1 + 2 * self.laser_range
        self.visualLaser = [[0, 0] for _ in range(self.laser_state)]
        self.regionChain = self.get_regionChain()
        # print(self.regionChain)

        '''physical parameters'''

        '''map database'''
        self.database = self.load_database(dataBasePath)  # 地图数据库
        self.numData = len(self.database)  # 数据库里面的数据数量
        '''map database'''

        '''rl_base'''
        self.state_dim = 8 + self.laser_state
        '''[ex/sizeX, ey/sizeY, x/sizeX, y/sizeY, phi, dx, dy, dphi,
            laser-4, laser-3, laser-2. laser-1. laser0, laser1, laser2, laser3, laser4]'''
        self.state_num = [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf]
        self.state_step = [None, None, None, None, None, None, None, None]
        self.state_space = [None, None, None, None, None, None, None, None]
        self.isStateContinuous = [True, True, True, True, True, True, True, True]
        self.state_range = [[-self.staticGain, self.staticGain],
                            [-self.staticGain, self.staticGain],
                            [0, self.staticGain],
                            [0, self.staticGain],
                            [-math.pi, math.pi],
                            [-self.r * self.wMax, self.r * self.wMax],
                            [-self.r * self.wMax, self.r * self.wMax],
                            [-self.r / self.L * 2 * self.r * self.wMax, self.r / self.L * 2 * self.r * self.wMax]]
        for _ in range(self.laser_state):
            self.state_num.append(2)
            self.state_step.append(None)
            self.state_space.append([0, 1])
            self.isStateContinuous.append(False)
            self.state_range.append([0, 1])

        self.initial_state = [(self.terminal[0] - self.x) / self.x_size * self.staticGain,
                              (self.terminal[1] - self.y) / self.y_size * self.staticGain,
                              self.x / self.x_size * self.staticGain,
                              self.y / self.y_size * self.staticGain,
                              self.phi, self.dx, self.dy, self.dphi] + self.get_fake_laser()
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
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功
        '''rl_base'''

        '''visualization_opencv'''
        self.show_dynamic_image(isWait=True)
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
            self.saveModel2XML()

    def draw_car(self):
        # self.image = self.image_temp.copy()
        cv.circle(self.image, self.dis2pixel([self.x, self.y]), self.length2pixel(self.rBody), Color().Orange, -1)  # 主体
        cv.circle(self.image, self.dis2pixel([self.x, self.y]), self.length2pixel(0.05), Color().White, -1)  # 主体
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
        head = [self.r + 0.05, 0]
        head = points_rotate(head, self.phi)
        head = points_move(head, [self.x, self.y])
        cv.circle(self.image, self.dis2pixel(head), self.length2pixel(0.04), Color().Black, -1)  # 主体

    def draw_terminal(self):
        if self.terminal is not None and self.terminal != []:
            cv.circle(self.image, self.dis2pixel(self.terminal), 5, Color().random_color_by_BGR(), -1)

    def draw_region_grid(self, xNUm: int = 3, yNum: int = 3):
        if xNUm <= 1 or yNum <= 1:
            pass
        else:
            xStep = self.x_size / xNUm
            yStep = self.y_size / yNum
            for i in range(yNum - 1):
                pt1 = self.dis2pixel([0, 0 + (i + 1) * yStep])
                pt2 = self.dis2pixel([self.x_size, 0 + (i + 1) * yStep])
                cv.line(self.image, pt1, pt2, Color().Red, 1)
            for i in range(xNUm - 1):
                pt1 = self.dis2pixel([0 + (i + 1) * xStep, 0])
                pt2 = self.dis2pixel([0 + (i + 1) * xStep, self.y_size])
                cv.line(self.image, pt1, pt2, Color().Red, 1)

    def draw_fake_laser(self):
        for item in self.visualLaser:
            pt1 = self.grid2pixel(coord_int=item, pos='left-bottom', xoffset=-0, yoffset=0)
            pt2 = self.grid2pixel(coord_int=item, pos='right-top', xoffset=0, yoffset=0)
            if self.map_flag[item[0]][item[1]] == 1:
                cv.rectangle(self.image, pt1, pt2, Color().Magenta, -1)
            else:
                cv.rectangle(self.image, pt1, pt2, Color().LightPink, -1)

    def show_dynamic_image(self, isWait):
        self.image = self.image_temp.copy()
        self.map_draw_gird_rectangle()  # 涂栅格
        self.map_draw_x_grid()  # 画栅格
        self.map_draw_y_grid()  # 画栅格
        self.draw_region_grid(xNUm=3, yNum=3)  # 画区域
        # self.map_draw_obs()  # 画障碍物
        self.draw_fake_laser()  # 画雷达
        self.draw_car()  # 画车
        self.draw_terminal()  # 画终点
        self.map_draw_photo_frame()  # 画最外边的白框
        self.map_draw_boundary()  # 画边界
        cv.putText(self.image, str(round(self.time, 3)), (0, 15), cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Purple, 1)
        cv.imshow(self.name4image, self.image)
        cv.waitKey(0) if isWait else cv.waitKey(1)
        self.save = self.image.copy()

    def is_out(self):
        """
        :return:
        """
        '''简化处理，只判断中心的大圆有没有出界就好'''
        if (self.x + self.rBody > self.x_size) or (self.x - self.rBody < 0) or (self.y + self.rBody > self.y_size) or (self.y - self.rBody < 0):
            return True
        return False

    def is_Terminal(self):
        # if self.is_out():
        #     print('...out...')
        #     self.terminal_flag = 1
        #     return True
        if self.delta_phi_absolute > 2 * math.pi + deg2rad(45):
            print('...转的角度太大了...')
            self.terminal_flag = 1
            return True
        if self.time > 8.0:
            print('...time out...')
            self.terminal_flag = 2
            return True
        if self.dis_two_points([self.x, self.y], self.terminal) <= self.miss:
            print('...success...')
            self.terminal_flag = 3
            return True
        if self.collision_check():
            print('...collision...')
            self.terminal_flag = 4
            return True
        self.terminal_flag = 0
        return False

    def get_regionChain(self):
        regionChain = []
        for i in range(2 * self.laser_dis):
            regionChain.append([self.laser_dis, -self.laser_dis + i])
        for i in range(2 * self.laser_dis):
            regionChain.append([self.laser_dis - i, self.laser_dis])
        for i in range(2 * self.laser_dis):
            regionChain.append([-self.laser_dis, self.laser_dis - i])
        for i in range(2 * self.laser_dis):
            regionChain.append([-self.laser_dis + i, -self.laser_dis])
        return regionChain

    def get_fake_laser(self):
        """
        :return:        the fake laser data
        """
        head_origin = self.point_in_grid([self.x, self.y])
        '''根据角度来确定模拟雷达的探测中心'''
        if -math.pi / 4 <= self.phi < math.pi / 4:  # 左
            crossPoint = [self.laser_dis, int(math.tan(self.phi) * self.laser_dis)]
        elif math.pi / 4 <= self.phi < 3 * math.pi / 4:  # 上
            crossPoint = [int(self.laser_dis / math.tan(self.phi)), self.laser_dis]
        elif -3 * math.pi / 4 <= self.phi < -math.pi / 4:  # 下
            crossPoint = [-int(self.laser_dis / math.tan(self.phi)), -self.laser_dis]
        else:  # 右
            crossPoint = [-self.laser_dis, -int(math.tan(self.phi) * self.laser_dis)]
        centerIndex = self.regionChain.index(crossPoint)  # 获得探测中心索引
        '''根据角度来确定模拟雷达的探测中心'''
        '''确定探测范围'''
        laser = []
        for i in range(1 + 2 * self.laser_range):
            index = centerIndex + (-self.laser_range + i)
            if index >= len(self.regionChain):
                index -= len(self.regionChain)
            grid = [head_origin[j] + self.regionChain[index][j] for j in [0, 1]]
            grid[0] = max(min(grid[0], self.x_grid - 1), 0)
            grid[1] = max(min(grid[1], self.y_grid - 1), 0)
            self.visualLaser[i] = grid.copy()
            laser.append(self.map_flag[grid[0]][grid[1]])
        '''确定探测范围'''
        return laser

    def collision_check(self):
        """
        :return:        True or False
        """
        '''简化查找过程，直接圆心周围一圈的网格中心，虽不精确，但是相差无几'''
        carCenterIndex = self.point_in_grid([self.x, self.y])
        for i in [-2, -1, 0, 1, 2]:
            for j in [-2, -1, 0, 1, 2]:
                index = [carCenterIndex[0] + i, carCenterIndex[1] + j]
                index[0] = max(min(index[0], self.x_grid - 1), 0)
                index[1] = max(min(index[1], self.y_grid - 1), 0)
                if (self.map_flag[index[0]][index[1]] == 1) and self.point_is_in_circle([self.x, self.y], self.rBody, self.grid_center_point(index)):
                    return True
        return False

    def get_reward(self, param=None):
        cex = self.current_state[0] * self.x_size / self.staticGain
        cey = self.current_state[1] * self.y_size / self.staticGain
        nex = self.next_state[0] * self.x_size / self.staticGain
        ney = self.next_state[1] * self.y_size / self.staticGain
        currentError = math.sqrt(cex ** 2 + cey ** 2)
        nextError = math.sqrt(nex ** 2 + ney ** 2)

        r1 = -1  # 常值误差，每运行一步，就 -1

        if currentError > nextError + 1e-2:
            r2 = 5
        elif 1e-2 + currentError < nextError:
            r2 = -5
        else:
            r2 = 0

        currentTheta = cal_vector_degree([cex, cey], [math.cos(self.current_state[4]), math.sin(self.current_state[4])])
        nextTheta = cal_vector_degree([nex, ney], [math.cos(self.next_state[4]), math.sin(self.next_state[4])])
        # print(currentTheta, nextTheta)
        if currentTheta > nextTheta + 1e-2:
            r3 = 2
        elif 1e-3 + currentTheta < nextTheta:
            r3 = -2
        else:
            r3 = 0

        '''4. 其他'''
        r4 = 0
        if self.terminal_flag == 3:  # 成功
            r4 = 500
        if self.terminal_flag == 1:  # 出界
            r4 = -200
        if self.terminal_flag == 4:  # 碰撞
            r4 = -50
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
        self.current_state = [(self.terminal[0] - self.x) / self.x_size * self.staticGain,
                              (self.terminal[1] - self.y) / self.y_size * self.staticGain,
                              self.x / self.x_size * self.staticGain,
                              self.y / self.y_size * self.staticGain,
                              self.phi, self.dx, self.dy, self.dphi] + self.get_fake_laser()

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
        self.delta_phi_absolute += math.fabs(self.phi - self.current_state[4])
        '''角度处理'''
        if self.phi > math.pi:
            self.phi -= 2 * math.pi
        if self.phi < -math.pi:
            self.phi += 2 * math.pi
        '''角度处理'''

        self.is_terminal = self.is_Terminal()
        self.next_state = [(self.terminal[0] - self.x) / self.x_size * self.staticGain,
                           (self.terminal[1] - self.y) / self.y_size * self.staticGain,
                           self.x / self.x_size * self.staticGain,
                           self.y / self.y_size * self.staticGain,
                           self.phi, self.dx, self.dy, self.dphi] + self.get_fake_laser()
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

    def reset_random(self):
        """
        :return:
        """
        '''physical parameters'''
        self.set_start([random.uniform(self.rBody, self.x_size - self.rBody), random.uniform(self.rBody, self.y_size - self.rBody)])
        self.set_terminal([random.uniform(self.rBody, self.x_size - self.rBody), random.uniform(self.rBody, self.y_size - self.rBody)])
        self.set_random_obstacles(20)
        self.map_rasterization()
        self.x = self.start[0]  # X
        self.y = self.start[1]  # Y
        self.initX = self.start[0]
        self.initY = self.start[1]
        self.phi = random.uniform(-math.pi, math.pi)
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
        self.initial_state = [(self.terminal[0] - self.x) / self.x_size * self.staticGain,
                              (self.terminal[1] - self.y) / self.y_size * self.staticGain,
                              self.x / self.x_size * self.staticGain,
                              self.y / self.y_size * self.staticGain,
                              self.phi, self.dx, self.dy, self.dphi] + self.get_fake_laser()
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

    def reset_random_with_database(self):
        num = random.randint(0, self.numData - 1)
        data = self.database[num]
        '''physical parameters and map'''
        self.start = data[0]
        self.terminal = data[1]
        self.obs = data[3]
        self.map_flag = data[4]
        self.x = self.start[0]  # X
        self.y = self.start[1]  # Y
        self.initX = self.start[0]
        self.initY = self.start[1]
        phi0 = cal_vector_degree([self.terminal[0] - self.x, self.terminal[1] - self.y], [1, 0])
        phi0 = phi0 if self.y <= self.terminal[1] else -phi0
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
        '''physical parameters and map'''

        '''RL_BASE'''
        self.initial_state = [(self.terminal[0] - self.x) / self.x_size * self.staticGain,
                              (self.terminal[1] - self.y) / self.y_size * self.staticGain,
                              self.x / self.x_size * self.staticGain,
                              self.y / self.y_size * self.staticGain,
                              self.phi, self.dx, self.dy, self.dphi] + self.get_fake_laser()
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

    def saveModel2XML(self, filename='UGV_Forward_Obstacle_Continuous.xml', filepath='../config/'):
        rootMsg = {
            'name': 'UGV_Forward_Obstacle_Continuous',
            'author': 'Yefeng YANG',
            'date': '2022.01.20',
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
            'miss': self.miss
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

    def saveData(self, is2file=False, filename='UGV_Forward_Continuous.csv', filepath=''):
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

    def load_database(self, path):
        DataBase = []
        names = os.listdir(path)
        for name in names:
            print('Start Loading' + name)
            DataBase.append(self.map_load_database(path + name))
            print('Finish Loading' + name)
        return self.merge_database(DataBase)
