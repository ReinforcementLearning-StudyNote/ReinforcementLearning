from common.common_func import *
from environment.envs import *


class TwoLinkManipulator(rl_base):
    def __init__(self,
                 theta0: np.ndarray = np.array([0.0, 0.0]),
                 omega0: np.ndarray = np.array([0.0, 0.0]),
                 map_size: np.ndarray = np.array([2.0, 2.0]),
                 target: np.ndarray = np.array([0.5, 0.5])):
        super(TwoLinkManipulator, self).__init__()

        self.init_theta = theta0
        self.init_omega = omega0
        self.init_target = target
        self.init_midPos = np.array([1.0, 0.65])
        self.init_endPos = np.array([1.0, 0.3])

        self.theta = theta0  # 两杆的角度，分别是杆1与y轴负半轴夹角、杆2与杆1延长线夹角，逆时针为正
        self.omega = omega0  # 两杆的角速度
        self.t = np.zeros(2, dtype=np.float32)  # 两个转轴输入转矩
        self.map_size = map_size
        self.target = target
        self.midPos = self.init_midPos.copy()  # 两杆铰接位置
        self.endPos = self.init_endPos.copy()  # 末端位置
        self.endVel = np.zeros(2)
        self.error = self.target - self.endPos

        '''hyper-parameters'''
        self.basePos = np.array([1.0, 1.0])  # 基座位置
        self.l = 0.35  # 杆长
        self.m = 0.5  # 单根杆质量
        self.g = 9.8  # 重力加速度
        self.J = self.m * (self.l ** 2) / 3  # 杆的转动惯量
        self.dt = 0.02  # 50Hz
        self.time = 0.  # time
        self.timeMax = 8.0
        '''hyper-parameters'''

        self.thetaMax = np.pi
        self.thetaMin = - np.pi
        self.omega1Max = np.pi
        self.omega1Min = -np.pi
        self.omega2Max = np.pi
        self.omega2Min = -np.pi
        self.tMax = 5.0
        self.tMin = -5.0

        self.miss = 0.01  # 容许误差
        self.name = 'TwoLinkManipulator'

        '''rl_base'''
        self.use_normalization = True
        self.static_gain = 2
        self.state_dim = 6  # error theta omega 末端误差 角度 角速度
        self.state_num = [math.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.isStateContinuous = [True for _ in range(self.state_dim)]
        self.state_range = np.array(
            [[-4 * self.l, 4 * self.l],
             [-4 * self.l, 4 * self.l],
             [self.thetaMin, self.thetaMax],
             [self.thetaMin, self.thetaMax],
             [self.omega1Min, self.omega1Max],
             [self.omega2Min, self.omega2Max]]
        )
        if self.use_normalization:
            self.initial_state = self.state_norm()
        else:
            self.initial_state = np.concatenate((self.error, self.theta, self.omega), axis=0)
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.action_dim = 2  # 两个转轴转矩
        self.action_step = [None, None]
        self.action_range = np.array(
            [[self.tMin, self.tMax],
             [self.tMin, self.tMax]]
        )
        self.action_num = [math.inf, math.inf]
        self.action_space = [None, None]
        self.isActionContinuous = [True, True]
        self.initial_action = self.t.copy()
        self.current_action = self.initial_action.copy()

        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功
        '''rl_base'''

        '''visualization'''
        self.x_offset = 20
        self.y_offset = 20
        self.board = 250
        self.pixel_per_meter = 300
        self.image_size = (np.array(self.pixel_per_meter * self.map_size) + 2 * np.array(
            [self.x_offset, self.y_offset])).astype(int)
        self.image_size[0] += self.board
        self.image = np.zeros([self.image_size[1], self.image_size[0], 3], np.uint8)
        self.image[:, :, 0] = np.ones([self.image_size[1], self.image_size[0]]) * 255
        self.image[:, :, 1] = np.ones([self.image_size[1], self.image_size[0]]) * 255
        self.image[:, :, 2] = np.ones([self.image_size[1], self.image_size[0]]) * 255
        self.image_white = self.image.copy()  # 纯白图
        self.base_x_pixel = 50
        self.base_y_pixel = 20
        '''visualization'''

        self.sum_d_theta = np.zeros(2)

    def dis2pixel(self, coord) -> tuple:
        """
        :brief:         the transformation of coordinate between physical world and image
        :param coord:   position in physical world
        :return:        position in image coordinate
        """
        x = self.x_offset + coord[0] * self.pixel_per_meter
        y = self.image_size[1] - self.y_offset - coord[1] * self.pixel_per_meter
        return int(x), int(y)

    def length2pixel(self, _l):
        """
        :brief:         the transformation of distance between physical world and image
        :param _l:      length in physical world
        :return:        length in image
        """
        return int(_l * self.pixel_per_meter)

    def show_dynamic_image(self, isWait=False):
        self.image = self.image_white.copy()
        self.draw_boundary()
        self.draw_manipulator()
        self.draw_target()
        # self.draw_grid()

        cv.putText(
            self.image,
            'time:   %.3fs' % (round(self.time, 3)),
            (self.image_size[0] - self.board - 5, 25), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)

        cv.putText(
            self.image,
            'endPos: [%.2f, %.2f]m' % (round(self.endPos[0], 3), round(self.endPos[1], 3)),
            (self.image_size[0] - self.board - 5, 60), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
        cv.putText(
            self.image,
            'error: [%.2f, %.2f]m' % (round(self.error[0], 3), round(self.error[1], 3)),
            (self.image_size[0] - self.board - 5, 95), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
        cv.putText(
            self.image,
            'theta: [%.2f, %.2f]m' % (round(rad2deg(self.theta[0]), 2), round(rad2deg(self.theta[1]), 2)),
            (self.image_size[0] - self.board - 5, 130), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)
        cv.putText(
            self.image,
            'omega: [%.2fPI, %.2fPI]m' % (round(self.omega[0] / np.pi, 2), round(self.omega[1] / np.pi, 2)),
            (self.image_size[0] - self.board - 5, 165), cv.FONT_HERSHEY_COMPLEX, 0.5, Color().Purple, 1)

        cv.imshow(self.name, self.image)
        cv.waitKey(0) if isWait else cv.waitKey(1)

    def draw_manipulator(self):
        # 基座
        base = self.dis2pixel(self.basePos)
        pt1 = (int(base[0] - self.base_x_pixel / 2), int(base[1] - self.base_y_pixel / 2))
        pt2 = (int(base[0] + self.base_x_pixel / 2), int(base[1] + self.base_y_pixel / 2))
        cv.rectangle(self.image, pt1=pt1, pt2=pt2, color=Color().DarkGray, thickness=-1)
        # 杆
        mid = self.dis2pixel(self.midPos)
        end = self.dis2pixel(self.endPos)
        pt1 = (base[0], base[1])
        pt2 = (mid[0], mid[1])
        cv.line(self.image, pt1=pt1, pt2=pt2, color=Color().Blue, thickness=4)
        pt1 = (end[0], end[1])
        pt2 = (mid[0], mid[1])
        cv.line(self.image, pt1=pt1, pt2=pt2, color=Color().Red, thickness=4)
        # 末端机械手(迫真)
        cv.line(self.image, pt1=(end[0] - 10, end[1]), pt2=(end[0] + 10, end[1]), color=Color().Chocolate2, thickness=2)
        cv.line(self.image, pt1=(end[0], end[1] - 10), pt2=(end[0], end[1] + 10), color=Color().Chocolate2, thickness=2)

    def draw_boundary(self):
        cv.line(self.image, (self.x_offset, self.y_offset),
                (self.image_size[0] - self.x_offset - self.board, self.y_offset), Color().Black, 2)
        cv.line(self.image, (self.x_offset, self.y_offset), (self.x_offset, self.image_size[1] - self.y_offset),
                Color().Black, 2)
        cv.line(
            self.image,
            (self.image_size[0] - self.x_offset - self.board, self.image_size[1] - self.y_offset),
            (self.x_offset, self.image_size[1] - self.y_offset), Color().Black, 2
        )
        cv.line(
            self.image,
            (self.image_size[0] - self.x_offset - self.board, self.image_size[1] - self.y_offset),
            (self.image_size[0] - self.x_offset - self.board, self.y_offset), Color().Black, 2
        )

    def draw_target(self):
        cv.circle(self.image, self.dis2pixel(self.target), 5, Color().random_color_by_BGR(), -1)

    def draw_grid(self, num: np.ndarray = np.array([10, 10])):
        if np.min(num) <= 1:
            pass
        else:
            step = self.map_size / num
            for i in range(num[1] - 1):
                cv.line(self.image,
                        self.dis2pixel([0, 0 + (i + 1) * step[1]]),
                        self.dis2pixel([self.map_size[0], 0 + (i + 1) * step[1]]),
                        Color().Black, 1)
            for i in range(num[0] - 1):
                cv.line(self.image,
                        self.dis2pixel([0 + (i + 1) * step[0], 0]),
                        self.dis2pixel([0 + (i + 1) * step[0], self.map_size[1]]),
                        Color().Black, 1)

    def state_norm(self):
        state = np.concatenate((self.error, self.theta, self.omega), axis=0)
        norm_min = self.state_range[:, 0]
        norm_max = self.state_range[:, 1]
        norm_s = (2 * state - (norm_min + norm_max)) / (norm_max - norm_min) * self.static_gain
        return norm_s

    def inverse_state_norm(self, s: np.ndarray):
        norm_min = self.state_range[:, 0]
        norm_max = self.state_range[:, 1]

        inverse_norm_s = (s * (norm_max - norm_min) / self.static_gain + (norm_min + norm_max)) / 2
        return inverse_norm_s

    def is_success(self):
        b1 = np.linalg.norm(self.error) <= self.miss
        b2 = np.fabs(self.omega) < deg2rad(5)
        return b1 and all(b2)

    def is_Terminal(self, param=None):
        if self.time > self.timeMax:
            # print('...time out...')
            self.terminal_flag = 2
            return True
        if self.is_success():
            print('...success...')
            self.terminal_flag = 3
            return True
        self.terminal_flag = 0
        return False

    def get_reward(self, param=None):
        if self.use_normalization:
            cur_s = self.inverse_state_norm(self.current_state)
            nex_s = self.inverse_state_norm(self.next_state)
        else:
            cur_s = self.current_state
            nex_s = self.next_state

        cur_error = np.linalg.norm(cur_s[0: 2])
        nex_error = np.linalg.norm(nex_s[0: 2])

        cur_norm_error = cur_error / np.linalg.norm([4 * self.l, 4 * self.l])
        nex_norm_error = nex_error / np.linalg.norm([4 * self.l, 4 * self.l])

        cur_norm_action = np.linalg.norm(self.t) / np.linalg.norm([self.tMax, self.tMax])

        if self.sum_d_theta[0] > 4 * np.pi or self.sum_d_theta[1] > 4 * np.pi:  # 如果转的超过两圈
            self.terminal_flag = 4

        '''4. 其他'''
        if self.terminal_flag == 4:  # 瞎几把转
            r4 = -0
        elif self.terminal_flag == 3:  # 成功
            r4 = 1000
        elif self.terminal_flag == 2:  # 超时
            r4 = -0
        elif self.terminal_flag == 1:  # 出界
            r4 = -0
        else:
            r4 = 0
        '''4. 其他'''

        '''r1 是位置'''
        delta = 0.1
        if nex_norm_error < delta:
            r1 = - nex_norm_error ** 2 / 2
        else:
            r1 = - delta * (nex_norm_error - delta / 2)
        '''r2 是输入'''
        # r1 = - nex_norm_error ** 2 / 2
        # if nex_norm_error > cur_norm_error + 1e-3:
        #     r1 = -1
        # elif nex_norm_error + 1e-3 < cur_norm_error:
        #     r1 = 1
        # else:
        #     r1 = 0
        r2 = - cur_norm_action ** 2 / 2
        self.reward = r1 * 10 + r4

    def ode(self, xx: np.ndarray):
        [_theta1, _theta2, _omega1, _omega2] = xx[:]
        _dtheta1, _dtheta2 = np.clip(_omega1, self.omega1Min, self.omega1Max), np.clip(_omega2, self.omega2Min, self.omega2Max)
        a = np.array([[self.J * (5 + 3 * np.cos(_theta2)), self.J * (1 + 3 / 2 * np.cos(_theta2))],
                     [self.J * (1 + 3 / 2 * np.cos(_theta2)), self.J]])
        b = np.array([self.t[0] + 3 / 2 * self.J * np.sin(_theta2) * (_dtheta2 ** 2) + 3 * self.J * np.sin(
            _theta2) * _dtheta1 * _dtheta2 - self.m * self.g * self.l * (
                              3 / 2 * np.sin(_theta1) + 1 / 2 * np.sin(_theta1 + _theta2)),
                      self.t[1] - 3 / 2 * self.J * np.sin(_theta2) * (
                              _dtheta1 ** 2) - 1 / 2 * self.m * self.g * self.l * np.sin(_theta1 + _theta2)])
        _domega1, _domega2 = np.linalg.solve(a, b)
        return np.array([_dtheta1, _dtheta2, _domega1, _domega2])

    def rk44(self, action: np.ndarray):
        self.t = np.clip(action, self.tMin, self.tMax)
        h = self.dt / 1
        tt = self.time + self.dt
        theta = self.theta
        while self.time < tt:
            xx_old = np.concatenate((self.theta, self.omega), axis=0)
            K1 = h * self.ode(xx_old)
            K2 = h * self.ode(xx_old + K1 / 2)
            K3 = h * self.ode(xx_old + K2 / 2)
            K4 = h * self.ode(xx_old + K3)
            xx_new = xx_old + (K1 + 2 * K2 + 2 * K3 + K4) / 6
            xx_new[2] = np.clip(xx_new[2], self.omega1Min, self.omega1Max)
            xx_new[3] = np.clip(xx_new[3], self.omega2Min, self.omega2Max)
            self.theta, self.omega = xx_new[:2].copy(), xx_new[2:].copy()
            self.time += h
        '''正运动学'''
        self.midPos = np.array([self.l * np.sin(self.theta[0]), -self.l * np.cos(self.theta[0])]) + self.basePos
        self.endPos = self.midPos + np.array([self.l * np.sin(sum(self.theta)), -self.l * np.cos(sum(self.theta))])
        self.endVel = np.array([self.l * np.cos(self.theta[0]) * self.omega[0] + self.l * np.cos(sum(self.theta)) * sum(self.omega),
                                self.l * np.sin(self.theta[0]) * self.omega[0] + self.l * np.sin(sum(self.theta)) * sum(self.omega)])
        '''正运动学'''
        self.error = self.target - self.endPos
        self.sum_d_theta += np.fabs(theta - self.theta)
        self.theta[0] = self.theta[0] - 2 * self.thetaMax if self.theta[0] > self.thetaMax else \
            self.theta[0] - 2 * self.thetaMin if self.theta[0] < self.thetaMin else self.theta[0]
        self.theta[1] = self.theta[1] - 2 * self.thetaMax if self.theta[1] > self.thetaMax else \
            self.theta[1] - 2 * self.thetaMin if self.theta[1] < self.thetaMin else self.theta[1]

    def step_update(self, action: np.ndarray):
        self.current_action = action.copy()
        if self.use_normalization:
            self.current_state = self.state_norm()
        else:
            self.current_state = np.concatenate((self.error, self.theta, self.omega), axis=0)

        '''rk44'''
        self.rk44(action=action)
        '''rk44'''

        self.is_terminal = self.is_Terminal()

        if self.use_normalization:
            self.next_state = self.state_norm()
        else:
            self.next_state = np.concatenate((self.error, self.theta, self.omega), axis=0)
        self.get_reward()

        '''其他'''

        '''其他'''

    def reset(self):
        self.theta = self.init_theta.copy()
        self.omega = self.init_omega.copy()
        self.t = np.zeros(2)
        self.target = self.init_target.copy()
        self.midPos = self.init_midPos.copy()
        self.endPos = self.init_endPos.copy()
        self.endVel = np.zeros(2)
        self.error = self.target - self.endPos
        self.time = 0.
        self.sum_d_theta = np.zeros(2)

        if self.use_normalization:
            self.initial_state = self.state_norm()
        else:
            self.initial_state = np.concatenate((self.error, self.theta, self.omega), axis=0)
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()
        self.initial_action = self.t.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功

    def reset_random(self):
        """
        与reset()区别仅为随机重置目标点位置
        @return:
        """
        '''通过极坐标方式在机械臂动作空间圆内生成随机目标点'''
        phi = random.random() * 2 * np.pi
        r = random.uniform(0.3 ** 2, (2 * self.l) ** 2)
        x = math.cos(phi) * (r ** 0.5) + self.basePos[0]
        y = math.sin(phi) * (r ** 0.5) + self.basePos[1]
        self.init_target = np.array([x, y])
        '''通过极坐标方式在机械臂动作空间圆内生成随机目标点'''

        self.theta = self.init_theta.copy()
        self.omega = self.init_omega.copy()
        self.t = np.zeros(2)
        self.target = self.init_target.copy()
        self.midPos = self.init_midPos.copy()
        self.endPos = self.init_endPos.copy()
        self.endVel = np.zeros(2)
        self.error = self.target - self.endPos
        self.time = 0.
        self.sum_d_theta = np.zeros(2)

        if self.use_normalization:
            self.initial_state = self.state_norm()
        else:
            self.initial_state = np.concatenate((self.error, self.theta, self.omega), axis=0)
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()
        self.initial_action = self.t.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功
