from common.common import *
from environment.envs import *
from environment.envs.pathplanning.rasterizedmap import rasterizedmap
from environment.envs.pathplanning.samplingmap import samplingmap


class UGV_Forward_Obstacle_Continuous(rasterizedmap, rl_base):
    def __init__(self,
                 initPhi: float,
                 save_cfg: bool,
                 x_size: float,
                 y_size: float,
                 start: list,
                 terminal: list,
                 x_grid: int,
                 y_grid: int):
        """
        :param initPhi:         initial heading angle
        :param save_cfg:        save to model file or not
        :param x_size:          map size X
        :param y_size:          map size Y
        :param start:           start position
        :param terminal:        terminal position
        """
        sample_map = samplingmap(width=500,
                                 height=500,
                                 x_size=x_size,
                                 y_size=y_size,
                                 image_name='ugv_forward_obstacle',
                                 start=start,
                                 terminal=terminal,
                                 obs=[],
                                 map_file=None,
                                 draw=False)
        super(UGV_Forward_Obstacle_Continuous, self).__init__(sample_map, x_grid, y_grid)
        '''physical parameters'''
        self.initX = self.sampling_map.start[0]
        self.initY = self.sampling_map.start[1]
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
        '''physical parameters'''

        '''rl_base'''   # TODO
        self.state_dim = 8  # [ex/sizeX, ey/sizeY, x/sizeX, y/sizeY, phi, dx, dy, dphi]
        self.state_num = [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf]
        self.state_step = [None, None, None, None, None, None, None, None]
        self.state_space = [None, None, None, None, None, None, None, None]
        self.state_range = [[-self.staticGain, self.staticGain],
                            [-self.staticGain, self.staticGain],
                            [0, self.staticGain],
                            [0, self.staticGain],
                            [-math.pi, math.pi],
                            [-self.r * self.wMax, self.r * self.wMax],
                            [-self.r * self.wMax, self.r * self.wMax],
                            [-self.r / self.L * 2 * self.r * self.wMax, self.r / self.L * 2 * self.r * self.wMax]]
        self.isStateContinuous = [True, True, True, True, True, True, True, True]
        self.initial_state = [(self.sampling_map.terminal[0] - self.x) / self.sampling_map.x_size * self.staticGain,
                              (self.sampling_map.terminal[1] - self.y) / self.sampling_map.y_size * self.staticGain,
                              self.x / self.sampling_map.x_size * self.staticGain,
                              self.y / self.sampling_map.y_size * self.staticGain,
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
        self.terminal_flag = 0  # 0-正常 1-出界 2-超时 3-成功
        '''rl_base'''