import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from common.common import *
import random
from environment.envs.flight_attitude_simulator import Flight_Attitude_Simulator
from environment.envs.flight_attitude_simulator_continuous import Flight_Attitude_Simulator_Continuous
from environment.envs.pathplanning.nav_emptyworld_continuous import Nav_EmptyWorld_Continuous
from environment.envs.pathplanning.nav_emptyworld import Nav_EmptyWorld
from environment.envs.two_wheel_ground_vehicle_continuous import Two_Wheel_Ground_Vehicle_Continuous


# Flight Attitude Simulator Test
def test_flight_attitude_simulator():
    env = Flight_Attitude_Simulator(initTheta=-60.0, setTheta=0., save_cfg=True)
    test_num = 1
    for _ in range(test_num):
        env.reset_random()
        while not env.is_terminal:
            env.show_dynamic_image(isWait=False)
            # action = random.choice(env.action_space)
            action = [0.0]
            env.current_state, env.current_action[0], env.reward, env.next_state, env.is_terminal = env.step_update(action=action)
    # env.saveData(is2file=True, filepath='../../datasave/')


# Flight Attitude Simulator Continuous Test
def test_flight_attitude_simulator_continuous():
    env = Flight_Attitude_Simulator_Continuous(initTheta=-60.0, setTheta=0., save_cfg=True)
    test_num = 1
    for _ in range(test_num):
        env.reset_random()
        while not env.is_terminal:
            env.show_dynamic_image(isWait=False)
            # action = random.choice(env.action_space)
            action = [0.0]
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action=action)
    # env.saveData(is2file=True, filepath='../../datasave/')


# 2D Navigation in Empty World Continuous Test
def test_nav_empty_world_continuous():
    samplingMap_dict = {'width': 500,
                        'height': 500,
                        'x_size': 10,
                        'y_size': 10,
                        'image_name': '2D_Nav_EmptyWorld',
                        'start': [5, 5],
                        'terminal': [10, 10],
                        'obs': [],
                        'draw': True}
    env = Nav_EmptyWorld_Continuous(samplingMap_dict=samplingMap_dict, vRange=[-3, 3], aRange=[-3, 3], save_cfg=True)
    for i in range(1):
        # env.reset_random()
        env.reset()
        # env.show_dynamic_image(isWait=True)
        print('======episode ', i, ' start======')
        while not env.is_terminal:
            env.show_dynamic_image(isWait=False)
            # action = [-10.,-6.]
            action = [random.uniform(env.aRange[0], env.aRange[1]), random.uniform(env.aRange[0], env.aRange[1])]
            print(env.current_state)
            # action = [3, 3]
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action=action)
        print(env.time)
        print('======episode ', i, ' end======')


# 2D Navigation in Empty World Test
def test_nav_empty_world():
    samplingMap_dict = {'width': 500,
                        'height': 500,
                        'x_size': 10,
                        'y_size': 10,
                        'image_name': '2D_Nav_EmptyWorld',
                        'start': [5, 5],
                        'terminal': [9, 7],
                        'obs': [],
                        'draw': True}
    env = Nav_EmptyWorld(samplingMap_dict=samplingMap_dict, vRange=[-3, 3], aRange=[-3, 3], save_cfg=True)
    for a in env.action_space[0]:
        # env.reset_random()
        env.reset()
        print('action:', a)
        while not env.is_terminal:
            env.show_dynamic_image(isWait=False)
            # action = [-10.,-6.]
            # action = [random.choice(env.action_space[0]), random.choice(env.action_space[0])]
            # print(env.current_state)
            action = [a]
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action=action)


def test_two_wheel_UGV():
    env = Two_Wheel_Ground_Vehicle_Continuous(initPhi=deg2rad(45),
                                              save_cfg=True,
                                              x_size=4.0,
                                              y_size=4.0,
                                              start=[2.0, 2.0],
                                              terminal=[4.0, 4.0])
    env.reset()
    while not env.is_terminal:
        # print(env.time)
        env.show_dynamic_image(isWait=True)
        action = [3, 3]
        env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action=action)
        # print(env.reward)
        print(env.current_state)


if __name__ == '__main__':
    # test_flight_attitude_simulator()
    # test_flight_attitude_simulator_continuous()
    # test_nav_empty_world_continuous()
    # test_nav_empty_world()
    test_two_wheel_UGV()
    pass
