import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from common.common import *
import random
from environment.envs.flight_attitude_simulator import Flight_Attitude_Simulator
from environment.envs.flight_attitude_simulator_continuous import Flight_Attitude_Simulator_Continuous
from environment.envs.ugv_bidirectional_continuous import UGV_Bidirectional_Continuous
from environment.envs.ugv_forward_continuous import UGV_Forward_Continuous


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


def test_ugv_bidirectional():
    env = UGV_Bidirectional_Continuous(initPhi=deg2rad(-135),
                                       save_cfg=True,
                                       x_size=4.0,
                                       y_size=4.0,
                                       start=[2.0, 2.0],
                                       terminal=[4.0, 4.0])
    env.reset()
    while not env.is_terminal:
        # print(env.time)
        env.show_dynamic_image(isWait=True)
        action = [-3, 3]
        env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action=action)
        print(env.reward)
        # print(env.current_state)


def test_two_ugv_forward_continuous():
    env = UGV_Forward_Continuous(initPhi=deg2rad(0),
                                 save_cfg=True,
                                 x_size=4.0,
                                 y_size=4.0,
                                 start=[2.0, 2.0],
                                 terminal=[4.0, 4.0])
    while True:
        env.reset_random()
        env.show_dynamic_image(isWait=True)
        while not env.is_terminal:
            # print(env.time)
            env.show_dynamic_image(isWait=False)
            action = [9, 9]
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action=action)
            print(env.reward)
            # print(env.current_state)


if __name__ == '__main__':
    # test_flight_attitude_simulator()
    # test_flight_attitude_simulator_continuous()
    # test_ugv_bidirectional()
    test_two_ugv_forward_continuous()
    pass
