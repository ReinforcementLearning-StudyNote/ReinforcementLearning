import os
import random
import sys
import cv2 as cv
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "./")
from common.common import *


# Flight Attitude Simulator Test
def test_flight_attitude_simulator():
    from FlightAttitudeSimulator.flight_attitude_simulator import Flight_Attitude_Simulator
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
    from FlightAttitudeSimulator.flight_attitude_simulator_continuous import Flight_Attitude_Simulator_Continuous
    env = Flight_Attitude_Simulator_Continuous(initTheta=-60.0, setTheta=0., save_cfg=True)
    test_num = 1
    for _ in range(test_num):
        env.reset_random()
        env.theta=0
        while not env.is_terminal:
            env.show_dynamic_image(isWait=False)
            # action = random.choice(env.action_space)
            action = [0.8]
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action=action)
            print(env.reward)
    # env.saveData(is2file=True, filepath='../../datasave/')


# UGV Bidirectional Continuous Test
def test_ugv_bidirectional_continuous():
    from UGV.ugv_bidirectional_continuous import UGV_Bidirectional_Continuous
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


# UGV Forward Continuous Test
def test_ugv_forward_continuous():
    from UGV.ugv_forward_continuous import UGV_Forward_Continuous
    env = UGV_Forward_Continuous(initPhi=deg2rad(60),
                                 save_cfg=True,
                                 x_size=5.0,
                                 y_size=5.0,
                                 start=[2.5, 2.5],
                                 terminal=[4.8, 4.8])
    while True:
        env.reset_random()
        # env.reset()
        env.show_dynamic_image(isWait=True)
        while not env.is_terminal:
            # print(env.time)
            if cv.waitKey(1) == 27:
                return
            env.show_dynamic_image(isWait=False)
            # action = [9, 6]
            action = env.towards_target_PID(threshold=10, kp=10, ki=0, kd=0)
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action=action)
            # print(env.reward)
            # print(env.current_state)


# UGV Forward Obstacles Continuous Test
def test_ugv_forward_obstacles_continuous():
    from UGV.ugv_forward_obstacle_continuous import UGV_Forward_Obstacle_Continuous
    env = UGV_Forward_Obstacle_Continuous(initPhi=deg2rad(0), save_cfg=True, x_size=11, y_size=11, start=[2.5, 2.5], terminal=[4.0, 2.5],
                                          dataBasePath='./pathplanning/11X11-AllCircle1/')
    num = 0
    while num < 1:
        # env.reset_random_with_database()
        env.reset_random()
        env.start = [1, 1]
        env.terminal = [9, 9]
        # env.show_dynamic_imagewithobs(isWait=False)
        # cv.imwrite(str(num) + '.png', env.image)

        while not env.is_terminal:
            # print(env.time)
            if cv.waitKey(1) == 27:
                return
            env.show_dynamic_imagewithobs(isWait=False)
            action = env.towards_target_PID(threshold=np.inf, kp=10, kd=0, ki=0)
            # cap.write(env.save)
            # action = [0, 1 * math.pi]
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action=action)
            # print(env.current_state[0 : 4])
            # if env.terminal_flag == 4:
            #     print(env.reward)
            # print(env.current _state)
        num += 1

# UGV Forward Discrete Test
def test_ugv_forward_discrete():
    from UGV.ugv_forward_discrete import UGV_Forward_Discrete
    env = UGV_Forward_Discrete(initPhi=deg2rad(60), save_cfg=True, x_size=5.0, y_size=5.0, start=[2.5, 2.5], terminal=[4.8, 4.8])
    while True:
        env.reset_random()
        env.show_dynamic_image(isWait=True)
        while not env.is_terminal:
            if cv.waitKey(1) == 27:
                return
            env.show_dynamic_image(isWait=False)
            action = env.towards_target_PID(threshold=10, kp=10, ki=0, kd=0)
            wLeft = random.choice(env.action_space[0])
            wRight = random.choice(env.action_space[1])
            action = [wLeft, wRight]
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action=action)


# UGV Forward OBstacle Discrete Test
def test_ugv_forward_obstacles_discrete():
    from UGV.ugv_forward_obstacle_discrete import UGV_Forward_Obstacle_Discrete
    env = UGV_Forward_Obstacle_Discrete(initPhi=deg2rad(0), save_cfg=True, x_size=11, y_size=11, start=[2.5, 2.5], terminal=[4.0, 2.5],
                                          dataBasePath='./pathplanning/11X11-AllCircle1/')
    num = 0
    while num < 30:
        # cap = cv.VideoWriter('test' +str(num)+'.mp4', cv.VideoWriter_fourcc('X', 'V', 'I', 'D'), 120.0, (env.width, env.height))
        # env.reset_random()
        # env.reset()
        env.reset_random_with_database()
        env.show_dynamic_image(isWait=True)
        while not env.is_terminal:
            # print(env.time)
            if cv.waitKey(1) == 27:
                return
            env.show_dynamic_image(isWait=False)
            wLeft = random.choice(env.action_space[0])
            wRight = random.choice(env.action_space[1])
            action = [wLeft, wRight]
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action=action)
            # print(env.current_state[0 : 4])
            if env.terminal_flag == 4:
                print(env.reward)
            # print(env.current _state)
        num += 1


if __name__ == '__main__':
    # test_flight_attitude_simulator()
    # test_flight_attitude_simulator_continuous()
    # test_ugv_bidirectional_continuous()
    # test_ugv_forward_continuous()
    test_ugv_forward_obstacles_continuous()
    # test_ugv_forward_discrete()
    # test_ugv_forward_obstacles_discrete()
    pass
