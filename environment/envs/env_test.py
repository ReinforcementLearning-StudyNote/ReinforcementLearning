import math
import random
from environment.envs.flight_attitude_simulator import Flight_Attitude_Simulator
from environment.envs.flight_attitude_simulator_continuous import Flight_Attitude_Simulator_Continuous
from environment.envs.pathplanning.nav_emptyworld_continuous import Nav_EmptyWorld

# '''Flight Attitude Simulator Test'''


# def test_flight_attitude_simulator():
#     env = Flight_Attitude_Simulator(initTheta=-60.0, setTheta=0., save_cfg=True)
#     test_num = 1
#     for _ in range(test_num):
#         env.reset_random()
#         while not env.is_terminal:
#             env.show_dynamic_image(isWait=False)
#             # action = random.choice(env.action_space)
#             action = [0.0]
#             env.current_state, env.current_action[0], env.reward, env.next_state, env.is_terminal = env.step_update(action=action)
#     # env.saveData(is2file=True, filepath='../../datasave/')
#
#
# test_flight_attitude_simulator()
# '''Flight Attitude Simulator Test'''

# '''Flight Attitude Simulator Continuous Test'''
#
#
# def test_flight_attitude_simulator_continuous():
#     env = Flight_Attitude_Simulator_Continuous(initTheta=-60.0, setTheta=0., save_cfg=True)
#     test_num = 1
#     for _ in range(test_num):
#         env.reset_random()
#         while not env.is_terminal:
#             env.show_dynamic_image(isWait=False)
#             # action = random.choice(env.action_space)
#             action = [0.0]
#             env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action=action)
#     # env.saveData(is2file=True, filepath='../../datasave/')
#
#
# test_flight_attitude_simulator_continuous()
# '''Flight Attitude Simulator Continuous Test'''

'''2D Navigation in Empty World Test'''


def test_nav_empty_world():
    samplingMap_dict = {'width': 500,
                        'height': 500,
                        'x_size': 10,
                        'y_size': 10,
                        'image_name': '2D_Nav_EmptyWorld',
                        'start': [5, 5],
                        'terminal': [9, 9],
                        'obs': [],
                        'draw': True}
    env = Nav_EmptyWorld(samplingMap_dict=samplingMap_dict, vRange=[-3, 3], aRange=[-3, 3], jRange=[-5, 5], save_cfg=True)
    for i in range(2):
        env.reset_random()
        env.show_dynamic_image(isWait=True)
        print('...episode ', i, ' start...')
        while not env.is_terminal:
            print('monitor...')
            print('X:', env.time, env.p[0], env.v[0], env.a[0], env.j[0])
            print('Y:', env.time, env.p[1], env.v[1], env.a[1], env.j[1])
            print('monitor...')
            env.show_dynamic_image(isWait=False)
            # action = [-10.,-6.]
            action = [random.uniform(env.jRange[0], env.jRange[1]), random.uniform(env.jRange[0], env.jRange[1])]
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action=action)
        print('...episode ', i, ' end...')


test_nav_empty_world()
'''2D Navigation in Empty World Test'''
