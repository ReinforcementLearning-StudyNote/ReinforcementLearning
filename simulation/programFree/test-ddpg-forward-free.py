import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from environment.envs.ugv_forward_continuous import UGV_Forward_Continuous
from algorithm.actor_critic.DDPG import DDPG
import cv2 as cv
from common.common import *
import datetime

cfgPath = '../../environment/config/'
cfgFile = 'UGV_Forward_Continuous.xml'
optPath = '../../datasave/network/'


if __name__ == '__main__':
    simulationPath = '../../datasave/log/' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-DDPG-UGV-Forward/'
    os.mkdir(simulationPath)

    env = UGV_Forward_Continuous(initPhi=deg2rad(0),
                                 save_cfg=False,
                                 x_size=4.0,
                                 y_size=4.0,
                                 start=[2.0, 2.0],
                                 terminal=[4.0, 4.0])
    '''初始位置，初始角度，目标位置均为随机'''
    agent = DDPG(gamma=0.9,
                 actor_learning_rate=1e-4,
                 critic_learning_rate=1e-3,
                 actor_soft_update=1e-2,
                 critic_soft_update=1e-2,
                 memory_capacity=40000,
                 batch_size=512,
                 modelFileXML=cfgPath + cfgFile,
                 path=simulationPath)

    c = cv.waitKey(1)

    print('TESTing...')
    agent.load_actor_optimal(path='../PG_based/DDPG-UGV-Forward测试/', file='Actor_ddpg')
    simulation_num = 50
    x_sep = 3
    y_sep = 3
    xStep = env.x_size / x_sep
    yStep = env.y_size / y_sep
    successfulRatio = []
    timeoutRatio = []
    for x in range(x_sep * y_sep):          # control the start
        for y in range(x_sep * y_sep):      # control the terminal
            cap = cv.VideoWriter(simulationPath + '/' + 'Optimal' + str(y + x * x_sep * y_sep) + '.mp4',
                                 cv.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                 120.0,
                                 (env.width, env.height))
            successCounter = 0
            timeOutCounter = 0
            xxS = x // x_sep
            yyS = x % x_sep
            xxT = y // y_sep
            yyT = y % y_sep
            for i in range(simulation_num):
                print('==========START==========')
                print('episode = ', i)
                env.reset_random()
                '''set start and target randomly according to x and y'''
                env.set_start([random.uniform(xxS * xStep, (xxS + 1) * xStep), random.uniform(yyS * yStep, (yyS + 1) * yStep)])
                env.set_terminal([random.uniform(xxT * xStep, (xxT + 1) * xStep), random.uniform(yyT * yStep, (yyT + 1) * yStep)])
                env.start[0] = max(min(env.start[0], env.x_size - 0.3), 0.3)
                env.start[1] = max(min(env.start[1], env.y_size - 0.3), 0.3)
                env.terminal[0] = max(min(env.terminal[0], env.x_size - 0.3), 0.3)
                env.terminal[1] = max(min(env.terminal[1], env.y_size - 0.3), 0.3)
                env.x = env.start[0]  # X
                env.y = env.start[1]  # Y
                '''set start and target randomly according to x and y'''
                while not env.is_terminal:
                    if cv.waitKey(1) == 27:
                        break
                    env.current_state = env.next_state.copy()
                    action_from_actor = agent.choose_action(env.current_state, True)
                    action = agent.action_linear_trans(action_from_actor)       # 将动作转换到实际范围上
                    env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action)
                    env.show_dynamic_image(isWait=False)
                    cap.write(env.save)
                    env.saveData(is2file=False)
                print('===========END===========')
                if env.terminal_flag == 2:
                    timeOutCounter += 1
                if env.terminal_flag == 3:
                    successCounter += 1
            print('Part:', [x, y], '  total:', simulation_num, '  successful:', successCounter, '  timeout:', timeOutCounter)
            successfulRatio.append(successCounter / simulation_num)
            timeoutRatio.append(timeOutCounter / simulation_num)
            print('...successfulRatio...', successfulRatio)
            print('...timeoutRatio...', timeoutRatio)
    cv.waitKey(0)
    env.saveData(is2file=True, filepath=simulationPath)
