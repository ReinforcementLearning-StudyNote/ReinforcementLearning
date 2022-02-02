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
                                 x_size=10.0,
                                 y_size=10.0,
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
    agent.load_actor_optimal(path='./DDPG-UGV-Forwad-Best/', file='Actor_ddpg')
    simulation_num = 100
    x_sep = 3
    y_sep = 3
    xStep = env.x_size / x_sep
    yStep = env.y_size / y_sep
    successfulRatio = []
    failRatio = []
    for x in range(x_sep * y_sep):          # control the start
        for y in range(x_sep * y_sep):      # control the terminal
            x = 3
            y = 2
            successCounter = 0
            failCounter = 0
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
                env.start[0] = max(min(env.start[0], env.x_size - env.L), env.L)
                env.start[1] = max(min(env.start[1], env.y_size - env.L), env.L)
                env.terminal[0] = max(min(env.terminal[0], env.x_size - env.L), env.L)
                env.terminal[1] = max(min(env.terminal[1], env.y_size - env.L), env.L)
                env.x = env.start[0]  # X
                env.y = env.start[1]  # Y
                phi0 = cal_vector_rad([env.terminal[0] - env.x, env.terminal[1] - env.y], [1, 0])
                phi0 = phi0 if env.y <= env.terminal[1] else -phi0
                # print(rad2deg(phi0))
                env.phi = random.uniform(phi0 - deg2rad(45), phi0 + deg2rad(45))  # 将初始化的角度放在初始对准目标的90度范围内
                '''角度处理'''
                if env.phi > math.pi:
                    env.phi -= 2 * math.pi
                if env.phi < -math.pi:
                    env.phi += 2 * math.pi
                '''角度处理'''
                env.initPhi = env.phi
                '''set start and target randomly according to x and y'''
                while not env.is_terminal:
                    if cv.waitKey(1) == 27:
                        break
                    env.current_state = env.next_state.copy()
                    action_from_actor = agent.choose_action(env.current_state, True)
                    action = agent.action_linear_trans(action_from_actor)       # 将动作转换到实际范围上
                    currentError = dis_two_points([env.x, env.y], env.terminal)
                    env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action)
                    nextError = dis_two_points([env.x, env.y], env.terminal)
                    env.show_dynamic_image(isWait=False)
                    env.saveData(is2file=False)
                    if 1e-2 + currentError < nextError:
                        print('TMD，调头了...失败')
                        env.terminal_flag = 2
                        env.is_terminal = True
                print('===========END===========')
                if env.terminal_flag == 2:
                    failCounter += 1
                if env.terminal_flag == 3:
                    successCounter += 1
            print('Part:', [x, y], '  total:', simulation_num, '  successful:', successCounter, '  timeout:', failCounter)
            successfulRatio.append(successCounter / simulation_num)
            failRatio.append(failCounter / simulation_num)
            print('...successfulRatio...', successfulRatio)
            print('...timeoutRatio...', failRatio)
    for x in range(x_sep * y_sep):          # control the start
        for y in range(x_sep * y_sep):
            print('start region:', x, 'terminal region:', y, 'success rate：', successfulRatio[x * x_sep * y_sep + y])
    cv.waitKey(0)
    env.saveData(is2file=True, filepath=simulationPath)
