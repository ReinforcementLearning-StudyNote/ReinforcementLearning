import datetime
import os
import sys
import cv2 as cv
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from environment.envs.UGV.ugv_forward_obstacle_continuous2 import UGV_Forward_Obstacle_Continuous2
from algorithm.actor_critic.DDPG2 import DDPG2
from algorithm.actor_critic.DDPG import ActorNetwork
from common.common import *

cfgPath = '../../environment/config/'
cfgFile = 'UGV_Forward_Obstacle_Continuous2.xml'
optPath = '../../datasave/network/'
dataBasePath = '../../environment/envs/pathplanning/5X5-DataBase-AllCircle2/'
dataBasePath2 = '../../environment/envs/pathplanning/5X5-DataBase-AllCircle3/'
show_per = 1


if __name__ == '__main__':
    # simulationPath = '../../datasave/log/' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-DDPG2-UGV-Forward-Obstacle2/'
    simulationPath = './sim_res/'
    if not os.path.isdir(simulationPath):
        os.mkdir(simulationPath)

    controller = ActorNetwork(1e-4, 8, 128, 128, 2, name='Actor', chkpt_dir='')
    controller.load_state_dict(torch.load('./DDPG-UGV-Forward-Best/Actor_ddpg'))
    env = UGV_Forward_Obstacle_Continuous2(initPhi=0.,
                                           save_cfg=False,
                                           x_size=5.0,
                                           y_size=5.0,
                                           start=[2.5, 2.5],
                                           terminal=[4.5, 4.5],
                                           dataBasePath='./DataBase/obs09/',
                                           controller=controller)
    '''初始位置，初始角度，目标位置均为随机'''
    agent = DDPG2(gamma=0.99,
                  actor_learning_rate=1e-4,
                  critic_learning_rate=1e-3,
                  actor_soft_update=1e-2,
                  critic_soft_update=1e-2,
                  memory_capacity=100000,  # 100000
                  batch_size=512,  # 1024
                  modelFileXML=cfgPath + cfgFile,
                  path=simulationPath)

    print('TESTing...')
    agent.load_actor_optimal(path='./DDPG2-UGV-Forward-Obstacle2-10000回合-73.94%/', file='Actor_ddpg')
    # DDPG2-UGV-Forward-Obstacle2-10000-65.86
    # cap = cv.VideoWriter(simulationPath + '/' + 'Optimal.mp4',
    #                      cv.VideoWriter_fourcc('X', 'V', 'I', 'D'),
    #                      120.0,
    #                      (env.width, env.height))
    # simulation_num = 500
    simulation_num = env.numData
    successCounter = 0
    timeOutCounter = 0
    collisionCounter = 0
    failNum = []
    for i in range(simulation_num):
        print('==========START==========')
        print('episode = ', i)
        # env.reset_random_with_database()
        env.reset_index_with_database(i)
        while not env.is_terminal:
            if cv.waitKey(1) == 27:
                break
            env.current_state = env.next_state.copy()
            action_from_actor = agent.choose_action(env.current_state, True)
            action = agent.action_linear_trans(action_from_actor)  # 将动作转换到实际范围上
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action)
            env.show_dynamic_imagewithobs(isWait=False)
            # cap.write(env.save)
            env.saveData(is2file=False)
        print('===========END===========')
        if env.terminal_flag == 1:
            print('咋也不咋地')
            pass
        if env.terminal_flag == 2:
            failNum.append(i)
            timeOutCounter += 1
            print('timeout')
        elif env.terminal_flag == 3:
            successCounter += 1
            print('success')
        elif env.terminal_flag == 4:
            failNum.append(i)
            collisionCounter += 1
            print('collision')
        elif env.terminal_flag == 5:
            failNum.append(i)
            print('out')
        else:
            print('没别的情况了')
    print('Total:', simulation_num, '  successful:', successCounter, '  timeout:', timeOutCounter, '  collision:', collisionCounter)
    print('Success rate:', round(successCounter / simulation_num, 3))
    print('Failure Num:', failNum)
    cv.waitKey(0)
    env.saveData(is2file=True, filepath=simulationPath)
