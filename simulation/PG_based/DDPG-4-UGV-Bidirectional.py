import math
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
# import copy
from environment.envs.ugv_bidirectional_continuous import UGV_Bidirectional_Continuous
from algorithm.actor_critic.DDPG import DDPG
import cv2 as cv
from common.common import *
import datetime

cfgPath = '../../environment/config/'
cfgFile = 'Two_Wheel_UGV.xml'
optPath = '../../datasave/network/'
show_per = 1


def fullFillReplayMemory_with_Optimal(randomEnv: bool,
                                      fullFillRatio: float,
                                      is_only_success: bool):
    print('Retraining...')
    print('Collecting...')
    # ddpg.load_models()
    ddpg.load_models(path='./DDPG-Two-Wheel-UGV还不错哦/')
    fullFillCount = int(fullFillRatio * ddpg.memory.mem_size)
    fullFillCount = max(min(fullFillCount, ddpg.memory.mem_size), ddpg.memory.batch_size)
    _new_state, _new_action, _new_reward, _new_state_, _new_done = [], [], [], [], []
    while ddpg.memory.mem_counter < fullFillCount:
        env.reset_random() if randomEnv else env.reset()
        _new_state.clear()
        _new_action.clear()
        _new_reward.clear()
        _new_state_.clear()
        _new_done.clear()
        while not env.is_terminal:
            env.current_state = env.next_state.copy()  # 状态更新
            _action_from_actor = ddpg.choose_action(env.current_state, is_optimal=False, sigma=1 / 2)
            _action = ddpg.action_linear_trans(_action_from_actor)
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(_action)
            env.show_dynamic_image(isWait=False)
            if is_only_success:
                _new_state.append(env.current_state)
                _new_action.append(env.current_action)
                _new_reward.append(env.reward)
                _new_state_.append(env.next_state)
                _new_done.append(1.0 if env.is_terminal else 0.0)
            else:
                if ddpg.memory.mem_counter % 100 == 0 and ddpg.memory.mem_counter > 0:
                    print('replay_count = ', ddpg.memory.mem_counter)
                ddpg.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
        if is_only_success:
            if env.terminal_flag == 3 or env.terminal_flag == 2:
                print('Update Replay Memory......')
                print('replay_count = ', ddpg.memory.mem_counter)
                ddpg.memory.store_transition_per_episode(_new_state, _new_action, _new_reward, _new_state_, _new_done)
    ddpg.memory.get_reward_sort()


def fullFillReplayMemory_Random(randomEnv: bool, fullFillRatio: float, is_only_success: bool):
    """
    :param randomEnv:           init env randomly
    :param fullFillRatio:       the ratio
    :param is_only_success:
    :return:
    """
    print('Collecting...')
    fullFillCount = int(fullFillRatio * ddpg.memory.mem_size)
    fullFillCount = max(min(fullFillCount, ddpg.memory.mem_size), ddpg.memory.batch_size)
    _new_state, _new_action, _new_reward, _new_state_, _new_done = [], [], [], [], []
    while ddpg.memory.mem_counter < fullFillCount:
        env.reset_random() if randomEnv else env.reset()
        _new_state.clear()
        _new_action.clear()
        _new_reward.clear()
        _new_state_.clear()
        _new_done.clear()
        while not env.is_terminal:
            if ddpg.memory.mem_counter % 100 == 0 and ddpg.memory.mem_counter > 0:
                print('replay_count = ', ddpg.memory.mem_counter)
            env.current_state = env.next_state.copy()  # 状态更新
            _action_from_actor = ddpg.choose_action(env.current_state, False, sigma=1.0)
            # _action_from_actor = ddpg.choose_action_random()
            _action = ddpg.action_linear_trans(_action_from_actor)
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(_action)
            env.show_dynamic_image(isWait=False)
            if is_only_success:
                _new_state.append(env.current_state)
                _new_action.append(env.current_action)
                _new_reward.append(env.reward)
                _new_state_.append(env.next_state)
                _new_done.append(1.0 if env.is_terminal else 0.0)
            else:
                ddpg.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
        if is_only_success:
            if env.terminal_flag == 3 or env.terminal_flag == 2:
                print('Update Replay Memory......')
                ddpg.memory.store_transition_per_episode(_new_state, _new_action, _new_reward, _new_state_, _new_done)
                print('replay_count = ', ddpg.memory.mem_counter)
    ddpg.memory.get_reward_sort()


if __name__ == '__main__':
    simulationPath = '../../datasave/log/' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-DDPG-Two-Wheel-UGV/'
    os.mkdir(simulationPath)

    env = UGV_Bidirectional_Continuous(initPhi=deg2rad(0),
                                       save_cfg=False,
                                       x_size=4.0,
                                       y_size=4.0,
                                       start=[2.0, 2.0],
                                       terminal=[4.0, 4.0])

    ddpg = DDPG(gamma=0.9,
                actor_learning_rate=1e-4,
                critic_learning_rate=1e-3,
                actor_soft_update=1e-2,
                critic_soft_update=1e-2,
                memory_capacity=24000,
                batch_size=256,
                modelFileXML=cfgPath + cfgFile,
                path=simulationPath)

    c = cv.waitKey(1)
    TRAIN = False  # 直接训练
    RETRAIN = False  # 基于之前的训练结果重新训练
    TEST = not TRAIN
    is_storage_only_success = True

    if RETRAIN:
        print('Retraining')
        fullFillReplayMemory_with_Optimal(randomEnv=True,
                                          fullFillRatio=0.5,
                                          is_only_success=True)
        # 如果注释掉，就是在上次的基础之上继续学习，如果不是就是重新学习，但是如果两次的奖励函数有变化，那么就必须执行这两句话
        '''生成初始数据之后要再次初始化网络'''
        ddpg.actor.initialization()
        ddpg.target_actor.initialization()
        ddpg.critic.initialization()
        ddpg.target_critic.initialization()
        '''生成初始数据之后要再次初始化网络'''

    if TRAIN:
        # random.seed(23)
        ddpg.DDPG_info()
        successCounter = 0
        timeOutCounter = 0
        # cv.waitKey(0)
        ddpg.save_episode.append(ddpg.episode)
        ddpg.save_reward.append(0.0)
        MAX_EPISODE = 5000
        if not RETRAIN:
            '''fullFillReplayMemory_Random'''
            fullFillReplayMemory_Random(randomEnv=True, fullFillRatio=0.5, is_only_success=True)
            '''fullFillReplayMemory_Random'''
        print('Start to train...')
        new_state, new_action, new_reward, new_state_, new_done = [], [], [], [], []
        while ddpg.episode <= MAX_EPISODE:
            # env.reset()
            print('=========START=========')
            print('Episode:', ddpg.episode)
            env.reset_random()
            sumr = 0
            new_state.clear()
            new_action.clear()
            new_reward.clear()
            new_state_.clear()
            new_done.clear()
            while not env.is_terminal:
                c = cv.waitKey(1)
                env.current_state = env.next_state.copy()
                epsilon = random.uniform(0, 1)
                if epsilon < 0.2:
                    # print('...random...')
                    action_from_actor = ddpg.choose_action_random()  # 有一定探索概率完全随机探索
                else:
                    action_from_actor = ddpg.choose_action(env.current_state, False, sigma=1 / 2)  # 剩下的是神经网络加噪声
                action = ddpg.action_linear_trans(action_from_actor)  # 将动作转换到实际范围上
                env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = \
                    env.step_update(action)  # 环境更新的action需要是物理的action
                if ddpg.episode % show_per == 0:
                    env.show_dynamic_image(isWait=False)
                sumr = sumr + env.reward
                if is_storage_only_success:
                    new_state.append(env.current_state)
                    new_action.append(env.current_action)
                    new_reward.append(env.reward)
                    new_state_.append(env.next_state)
                    new_done.append(1.0 if env.is_terminal else 0.0)
                else:
                    ddpg.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
                ddpg.learn(is_reward_ascent=False)
            # cv.destroyAllWindows()
            '''跳出循环代表回合结束'''
            if env.terminal_flag == 3:
                successCounter += 1
            if env.terminal_flag == 2:
                timeOutCounter += 1
            if is_storage_only_success:  # 只用超时的或者成功的训练
                if (env.terminal_flag == 3) or (env.terminal_flag == 2):
                    print('Update Replay Memory......')
                    ddpg.memory.store_transition_per_episode(new_state, new_action, new_reward, new_state_, new_done)
                    # ddpg.memory.get_reward_resort(per=5)
            '''跳出循环代表回合结束'''
            print('Cumulative reward:', round(sumr, 3))
            print('共：', ddpg.episode, ' 回合，成功', successCounter, ' 回合，超时', timeOutCounter, ' 回合')
            print('==========END=========')
            print()
            ddpg.saveData_EpisodeReward(ddpg.episode, sumr)
            ddpg.episode += 1
            if ddpg.episode % 10 == 0:
                ddpg.save_models()
            if c == 27:
                print('Over......')
                break
        '''dataSave'''
        ddpg.saveData_EpisodeReward(0.0, 0.0, True, 'EpisodeReward.csv', simulationPath)
        '''dataSave'''

    if TEST:
        print('TESTing...')
        ddpg.load_actor_optimal(path=optPath, file='ddpg-4-ugv-bidirectional')
        # ddpg.load_models()
        cap = cv.VideoWriter(simulationPath + '/' + 'Optimal.mp4',
                             cv.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                             120.0,
                             (env.width, env.height))
        simulation_num = 10
        successTotal = []
        timeOutTotal = []
        x_sep = 3
        y_sep = 3
        for partIndex_X in range(x_sep):
            for partIndex_Y in range(y_sep):
                successCounter = 0
                timeOutCounter = 0
                for i in range(simulation_num):
                    print('==========START==========')
                    print('episode = ', i)
                    env.reset_random()
                    '''Seperate the map into nine parts'''
                    part = [0 + partIndex_X * env.x_size / x_sep, 0 + (partIndex_X + 1) * env.x_size / x_sep,
                            0 + partIndex_Y * env.y_size / y_sep, 0 + (partIndex_Y + 1) * env.y_size / y_sep]
                    env.set_terminal([random.uniform(part[0], part[1]), random.uniform(part[2], part[3])])
                    '''Seperate the map into nine parts'''
                    while not env.is_terminal:
                        if cv.waitKey(1) == 27:
                            break
                        env.current_state = env.next_state.copy()
                        action_from_actor = ddpg.choose_action(env.current_state, True)
                        action = ddpg.action_linear_trans(action_from_actor)  # 将动作转换到实际范围上
                        env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action)
                        env.show_dynamic_image(isWait=False)
                        cap.write(env.save)
                        env.saveData(is2file=False)
                    print('===========END===========')
                    if env.terminal_flag == 2:
                        timeOutCounter += 1
                    if env.terminal_flag == 3:
                        successCounter += 1
                print('Total:', simulation_num, '  successful:', successCounter, '  timeout:', timeOutCounter)
                successTotal.append(successCounter)
                timeOutTotal.append(timeOutCounter)
        cv.waitKey(0)
        print(successTotal)
        print(timeOutTotal)
        env.saveData(is2file=True, filepath=simulationPath)