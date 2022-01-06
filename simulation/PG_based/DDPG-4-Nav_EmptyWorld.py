import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
# import copy
from environment.envs.pathplanning.nav_emptyworld_continuous import Nav_EmptyWorld
from algorithm.actor_critic.DDPG import DDPG
import cv2 as cv
# from common.common import *
import datetime

cfgPath = '../../environment/config/'
cfgFile = 'Nav_EmptyWorld_Continuous.xml'
optPath = '../../datasave/network/'
show_per = 1


def fullFillReplayMemory_with_Optimal(randomEnv: bool,
                                      fullFillRatio: float,
                                      is_only_success: bool):
    print('Retraining...')
    print('Collecting...')
    ddpg.load_models()
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
            if ddpg.memory.mem_counter % 100 == 0:
                print('replay_count = ', ddpg.memory.mem_counter)
            env.current_state = env.next_state.copy()  # 状态更新
            _action_from_actor = ddpg.choose_action(env.current_state, is_optimal=True)
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
            if env.terminal_flag == 3:
                print('Update Replay Memory......')
                ddpg.memory.store_transition_per_episode(_new_state, _new_action, _new_reward, _new_state_, _new_done)


def fullFillReplayMemory_Random(randomEnv: bool, fullFillRatio: float):
    """
    :param randomEnv:       init env randomly
    :param fullFillRatio:   the ratio
    :return:                None
    """
    print('Collecting...')
    fullFillCount = int(fullFillRatio * ddpg.memory.mem_size)
    fullFillCount = max(min(fullFillCount, ddpg.memory.mem_size), ddpg.memory.batch_size)
    while ddpg.memory.mem_counter < fullFillCount:
        env.reset_random() if randomEnv else env.reset()
        while not env.is_terminal:
            if ddpg.memory.mem_counter % 100 == 0:
                print('replay_count = ', ddpg.memory.mem_counter)
            env.current_state = env.next_state.copy()  # 状态更新
            _action_from_actor = ddpg.choose_action(env.current_state, False)
            _action = ddpg.action_linear_trans(_action_from_actor)
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(_action)
            env.show_dynamic_image(isWait=False)
            ddpg.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)


if __name__ == '__main__':
    simulationPath = '../../datasave/log/' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-DDPG-Nav_EmptyWorld/'
    os.mkdir(simulationPath)

    samplingMap_dict = {'width': 500,
                        'height': 500,
                        'x_size': 10,
                        'y_size': 10,
                        'image_name': '2D_Nav_EmptyWorld',
                        'start': [5, 5],
                        'terminal': [9, 9],
                        'obs': [],
                        'draw': True}
    env = Nav_EmptyWorld(samplingMap_dict=samplingMap_dict, vRange=[-3, 3], aRange=[-3, 3], jRange=[-5, 5], save_cfg=False)
    # initial position is [5, 5], terminal position is [9, 9], and it doesn't matter where the initial and terminal position are.
    # because the environment initialize itself stochasticly.
    # The initial velocity, acceleration, and jerk should be zero. The velocity range is [-3, 3], acceleration range is [-3. 3], and jerk range is [-5, 5]
    ddpg = DDPG(gamma=0.9,
                actor_learning_rate=1e-4,
                critic_learning_rate=1e-3,
                actor_soft_update=1e-2,
                critic_soft_update=1e-2,
                memory_capacity=100000,
                batch_size=512,
                modelFileXML=cfgPath + cfgFile,
                path=simulationPath)

    c = cv.waitKey(1)
    TRAIN = True  # 直接训练
    RETRAIN = False  # 基于之前的训练结果重新训练
    TEST = not TRAIN
    is_storage_only_success = False
    assert TRAIN ^ TEST  # 训练测试不可以同时进行

    if RETRAIN:
        print('Retraining')
        fullFillReplayMemory_with_Optimal(randomEnv=True,
                                          fullFillRatio=0.5,
                                          is_only_success=True)
        # 如果注释掉，就是在上次的基础之上继续学习，如果不是就是重新学习，但是如果两次的奖励函数有变化，那么就必须执行这两句话
        '''生成初始数据之后要再次初始化网络'''
        # ddpg.actor.initialization()
        # ddpg.target_actor.initialization()
        # ddpg.critic.initialization()
        # ddpg.target_critic.initialization()
        '''生成初始数据之后要再次初始化网络'''

    if TRAIN:
        ddpg.DDPG_info()
        successCounter = 0
        timeOutCounter = 0
        # cv.waitKey(0)
        ddpg.save_episode.append(ddpg.episode)
        ddpg.save_reward.append(0.0)
        MAX_EPISODE = 1500
        if not RETRAIN:
            '''fullFillReplayMemory_Random'''
            fullFillReplayMemory_Random(randomEnv=True, fullFillRatio=0.5)
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
                action_from_actor = ddpg.choose_action(env.current_state, False)
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
                ddpg.learn()
            '''跳出循环代表回合结束'''
            if env.terminal_flag == 3:
                successCounter += 1
            if env.terminal_flag == 2:
                timeOutCounter += 1
            if is_storage_only_success:  # 只用超时的或者成功的训练
                if (env.terminal_flag == 3) or (env.terminal_flag == 2):
                    print('Update Replay Memory......')
                    ddpg.memory.store_transition_per_episode(new_state, new_action, new_reward, new_state_, new_done)
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
        ddpg.actor_load_optimal(path='../../datasave/network/', file='ddpg-4-nav-empty-world')
        # ddpg.load_models()
        cap = cv.VideoWriter(simulationPath + '/' + 'Optimal.mp4',
                             cv.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                             120.0,
                             (env.width, env.height))
        simulation_num = 100
        for i in range(simulation_num):
            print('==========START==========')
            print('episode = ', i)
            env.reset_random()
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
        cv.waitKey(0)
        env.saveData(is2file=True, filepath=simulationPath)