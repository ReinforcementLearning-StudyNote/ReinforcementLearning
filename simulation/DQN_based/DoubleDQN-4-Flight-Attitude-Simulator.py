import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

import copy
from environment.envs.flight_attitude_simulator import Flight_Attitude_Simulator as flight_sim
from algorithm.value_base.Double_DQN import Double_DQN
import datetime
import os
import torch
import numpy as np
import cv2 as cv

cfgPath = '../../environment/config/'
cfgFile = 'Flight_Attitude_Simulator.xml'
show_per = 1       # 每个回合显示一次


def fullFillReplayMemory_with_Optimal_Exploration(torch_pkl_file: str, randomEnv: bool, fullFillRatio: float, epsilon: float, is_only_success: bool):
    """
    :brief:                     Full-fill the replay memory with current optimal policy
    :param torch_pkl_file:      ****.pkl, the neural network file
    :param randomEnv:           Initialize environment randomly or not
    :param fullFillRatio:       Percentage to fill up the replay memory
    :param epsilon:             exploration probability
    :param is_only_success:     only data leads to a stable episode can be added into replay memory
    :return:
    """
    double_dqn.target_net.load_state_dict(torch.load(torch_pkl_file))
    double_dqn.eval_net.load_state_dict(torch.load(torch_pkl_file))
    env.reset_random() if randomEnv else env.reset()
    print('Collecting...')
    fullFillCount = int(fullFillRatio * double_dqn.memory_capacity)
    fullFillCount = max(min(fullFillCount, double_dqn.memory_capacity), double_dqn.batch_size)
    _new_transitions = []
    while double_dqn.replay_count < fullFillCount:
        env.reset_random() if randomEnv else env.reset()
        _new_transitions.clear()
        while not env.is_terminal:
            env.current_state = env.next_state.copy()       # 状态更新
            _numAction = double_dqn.get_action_with_fixed_epsilon(env.current_state, epsilon)
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(double_dqn.actionNUm2PhysicalAction(_numAction))
            env.show_dynamic_image(isWait=False)
            _boolend = not env.is_terminal
            _new_transition = list(np.hstack((env.current_state, env.current_action, env.reward, env.next_state, _boolend)))
            if not is_only_success:
                double_dqn.update_replay_memory(_new_transition)
                if double_dqn.replay_count % 100 == 0:
                    print('replay_count = ', double_dqn.replay_count)
            else:
                _new_transitions.append(_new_transition.copy())
        if is_only_success and env.terminal_flag == 3:
            double_dqn.update_replay_memory_per_episode(_new_transitions)
            print('replay_count = ', double_dqn.replay_count)


def fullFillReplayMemory_Random(randomEnv: bool, fullFillRatio: float):
    """
    :brief:                     Full fill the replay memory with random policy
    :param randomEnv:           Initialize environment randomly or not
    :param fullFillRatio:       Percentage to fill up the replay memory
    :return:                    None
    """
    print('Collecting...')
    fullFillCount = int(fullFillRatio * double_dqn.memory_capacity)
    fullFillCount = max(min(fullFillCount, double_dqn.memory_capacity), double_dqn.batch_size)
    while double_dqn.replay_count < fullFillCount:
        env.reset_random() if randomEnv else env.reset()
        while not env.is_terminal:
            if double_dqn.replay_count % 100 == 0:
                print('replay_count = ', double_dqn.replay_count)
            env.current_state = env.next_state.copy()       # 状态更新
            _numAction = double_dqn.get_action_random()
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(double_dqn.actionNUm2PhysicalAction(_numAction))
            env.show_dynamic_image(isWait=False)
            _boolend = not env.is_terminal
            _new_transition = list(np.hstack((env.current_state, env.current_action, env.reward, env.next_state, _boolend)))
            double_dqn.update_replay_memory(_new_transition)


if __name__ == '__main__':
    env = flight_sim(initTheta=-60.0, setTheta=0.0, save_cfg=False)
    double_dqn = Double_DQN(gamma=0.9,
                            epsilon=0.95,
                            learning_rate=5e-4,
                            memory_capacity=20000,     # 10000
                            batch_size=256,
                            target_replace_iter=200,
                            modelFileXML=cfgPath + cfgFile)
    # env.show_initial_image(isWait=True)
    c = cv.waitKey(1)
    simulationPath = '../../datasave/' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-Double-DQN-FlightAttitudeSimulator/'
    os.mkdir(simulationPath)
    TRAIN = True            # 直接训练
    RETRAIN = False         # 基于之前的训练结果重新训练
    TEST = False
    assert TRAIN ^ TEST     # 训练测试不可以同时进行

    if RETRAIN:
        print('Retraining')
        fullFillReplayMemory_with_Optimal_Exploration(torch_pkl_file='dqn_parameters_ok3.pkl',
                                                      randomEnv=True,
                                                      fullFillRatio=0.5,
                                                      epsilon=0.5,
                                                      is_only_success=True)
        # 如果注释掉，就是在上次的基础之上继续学习，如果不是就是重新学习，但是如果两次的奖励函数有变化，那么就必须执行这两句话
        '''生成初始数据之后要再次初始化网络'''
        # dqn.eval_net.init()
        # dqn.target_net.init()
        '''生成初始数据之后要再次初始化网络'''

    if TRAIN:
        double_dqn.DoubleDQN_info()
        # cv.waitKey(0)
        double_dqn.save_episode.append(double_dqn.episode)
        double_dqn.save_reward.append(0.0)
        double_dqn.save_epsilon.append(double_dqn.epsilon)
        MAX_EPISODE = 1500
        double_dqn.episode = 0  # 设置起始回合
        if not RETRAIN:
            '''fullFillReplayMemory_Random'''
            fullFillReplayMemory_Random(randomEnv=True, fullFillRatio=0.5)
            '''fullFillReplayMemory_Random'''
        print('Start to train...')
        new_transitions = []
        while double_dqn.episode <= MAX_EPISODE:
            # env.reset()
            env.reset_random()
            sumr = 0
            new_transitions.clear()
            while not env.is_terminal:
                c = cv.waitKey(1)
                env.current_state = env.next_state.copy()
                double_dqn.epsilon = double_dqn.get_epsilon()
                # double_dqn.epsilon = 0.4
                numAction = double_dqn.get_action_with_fixed_epsilon(env.current_state, double_dqn.epsilon)
                s, a, r, s_, env.is_terminal = env.step_update(double_dqn.actionNUm2PhysicalAction(numAction))     # 环境更新的action需要是物理的action
                env.current_state = copy.deepcopy(s)
                env.current_action = copy.deepcopy(a)
                env.reward = r
                env.next_state = copy.deepcopy(s_)
                if double_dqn.episode % show_per == 0:
                    env.show_dynamic_image(isWait=False)
                sumr = sumr + env.reward
                boolend = not env.is_terminal
                new_transition = list(np.hstack((s, a, r, s_, boolend)))
                new_transitions.append(new_transition.copy())
                # dqn.update_replay_memory(new_transition)
                double_dqn.nn_training(saveNNPath=simulationPath)
            '''跳出循环代表回合结束'''
            if env.terminal_flag == 3:
                print('Update Replay Memory......')
                double_dqn.update_replay_memory_per_episode(new_transitions)
            '''跳出循环代表回合结束'''
            print(
                '=========START=========',
                'Episode:', double_dqn.episode,
                'Epsilon', double_dqn.epsilon,
                'Cumulative reward:', round(sumr, 3),
                '==========END=========')
            print()
            double_dqn.saveData_EpisodeRewardEpsilon(double_dqn.episode, sumr, double_dqn.epsilon)
            double_dqn.episode += 1
            if c == 27:
                print('Over......')
                break
        '''dataSave'''
        double_dqn.saveData_EpisodeRewardEpsilon(0.0, 0.0, 0.0, True, 'EpisodeRewardEpsilon.csv', simulationPath)
        double_dqn.saveData_StepTDErrorNNLose(0.0, 0.0, 0.0, True, 'StepTDErrorNNLose.csv', simulationPath)
        '''dataSave'''

    if TEST:
        print('TESTing...')
        double_dqn.get_optimalfrompkl('dqn_parameters_ok3.pkl')
        cap = cv.VideoWriter(simulationPath + '/' + 'Optimal.mp4',
                             cv.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                             120.0,
                             (env.width, env.height))
        simulation_num = 100
        for i in range(simulation_num):
            print('测试数 = ', i)
            env.reset_random()
            while not env.is_terminal:
                if cv.waitKey(1) == 27:
                    break
                cap.write(env.show)
                env.current_state = env.next_state.copy()
                numAction = double_dqn.get_action_with_fixed_epsilon(env.current_state, 0.0)
                # numAction = dqn.get_action_optimal_in_DQN(env.current_state)
                env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(double_dqn.actionNUm2PhysicalAction(numAction))
                env.show_dynamic_image(isWait=False)
                env.saveData(is2file=False)
        cv.waitKey(0)
        env.saveData(is2file=True, filepath=simulationPath)
