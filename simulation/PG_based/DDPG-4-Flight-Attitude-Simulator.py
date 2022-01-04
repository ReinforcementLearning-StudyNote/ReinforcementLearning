import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
import copy
from environment.envs.flight_attitude_simulator_continuous import Flight_Attitude_Simulator_Continuous as flight_sim_con
from algorithm.actor_critic.DDPG import DDPG
import cv2 as cv

cfgPath = '../../environment/config/'
cfgFile = 'Flight_Attitude_Simulator_Continuous.xml'
optPath = '../../datasave/network/'
show_per = 1
simulationPath = '../../datasave/log/ddpg'


def fullFillReplayMemory_with_Optimal(torch_pkl_file: str,
                                      randomEnv: bool,
                                      fullFillRatio: float,
                                      is_only_success: bool):
    pass


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
            _action_from_actor = ddpg.choose_action(env.current_state)
            _action = ddpg.action_linear_trans(_action_from_actor)
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(_action)
            env.show_dynamic_image(isWait=False)
            ddpg.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)


if __name__ == '__main__':
    env = flight_sim_con(initTheta=-60.0, setTheta=0.0, save_cfg=False)
    ddpg = DDPG(gamma=0.9,
                actor_learning_rate=2.5e-5,
                critic_learning_rate=2.5e-4,
                actor_soft_update=1e-3,
                critic_soft_update=1e-3,
                memory_capacity=20000,  # 10000
                batch_size=256,
                modelFileXML=cfgPath + cfgFile)
    c = cv.waitKey(1)
    TRAIN = True  # 直接训练
    RETRAIN = False  # 基于之前的训练结果重新训练
    TEST = False
    is_storage_only_success = True
    assert TRAIN ^ TEST  # 训练测试不可以同时进行

    if RETRAIN:
        print('Retraining')
        fullFillReplayMemory_with_Optimal(torch_pkl_file='dqn_parameters_ok3.pkl',
                                          randomEnv=True,
                                          fullFillRatio=0.5,
                                          is_only_success=True)
        # 如果注释掉，就是在上次的基础之上继续学习，如果不是就是重新学习，但是如果两次的奖励函数有变化，那么就必须执行这两句话
        '''生成初始数据之后要再次初始化网络'''
        # dqn.eval_net.init()
        # dqn.target_net.init()
        '''生成初始数据之后要再次初始化网络'''

    if TRAIN:
        ddpg.DDPG_info()
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
                action_from_actor = ddpg.choose_action(env.current_state)
                action = ddpg.action_linear_trans(action_from_actor)       # 将动作转换到实际范围上
                s, a, r, s_, env.is_terminal = env.step_update(action)  # 环境更新的action需要是物理的action
                env.current_state = copy.deepcopy(s)
                env.current_action = copy.deepcopy(a)
                env.reward = r
                env.next_state = copy.deepcopy(s_)
                if ddpg.episode % show_per == 0:
                    env.show_dynamic_image(isWait=False)
                sumr = sumr + env.reward
                ddpg.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
                if is_storage_only_success:
                    new_state.append(env.current_state)
                    new_action.append(env.current_action)
                    new_reward.append(env.reward)
                    new_state_.append(env.next_state)
                    new_done.append(1.0 if env.is_terminal else 0.0)
                ddpg.learn()
            '''跳出循环代表回合结束'''
            if is_storage_only_success:
                if env.terminal_flag == 3:
                    print('Update Replay Memory......')
                    ddpg.memory.store_transition_per_episode(new_state, new_action, new_reward, new_state_, new_done)
            '''跳出循环代表回合结束'''
            print(
                '=========START=========',
                'Episode:', ddpg.episode,
                'Cumulative reward:', round(sumr, 3),
                '==========END=========')
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
