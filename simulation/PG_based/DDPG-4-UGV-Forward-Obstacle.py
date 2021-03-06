import math
import os
import sys
import datetime
import time
import cv2 as cv
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
# import copy
from environment.envs.UGV.ugv_forward_obstacle_continuous import UGV_Forward_Obstacle_Continuous
from algorithm.actor_critic.DDPG import DDPG
from common.common import *

cfgPath = '../../environment/config/'
cfgFile = 'UGV_Forward_Obstacle_Continuous.xml'
optPath = '../../datasave/network/'
dataBasePath1 = '../../environment/envs/pathplanning/11X11-AllCircle1/'
dataBasePath2 = '../../environment/envs/pathplanning/5X5-AllCircle2/'
dataBasePath3 = '../../environment/envs/pathplanning/5X5-AllCircle3/'
dataBasePath4 = '../../environment/envs/pathplanning/5X5-AllCircle4/'
show_per = 1


class CriticNetWork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, name, chkpt_dir):
        super(CriticNetWork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

        self.fc1 = nn.Linear(self.state_dim, 128)  # state -> hidden1
        self.batch_norm1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 64)  # hidden1 -> hidden2
        self.batch_norm2 = nn.LayerNorm(64)

        self.action_value = nn.Linear(self.action_dim, 64)  # action -> hidden2
        self.q = nn.Linear(64, 1)  # hidden2 -> output action value

        # self.initialization()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, _action):
        state_value = self.fc1(state)  # forward
        state_value = self.batch_norm1(state_value)  # batch normalization
        state_value = func.relu(state_value)  # relu

        state_value = self.fc2(state_value)
        state_value = self.batch_norm2(state_value)

        action_value = func.relu(self.action_value(_action))
        state_action_value = func.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def initialization_default(self):
        self.fc1.reset_parameters()
        self.batch_norm1.reset_parameters()
        self.fc2.reset_parameters()
        self.batch_norm2.reset_parameters()

        self.action_value.reset_parameters()
        self.q.reset_parameters()

    def initialization(self):
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3 = 0.003
        nn.init.uniform_(self.q.weight.data, -f3, f3)
        nn.init.uniform_(self.q.bias.data, -f3, f3)

    def save_checkpoint(self, name=None, path='', num=None):
        print('...saving checkpoint...')
        if name is None:
            torch.save(self.state_dict(), self.checkpoint_file)
        else:
            if num is None:
                torch.save(self.state_dict(), path + name)
            else:
                torch.save(self.state_dict(), path + name + str(num))

    def save_all_net(self):
        print('...saving all net...')
        torch.save(self, self.checkpoint_file_whole_net)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim1, state_dim2, action_dim, name, chkpt_dir):
        super(ActorNetwork, self).__init__()
        self.state_dim1 = state_dim1
        self.state_dim2 = state_dim2
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

        self.linear11 = nn.Linear(self.state_dim1, 128)  # ???????????????????????????
        self.batch_norm11 = nn.LayerNorm(128)
        self.linear12 = nn.Linear(128, 64)  # ???????????????????????????
        self.batch_norm12 = nn.LayerNorm(64)
        self.linear13 = nn.Linear(64, 64)
        self.batch_norm13 = nn.LayerNorm(64)  # ???????????????????????????

        self.linear21 = nn.Linear(self.state_dim2, 128)  # ???????????????????????????
        self.batch_norm21 = nn.LayerNorm(128)
        self.linear22 = nn.Linear(128, 64)  # ???????????????????????????
        self.batch_norm22 = nn.LayerNorm(64)
        self.linear23 = nn.Linear(64, 32)
        self.batch_norm23 = nn.LayerNorm(32)  # ???????????????????????????

        self.mu = nn.Linear(64 + 32, self.action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def initialization_default(self):
        self.linear11.reset_parameters()
        self.batch_norm11.reset_parameters()
        self.linear12.reset_parameters()
        self.batch_norm12.reset_parameters()
        self.linear13.reset_parameters()
        self.batch_norm13.reset_parameters()

        self.linear21.reset_parameters()
        self.batch_norm21.reset_parameters()
        self.linear22.reset_parameters()
        self.batch_norm22.reset_parameters()
        self.linear23.reset_parameters()
        self.batch_norm23.reset_parameters()

        # self.combine.reset_parameters()
        self.mu.reset_parameters()

    def forward(self, state):
        """
        :param state:
        :return:            output of the net
        """
        if state.dim() == 1:
            split_state = torch.split(state, [self.state_dim1, self.state_dim2], dim=0)
        else:
            split_state = torch.split(state, [self.state_dim1, self.state_dim2], dim=1)
        x1 = self.linear11(split_state[0])
        x1 = self.batch_norm11(x1)
        x1 = func.relu(x1)

        x1 = self.linear12(x1)
        x1 = self.batch_norm12(x1)
        x1 = func.relu(x1)

        x1 = self.linear13(x1)
        x1 = self.batch_norm13(x1)
        x1 = func.relu(x1)  # ????????????

        x2 = self.linear21(split_state[1])
        x2 = self.batch_norm21(x2)
        x2 = func.relu(x2)

        x2 = self.linear22(x2)
        x2 = self.batch_norm22(x2)
        x2 = func.relu(x2)

        x2 = self.linear23(x2)
        x2 = self.batch_norm23(x2)
        x2 = func.relu(x2)  # ????????????

        x = torch.cat((x1, x2)) if x1.dim() == 1 else torch.cat((x1, x2), dim=1)
        # print(x1.size(), x2.size(), x.size())
        # x = self.combine(x)
        # x = func.relu(x)

        x = torch.tanh(self.mu(x))
        return x

    def save_checkpoint(self, name=None, path='', num=None):
        print('...saving checkpoint...')
        if name is None:
            torch.save(self.state_dict(), self.checkpoint_file)
        else:
            if num is None:
                torch.save(self.state_dict(), path + name)
            else:
                torch.save(self.state_dict(), path + name + str(num))

    def save_all_net(self):
        print('...saving all net...')
        torch.save(self, self.checkpoint_file_whole_net)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


def fullFillReplayMemory_with_Optimal(randomEnv: bool,
                                      fullFillRatio: float,
                                      is_only_success: bool):
    print('Retraining...')
    print('Collecting...')
    agent.load_models(path='./500_0.716_save-?????????/')
    fullFillCount = int(fullFillRatio * agent.memory.mem_size)
    fullFillCount = max(min(fullFillCount, agent.memory.mem_size), agent.memory.batch_size)
    _new_state, _new_action, _new_reward, _new_state_, _new_done = [], [], [], [], []
    while agent.memory.mem_counter < fullFillCount:
        env.reset_random_with_database() if randomEnv else env.reset()
        _new_state.clear()
        _new_action.clear()
        _new_reward.clear()
        _new_state_.clear()
        _new_done.clear()
        while not env.is_terminal:
            env.current_state = env.next_state.copy()  # ????????????
            if random.uniform(0, 1) < 0.2:
                _action_from_actor = agent.choose_action_random()
            else:
                _action_from_actor = agent.choose_action(env.current_state, is_optimal=False, sigma=1 / 3)
            _action = agent.action_linear_trans(_action_from_actor)
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(_action)
            env.show_dynamic_imagewithobs(isWait=False)
            if is_only_success:
                _new_state.append(env.current_state)
                _new_action.append(env.current_action)
                _new_reward.append(env.reward)
                _new_state_.append(env.next_state)
                _new_done.append(1.0 if env.is_terminal else 0.0)
            else:
                if agent.memory.mem_counter % 100 == 0 and agent.memory.mem_counter > 0:
                    print('replay_count = ', agent.memory.mem_counter)
                '''????????????????????????????????????????????????[s a r s' done]?????????????????????'''
                # if env.reward >= -4.5:
                if True:
                    agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
        if is_only_success:
            if env.terminal_flag == 3 or env.terminal_flag == 2:
                print('Update Replay Memory......')
                print('replay_count = ', agent.memory.mem_counter)
                agent.memory.store_transition_per_episode(_new_state, _new_action, _new_reward, _new_state_, _new_done)
    agent.memory.get_reward_sort()


def fullFillReplayMemory_Random(randomEnv: bool, fullFillRatio: float, is_only_success: bool):
    """
    :param randomEnv:           init env randomly
    :param fullFillRatio:       the ratio
    :param is_only_success:
    :return:
    """
    print('Collecting...')
    fullFillCount = int(fullFillRatio * agent.memory.mem_size)
    fullFillCount = max(min(fullFillCount, agent.memory.mem_size), agent.memory.batch_size)
    _new_state, _new_action, _new_reward, _new_state_, _new_done = [], [], [], [], []
    while agent.memory.mem_counter < fullFillCount:
        env.reset_random_with_database() if randomEnv else env.reset()
        # env.reset_random() if randomEnv else env.reset()
        _new_state.clear()
        _new_action.clear()
        _new_reward.clear()
        _new_state_.clear()
        _new_done.clear()
        while not env.is_terminal:
            env.current_state = env.next_state.copy()  # ????????????
            # _action_from_actor = agent.choose_action(env.current_state, False, sigma=1/2)
            _action_from_actor = agent.choose_action_random()
            _action = agent.action_linear_trans(_action_from_actor)
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(_action)
            env.show_dynamic_imagewithobs(isWait=False)
            if is_only_success:
                _new_state.append(env.current_state)
                _new_action.append(env.current_action)
                _new_reward.append(env.reward)
                _new_state_.append(env.next_state)
                _new_done.append(1.0 if env.is_terminal else 0.0)
            else:
                if agent.memory.mem_counter % 100 == 0 and agent.memory.mem_counter > 0:
                    print('replay_count = ', agent.memory.mem_counter)
                '''????????????????????????????????????????????????[s a r s' done]?????????????????????'''
                # if (env.reward >= -3) or (env.reward <= -10):
                # if env.reward >= -3.5:
                if True:
                    agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
        if is_only_success:
            if env.terminal_flag == 3 or env.terminal_flag == 2:
                print('Update Replay Memory......')
                agent.memory.store_transition_per_episode(_new_state, _new_action, _new_reward, _new_state_, _new_done)
                print('replay_count = ', agent.memory.mem_counter)
    agent.memory.get_reward_sort()


if __name__ == '__main__':
    simulationPath = '../../datasave/log/' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-DDPG-UGV-Forward-Obstacle/'
    os.mkdir(simulationPath)
    c = cv.waitKey(1)
    TRAIN = False  # ????????????
    RETRAIN = False  # ???????????????????????????????????????
    TEST = not TRAIN
    is_storage_only_success = False

    env = UGV_Forward_Obstacle_Continuous(initPhi=0.,
                                          save_cfg=False,
                                          x_size=11.0,
                                          y_size=11.0,
                                          start=[2.5, 2.5],
                                          terminal=[4.5, 4.5],
                                          dataBasePath=dataBasePath1)
    if TRAIN:
        agent = DDPG(gamma=0.99,
                     actor_soft_update=1e-2,
                     critic_soft_update=1e-2,
                     memory_capacity=60000,  # 100000
                     batch_size=512,  # 1024
                     modelFileXML=cfgPath + cfgFile,
                     path=simulationPath)
        '''????????????actor???critic????????????????????????????????????'''
        agent.actor = ActorNetwork(1e-4, 8, agent.state_dim_nn - 8, agent.action_dim_nn, 'Actor', simulationPath)
        agent.target_actor = ActorNetwork(1e-4, 8, agent.state_dim_nn - 8, agent.action_dim_nn, 'TargetActor', simulationPath)
        agent.critic = CriticNetWork(1e-3, agent.state_dim_nn, agent.action_dim_nn, 'Critic', simulationPath)
        agent.target_critic = CriticNetWork(1e-3, agent.state_dim_nn, agent.action_dim_nn, 'TargetCritic', simulationPath)
        '''????????????actor???critic????????????????????????????????????'''
        agent.DDPG_info()
        successCounter = 0
        timeOutCounter = 0
        collisionCounter = 0
        # cv.waitKey(0)
        MAX_EPISODE = 20000
        if RETRAIN:
            print('Retraining')
            fullFillReplayMemory_with_Optimal(randomEnv=True, fullFillRatio=0.5, is_only_success=is_storage_only_success)
            # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
            '''????????????????????????????????????????????????'''
            # agent.actor.initialization_default()
            # agent.target_actor.initialization_default()
            # agent.critic.initialization_default()
            # agent.target_critic.initialization_default()
            '''????????????????????????????????????????????????'''
        else:
            '''fullFillReplayMemory_Random'''
            fullFillReplayMemory_Random(randomEnv=True, fullFillRatio=0.25, is_only_success=is_storage_only_success)
            '''fullFillReplayMemory_Random'''
        print('Start to train...')
        new_state, new_action, new_reward, new_state_, new_done = [], [], [], [], []
        step = 0
        while agent.episode <= MAX_EPISODE:
            # env.reset()
            print('=========START=========')
            print('Episode:', agent.episode)
            env.reset_random_with_database()
            # env.reset_random()
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
                if epsilon < 0.15:
                    action_from_actor = agent.choose_action_random()  # ???????????????????????????????????????
                else:
                    action_from_actor = agent.choose_action(env.current_state, False, sigma=1 / 4)  # ?????????????????????????????????
                action = agent.action_linear_trans(action_from_actor)  # ?????????????????????????????????
                env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action)  # ???????????????action??????????????????action
                agent.saveData_Step_Reward(step=step, reward=env.reward, is2file=False, filename='StepReward.csv')
                step += 1
                if agent.episode % show_per == 0:
                    env.show_dynamic_imagewithobs(isWait=False)
                sumr = sumr + env.reward
                if is_storage_only_success:
                    new_state.append(env.current_state)
                    new_action.append(env.current_action)
                    new_reward.append(env.reward)
                    new_state_.append(env.next_state)
                    new_done.append(1.0 if env.is_terminal else 0.0)
                else:
                    '''????????????????????????????????????????????????[s a r s' done]?????????????????????'''
                    # if (env.reward >= -3) or (env.reward <= -10):
                    # if env.reward >= -3.5:
                    if True:
                        agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
                agent.learn(is_reward_ascent=False)
            # agent.memory.get_reward_resort(per=10)
            '''??????????????????????????????'''
            if env.terminal_flag == 4:
                collisionCounter += 1
            if env.terminal_flag == 3:
                successCounter += 1
            if env.terminal_flag == 2:
                timeOutCounter += 1
            if is_storage_only_success:  # ????????????????????????????????????
                if (env.terminal_flag == 3) or (env.terminal_flag == 2):
                    print('Update Replay Memory......')
                    agent.memory.store_transition_per_episode(new_state, new_action, new_reward, new_state_, new_done)
                    # agent.memory.get_reward_resort(per=5)
            '''??????????????????????????????'''
            print('Cumulative reward:', round(sumr, 3))
            print('??????', agent.episode, ' ???????????????', successCounter, ' ???????????????', timeOutCounter, ' ???????????????', collisionCounter, '??????')
            if agent.episode > 0:
                print('???????????????', round(successCounter / agent.episode * 100, 3), '%')
            print('==========END=========')
            print()
            agent.saveData_EpisodeReward(episode=agent.episode,
                                         time=env.time,
                                         reward=sumr,
                                         average_reward=sumr / env.time,
                                         successrate=round(successCounter / max(agent.episode, 1), 3),
                                         is2file=False, filename='EpisodeReward.csv')
            agent.episode += 1
            if agent.episode % 10 == 0:
                agent.save_models()
            if agent.episode % 100 == 0:
                print('check point save')
                temp = simulationPath + str(agent.episode) + '_' + str(successCounter / agent.episode) + '_save/'
                os.mkdir(temp)
                time.sleep(0.01)
                agent.actor.save_checkpoint(name='Actor_ddpg', path=temp, num=None)
                agent.target_actor.save_checkpoint(name='TargetActor_ddpg', path=temp, num=None)
                agent.critic.save_checkpoint(name='Critic_ddpg', path=temp, num=None)
                agent.target_critic.save_checkpoint(name='TargetCritic_ddpg', path=temp, num=None)
            if c == 27:
                print('Over......')
                break
        '''dataSave'''
        agent.saveData_EpisodeReward(episode=0, time=0, reward=0, average_reward=0, successrate=0, is2file=True, filename='EpisodeReward.csv')
        agent.saveData_Step_Reward(step=0, reward=0, is2file=True, filename='StepReward.csv')
        '''dataSave'''

    if TEST:
        print('TESTing...')
        RECORD = True
        optPath = '../../datasave/network/DDPG-UGV-Obstacle-Avoidance/parameters/'
        agent = DDPG(modelFileXML=cfgPath + cfgFile, path=simulationPath)
        '''????????????actor????????????????????????????????????'''
        agent.actor = ActorNetwork(1e-4, 8, agent.state_dim_nn - 8, agent.action_dim_nn, 'Actor', simulationPath)
        agent.load_actor_optimal(path=optPath, file='Actor_ddpg')
        '''????????????actor????????????????????????????????????'''
        cap = cv.VideoWriter(simulationPath + '/' + 'Optimal.mp4', cv.VideoWriter_fourcc('X', 'V', 'I', 'D'), 30.0, (env.width, env.height)) if RECORD else None
        simulation_num = 5
        successCounter = 0
        timeOutCounter = 0
        collisionCounter = 0
        for i in range(simulation_num):
            print('==========START' + str(i) + '==========')
            env.reset_random_with_database()
            while not env.is_terminal:
                if cv.waitKey(1) == 27:
                    break
                env.current_state = env.next_state.copy()
                action_from_actor = agent.choose_action(env.current_state, True)
                action = agent.action_linear_trans(action_from_actor)  # ?????????????????????????????????
                env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action)
                env.show_dynamic_imagewithobs(isWait=False)
                cap.write(env.save) if RECORD else None
                env.saveData(is2file=False)
            if env.terminal_flag == 2:
                timeOutCounter += 1
                print('timeout')
            if env.terminal_flag == 3:
                successCounter += 1
                print('success')
            if env.terminal_flag == 4:
                collisionCounter += 1
                print('collision')
            print('===========END===========')
        print('Total:', simulation_num, '  successful:', successCounter, '  timeout:', timeOutCounter, '  collision:', collisionCounter)
        cv.waitKey(0)
        env.saveData(is2file=True, filepath=simulationPath)
        agent.save_models_all()
