import os
import sys
import datetime
import copy
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from environment.envs.FlightAttitudeSimulator.flight_attitude_simulator_continuous import Flight_Attitude_Simulator_Continuous as flight_sim_con
from algorithm.actor_critic.DDPG import DDPG
from common.common import *

cfgPath = '../../environment/config/'
cfgFile = 'Flight_Attitude_Simulator_Continuous.xml'
optPath = '../../datasave/network/'
show_per = 1


class CriticNetWork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, name, chkpt_dir):
        super(CriticNetWork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

        self.fc1 = nn.Linear(self.state_dim, 64)  # state -> hidden1
        self.batch_norm1 = nn.LayerNorm(64)

        self.fc2 = nn.Linear(64, 64)  # hidden1 -> hidden2
        self.batch_norm2 = nn.LayerNorm(64)

        self.action_value = nn.Linear(self.action_dim, 64)  # action -> hidden2
        self.q = nn.Linear(64, 1)  # hidden2 -> output action value

        self.initialization()

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
    def __init__(self, alpha, state_dim, action_dim, name, chkpt_dir):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

        self.fc1 = nn.Linear(self.state_dim, 128)  # ?????? -> ??????????????????
        self.batch_norm1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 64)  # ?????????????????? -> ??????????????????
        self.batch_norm2 = nn.LayerNorm(64)

        self.mu = nn.Linear(64, self.action_dim)  # ?????????????????? -> ?????????

        self.initialization()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def initialization(self):
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3 = 0.003
        nn.init.uniform_(self.mu.weight.data, -f3, f3)
        nn.init.uniform_(self.mu.bias.data, -f3, f3)

    def forward(self, state):
        x = self.fc1(state)
        x = self.batch_norm1(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = func.relu(x)

        x = torch.tanh(self.mu(x))  # bound the output to [-1, 1]

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
    agent.load_models(optPath)
    fullFillCount = int(fullFillRatio * agent.memory.mem_size)
    fullFillCount = max(min(fullFillCount, agent.memory.mem_size), agent.memory.batch_size)
    _new_state, _new_action, _new_reward, _new_state_, _new_done = [], [], [], [], []
    while agent.memory.mem_counter < fullFillCount:
        env.reset_random() if randomEnv else env.reset()
        _new_state.clear()
        _new_action.clear()
        _new_reward.clear()
        _new_state_.clear()
        _new_done.clear()
        while not env.is_terminal:
            if agent.memory.mem_counter % 100 == 0:
                print('replay_count = ', agent.memory.mem_counter)
            env.current_state = env.next_state.copy()  # ????????????
            _action_from_actor = agent.choose_action(env.current_state, is_optimal=True)
            _action = agent.action_linear_trans(_action_from_actor)
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(_action)
            env.show_dynamic_image(isWait=False)
            if is_only_success:
                _new_state.append(env.current_state)
                _new_action.append(env.current_action)
                _new_reward.append(env.reward)
                _new_state_.append(env.next_state)
                _new_done.append(1.0 if env.is_terminal else 0.0)
            else:
                agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
        if is_only_success:
            if env.terminal_flag == 3:
                print('Update Replay Memory......')
                agent.memory.store_transition_per_episode(_new_state, _new_action, _new_reward, _new_state_, _new_done)


def fullFillReplayMemory_Random(randomEnv: bool, fullFillRatio: float):
    """
    :param randomEnv:       init env randomly
    :param fullFillRatio:   the ratio
    :return:                None
    """
    print('Collecting...')
    fullFillCount = int(fullFillRatio * agent.memory.mem_size)
    fullFillCount = max(min(fullFillCount, agent.memory.mem_size), agent.memory.batch_size)
    while agent.memory.mem_counter < fullFillCount:
        env.reset_random() if randomEnv else env.reset()
        while not env.is_terminal:
            if agent.memory.mem_counter % 100 == 0:
                print('replay_count = ', agent.memory.mem_counter)
            env.current_state = env.next_state.copy()  # ????????????
            _action_from_actor = agent.choose_action(env.current_state, False)
            _action = agent.action_linear_trans(_action_from_actor)
            env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(_action)
            env.show_dynamic_image(isWait=False)
            # if env.reward > 0:
            agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)


if __name__ == '__main__':
    simulationPath = '../../datasave/log/' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-DDPG-FlightAttitudeSimulator/'
    os.mkdir(simulationPath)
    c = cv.waitKey(1)
    TRAIN = False  # ????????????
    RETRAIN = False  # ???????????????????????????????????????
    TEST = not TRAIN
    is_storage_only_success = False

    env = flight_sim_con(initTheta=-60.0, setTheta=0.0, save_cfg=False)

    if TRAIN:
        agent = DDPG(gamma=0.9,
                     actor_soft_update=1e-2,
                     critic_soft_update=1e-2,
                     memory_capacity=20000,  # 10000
                     batch_size=512,
                     modelFileXML=cfgPath + cfgFile,
                     path=simulationPath)
        '''????????????actor???critic????????????????????????????????????'''
        agent.actor = ActorNetwork(1e-4, agent.state_dim_nn, agent.action_dim_nn, 'Actor', simulationPath)
        agent.target_actor = ActorNetwork(1e-4, agent.state_dim_nn, agent.action_dim_nn, 'TargetActor', simulationPath)
        agent.critic = CriticNetWork(1e-3, agent.state_dim_nn, agent.action_dim_nn, 'Critic', simulationPath)
        agent.target_critic = CriticNetWork(1e-3, agent.state_dim_nn, agent.action_dim_nn, 'TargetCritic', simulationPath)
        '''????????????actor???critic????????????????????????????????????'''
        agent.DDPG_info()
        # cv.waitKey(0)
        MAX_EPISODE = 1500
        if RETRAIN:
            print('Retraining')
            fullFillReplayMemory_with_Optimal(randomEnv=True,
                                              fullFillRatio=0.5,
                                              is_only_success=True)
            # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
            '''????????????????????????????????????????????????'''
            # ddpg.actor.initialization()
            # ddpg.target_actor.initialization()
            # ddpg.critic.initialization()
            # ddpg.target_critic.initialization()
            '''????????????????????????????????????????????????'''
        else:
            '''fullFillReplayMemory_Random'''
            fullFillReplayMemory_Random(randomEnv=True, fullFillRatio=0.5)
            '''fullFillReplayMemory_Random'''
        print('Start to train...')
        new_state, new_action, new_reward, new_state_, new_done = [], [], [], [], []
        step = 0
        while agent.episode <= MAX_EPISODE:
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
                if random.uniform(0, 1) < 0.2:
                    # print('...random...')
                    action_from_actor = agent.choose_action_random()  # ???????????????????????????????????????
                else:
                    action_from_actor = agent.choose_action(env.current_state, False, sigma=1 / 3)  # ?????????????????????????????????
                action = agent.action_linear_trans(action_from_actor)  # ?????????????????????????????????
                env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action)  # ???????????????action??????????????????action
                agent.saveData_Step_Reward(step=step, reward=env.reward, is2file=False, filename='StepReward.csv')
                step += 1
                if agent.episode % show_per == 0:
                    env.show_dynamic_image(isWait=False)
                sumr = sumr + env.reward
                if is_storage_only_success:
                    new_state.append(env.current_state)
                    new_action.append(env.current_action)
                    new_reward.append(env.reward)
                    new_state_.append(env.next_state)
                    new_done.append(1.0 if env.is_terminal else 0.0)
                else:
                    # if env.reward > 0:
                    agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
                agent.learn(is_reward_ascent=False)
            '''??????????????????????????????'''
            if is_storage_only_success:
                if env.terminal_flag == 3:
                    print('Update Replay Memory......')
                    agent.memory.store_transition_per_episode(new_state, new_action, new_reward, new_state_, new_done)
            '''??????????????????????????????'''
            print(
                '=========START=========',
                'Episode:', agent.episode,
                'Cumulative reward:', round(sumr, 3),
                '==========END=========')
            print()
            agent.saveData_EpisodeReward(agent.episode, sumr)
            agent.episode += 1
            if agent.episode % 10 == 0:
                agent.save_models()
            if c == 27:
                print('Over......')
                break
        '''dataSave'''
        agent.saveData_EpisodeReward(0.0, 0.0, True, 'EpisodeReward.csv')
        agent.saveData_Step_Reward(step=0, reward=0, is2file=True, filename='StepReward.csv')
        '''dataSave'''

    if TEST:
        print('TESTing...')
        optPath = '../../datasave/network/DDPG-Flight-Attitude-Simulator/parameters/'
        agent = DDPG(modelFileXML=cfgPath + cfgFile)
        '''????????????actor???critic????????????????????????????????????'''
        agent.actor = ActorNetwork(1e-4, agent.state_dim_nn, agent.action_dim_nn, 'Actor', simulationPath)
        agent.target_actor = ActorNetwork(1e-4, agent.state_dim_nn, agent.action_dim_nn, 'TargetActor', simulationPath)
        agent.critic = CriticNetWork(1e-3, agent.state_dim_nn, agent.action_dim_nn, 'Critic', simulationPath)
        agent.target_critic = CriticNetWork(1e-3, agent.state_dim_nn, agent.action_dim_nn, 'TargetCritic', simulationPath)
        agent.load_actor_optimal(path=optPath, file='Actor_ddpg')
        agent.load_target_actor_optimal(path=optPath, file='TargetActor_ddpg')
        agent.load_critic_optimal(path=optPath, file='Critic_ddpg')
        agent.load_target_critic_optimal(path=optPath, file='TargetCritic_ddpg')
        '''????????????actor???critic????????????????????????????????????'''
        cap = cv.VideoWriter(simulationPath + '/' + 'Optimal.mp4',
                             cv.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                             120.0,
                             (env.width, env.height))
        simulation_num = 5
        for i in range(simulation_num):
            print('==========START==========')
            print('episode = ', i)
            env.reset_random()
            while not env.is_terminal:
                if cv.waitKey(1) == 27:
                    break
                env.current_state = env.next_state.copy()
                action_from_actor = agent.choose_action(env.current_state, True)
                action = agent.action_linear_trans(action_from_actor)  # ?????????????????????????????????
                env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action)
                env.show_dynamic_image(isWait=False)
                cap.write(env.save)
                env.saveData(is2file=False)
            print('Stable Theta:', rad2deg(env.theta), '\t', 'Stable error:', rad2deg(env.setTheta - env.theta))
            print('===========END===========')
        cv.waitKey(0)
        env.saveData(is2file=True, filepath=simulationPath)
        agent.save_models_all()
