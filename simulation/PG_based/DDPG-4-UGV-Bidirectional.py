import os
import sys
import datetime
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
# import copy
from environment.envs.UGV.ugv_bidirectional_continuous import UGV_Bidirectional_Continuous
from algorithm.actor_critic.DDPG import DDPG
from common.common import *

cfgPath = '../../environment/config/'
cfgFile = 'UGV_Bidirectional_Continuous.xml'
optPath = '../../datasave/network/'
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

        self.fc2 = nn.Linear(128, 128)  # hidden1 -> hidden2
        self.batch_norm2 = nn.LayerNorm(128)

        self.action_value = nn.Linear(self.action_dim, 128)  # action -> hidden2
        self.q = nn.Linear(128, 1)  # hidden2 -> output action value

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

        self.fc2 = nn.Linear(128, 128)  # ?????????????????? -> ??????????????????
        self.batch_norm2 = nn.LayerNorm(128)

        self.mu = nn.Linear(128, self.action_dim)  # ?????????????????? -> ?????????

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


def fullFillReplayMemory_with_Optimal(randomEnv: bool, fullFillRatio: float, is_only_success: bool):
    print('Retraining...')
    print('Collecting...')
    agent.load_models(path='./DDPG-Two-Wheel-UGV????????????/')
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
            env.current_state = env.next_state.copy()  # ????????????
            _action_from_actor = agent.choose_action(env.current_state, is_optimal=False, sigma=1 / 2)
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
                if agent.memory.mem_counter % 100 == 0 and agent.memory.mem_counter > 0:
                    print('replay_count = ', agent.memory.mem_counter)
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
        env.reset_random() if randomEnv else env.reset()
        _new_state.clear()
        _new_action.clear()
        _new_reward.clear()
        _new_state_.clear()
        _new_done.clear()
        while not env.is_terminal:
            if agent.memory.mem_counter % 100 == 0 and agent.memory.mem_counter > 0:
                print('replay_count = ', agent.memory.mem_counter)
            env.current_state = env.next_state.copy()  # ????????????
            _action_from_actor = agent.choose_action(env.current_state, False, sigma=1.0)
            # _action_from_actor = ddpg.choose_action_random()
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
            if env.terminal_flag == 3 or env.terminal_flag == 2:
                print('Update Replay Memory......')
                agent.memory.store_transition_per_episode(_new_state, _new_action, _new_reward, _new_state_, _new_done)
                print('replay_count = ', agent.memory.mem_counter)
    agent.memory.get_reward_sort()


if __name__ == '__main__':
    simulationPath = '../../datasave/log/' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-DDPG-Two-Wheel-UGV/'
    os.mkdir(simulationPath)
    c = cv.waitKey(1)
    TRAIN = False  # ????????????
    RETRAIN = False  # ???????????????????????????????????????
    TEST = not TRAIN
    is_storage_only_success = True

    env = UGV_Bidirectional_Continuous(initPhi=deg2rad(0),
                                       save_cfg=False,
                                       x_size=4.0,
                                       y_size=4.0,
                                       start=[2.0, 2.0],
                                       terminal=[4.0, 4.0])

    if TRAIN:
        agent = DDPG(gamma=0.9,
                     actor_soft_update=1e-2,
                     critic_soft_update=1e-2,
                     memory_capacity=24000,
                     batch_size=256,
                     modelFileXML=cfgPath + cfgFile,
                     path=simulationPath)
        '''????????????actor???critic????????????????????????????????????'''
        agent.actor = ActorNetwork(1e-4, agent.state_dim_nn, agent.action_dim_nn, 'Actor', simulationPath)
        agent.target_actor = ActorNetwork(1e-4, agent.state_dim_nn, agent.action_dim_nn, 'TargetActor', simulationPath)
        agent.critic = CriticNetWork(1e-3, agent.state_dim_nn, agent.action_dim_nn, 'Critic', simulationPath)
        agent.target_critic = CriticNetWork(1e-3, agent.state_dim_nn, agent.action_dim_nn, 'TargetCritic', simulationPath)
        '''????????????actor???critic????????????????????????????????????'''
        agent.DDPG_info()
        successCounter = 0
        timeOutCounter = 0
        # cv.waitKey(0)
        MAX_EPISODE = 5000
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
            fullFillReplayMemory_Random(randomEnv=True, fullFillRatio=0.5, is_only_success=True)
            '''fullFillReplayMemory_Random'''
        print('Start to train...')
        new_state, new_action, new_reward, new_state_, new_done = [], [], [], [], []
        step = 0
        while agent.episode <= MAX_EPISODE:
            # env.reset()
            print('=========START=========')
            print('Episode:', agent.episode)
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
                    action_from_actor = agent.choose_action_random()  # ???????????????????????????????????????
                else:
                    action_from_actor = agent.choose_action(env.current_state, False, sigma=1 / 2)  # ?????????????????????????????????
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
                    agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
                agent.learn(is_reward_ascent=False)
            # cv.destroyAllWindows()
            '''??????????????????????????????'''
            if env.terminal_flag == 3:
                successCounter += 1
            if env.terminal_flag == 2:
                timeOutCounter += 1
            if is_storage_only_success:  # ????????????????????????????????????
                if (env.terminal_flag == 3) or (env.terminal_flag == 2):
                    print('Update Replay Memory......')
                    agent.memory.store_transition_per_episode(new_state, new_action, new_reward, new_state_, new_done)
                    # ddpg.memory.get_reward_resort(per=5)
            '''??????????????????????????????'''
            print('Cumulative reward:', round(sumr, 3))
            print('??????', agent.episode, ' ???????????????', successCounter, ' ???????????????', timeOutCounter, ' ??????')
            print('==========END=========')
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
        agent.saveData_Step_Reward(step=step, reward=env.reward, is2file=True, filename='StepReward.csv')
        '''dataSave'''

    if TEST:
        print('TESTing...')
        agent = DDPG(modelFileXML=cfgPath + cfgFile)
        optPath = '../../datasave/network/DDPG-UGV-Bidirectional/parameters/'
        '''????????????actor???critic????????????????????????????????????'''
        agent.actor = ActorNetwork(1e-4, agent.state_dim_nn, agent.action_dim_nn, 'Actor', simulationPath)
        '''????????????actor???critic????????????????????????????????????'''
        agent.load_actor_optimal(path=optPath, file='Actor_ddpg')
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
                        action_from_actor = agent.choose_action(env.current_state, True)
                        action = agent.action_linear_trans(action_from_actor)  # ?????????????????????????????????
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
        # agent.save_models_all()
