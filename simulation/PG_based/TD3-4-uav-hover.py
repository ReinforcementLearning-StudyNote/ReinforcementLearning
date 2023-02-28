import math
import os
import sys
import datetime
import time
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
# import copy
from environment.envs.UAV.uav_hover import UAV_Hover
from algorithm.actor_critic.Twin_Delayed_DDPG import Twin_Delayed_DDPG as TD3
from common.common_func import *
from common.common_cls import *
import matplotlib.pyplot as plt


cfgPath = '../../environment/config/'
cfgFile = 'UGV_Forward_Obstacle_Continuous.xml'
optPath = '../../datasave/network/'
show_per = 10


class CriticNetWork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, name, chkpt_dir):
        super(CriticNetWork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_td3'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_td3ALL'

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
    def __init__(self, alpha, state_dim, action_dim, name, chkpt_dir):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_td3'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_td3ALL'

        self.fc1 = nn.Linear(self.state_dim, 128)  # 输入 -> 第一个隐藏层
        self.batch_norm1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 128)  # 第一个隐藏层 -> 第二个隐藏层
        self.batch_norm2 = nn.LayerNorm(128)

        # self.fc3 = nn.Linear(64, 32)  # 第2个隐藏层 -> 第3个隐藏层
        # self.batch_norm3 = nn.LayerNorm(32)

        self.mu = nn.Linear(128, self.action_dim)  # 第3个隐藏层 -> 输出层

        # self.initialization()
        self.initialization_default()

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

    def initialization_default(self):
        self.fc1.reset_parameters()
        self.batch_norm1.reset_parameters()
        self.fc2.reset_parameters()
        self.batch_norm2.reset_parameters()
        # self.fc3.reset_parameters()
        # self.batch_norm3.reset_parameters()
        self.mu.reset_parameters()

    def forward(self, state):
        x = self.fc1(state)
        x = self.batch_norm1(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = func.relu(x)

        # x = self.fc3(x)
        # x = self.batch_norm3(x)
        # x = func.relu(x)

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
    pass


def fullFillReplayMemory_Random(randomEnv: bool, fullFillRatio: float, is_only_success: bool):
    """
    :param randomEnv:           init env randomly
    :param fullFillRatio:       the ratio
    :param is_only_success:
    :return:
    """
    print('Collecting...')
    pass


if __name__ == '__main__':
    simulationPath = '../../datasave/log/' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-TD3-uav-hover/'
    os.mkdir(simulationPath)
    c = cv.waitKey(1)
    TRAIN = False  # 直接训练
    RETRAIN = False  # 基于之前的训练结果重新训练
    TEST = not TRAIN
    is_storage_only_success = False

    env = UAV_Hover()   # 所有初始值都是0，合理

    if TRAIN:
        agent = TD3(gamma=0., noise_clip=1 / 2, noise_policy=1 / 4, policy_delay=3,
                    critic1_soft_update=1e-2,
                    critic2_soft_update=1e-2,
                    actor_soft_update=1e-2,
                    memory_capacity=60000,  # 100000
                    batch_size=512,  # 1024
                    modelFileXML=cfgPath + cfgFile,
                    path=simulationPath)

        '''重新加载actor和critic网络结构，这是必须的操作'''
        agent.actor = ActorNetwork(1e-4, agent.state_dim_nn, agent.action_dim_nn, 'Actor', simulationPath)
        agent.target_actor = ActorNetwork(1e-4, agent.state_dim_nn, agent.action_dim_nn, 'TargetActor', simulationPath)
        agent.critic1 = CriticNetWork(1e-3, agent.state_dim_nn, agent.action_dim_nn, 'Critic1', simulationPath)
        agent.target_critic1 = CriticNetWork(1e-3, agent.state_dim_nn, agent.action_dim_nn, 'TargetCritic1', simulationPath)
        agent.critic2 = CriticNetWork(1e-3, agent.state_dim_nn, agent.action_dim_nn, 'Critic2', simulationPath)
        agent.target_critic2 = CriticNetWork(1e-3, agent.state_dim_nn, agent.action_dim_nn, 'TargetCritic2', simulationPath)
        '''重新加载actor和critic网络结构，这是必须的操作'''

        agent.DDPG_info()
        successCounter = 0
        timeOutCounter = 0
        collisionCounter = 0
        # cv.waitKey(0)
        MAX_EPISODE = 20000

        if RETRAIN:
            print('Retraining')
            fullFillReplayMemory_with_Optimal(randomEnv=True, fullFillRatio=0.5, is_only_success=is_storage_only_success)
            # 如果注释掉，就是在上次的基础之上继续学习，如果不是就是重新学习，但是如果两次的奖励函数有变化，那么就必须执行这两句话
            '''生成初始数据之后要再次初始化网络'''
            # agent.actor.initialization_default()
            # agent.target_actor.initialization_default()
            # agent.critic.initialization_default()
            # agent.target_critic.initialization_default()
            '''生成初始数据之后要再次初始化网络'''
        else:
            '''fullFillReplayMemory_Random'''
            fullFillReplayMemory_Random(randomEnv=True, fullFillRatio=0.25, is_only_success=is_storage_only_success)
            '''fullFillReplayMemory_Random'''
        print('Start to train...')
        new_state, new_action, new_reward, new_state_, new_done = [], [], [], [], []
        step = 0
        plt.ion()
        while agent.episode <= MAX_EPISODE:
            print('=========START=========')
            print('Episode:', agent.episode)
            env.reset_random()      # 随机初始化初始位置和目标点
            sumr = 0
            new_state.clear()
            new_action.clear()
            new_reward.clear()
            new_state_.clear()
            new_done.clear()

            while not env.is_terminal:
                env.current_state = env.next_state.copy()
                epsilon = random.uniform(0, 1)
                if epsilon < 0.15:
                    action_from_actor = agent.choose_action_random()  # 有一定探索概率完全随机探索
                else:
                    action_from_actor = agent.choose_action(env.current_state, False, sigma=1 / 4)  # 剩下的是神经网络加噪声
                action = agent.action_linear_trans(action_from_actor)  # 将动作转换到实际范围上
                env.current_state, env.current_action, env.reward, env.next_state, env.is_terminal = env.step_update(action)  # 环境更新的action需要是物理的action
                agent.saveData_Step_Reward(step=step, reward=env.reward, is2file=False, filename='StepReward.csv')
                step += 1
                env.show_dynamic_image(per_show=show_per)       # 画图
                sumr = sumr + env.reward
                if is_storage_only_success:
                    '''设置一个限制，只有满足某些条件的[s a r s' done]才可以被加进去'''
                    new_state.append(env.current_state)
                    new_action.append(env.current_action)
                    new_reward.append(env.reward)
                    new_state_.append(env.next_state)
                    new_done.append(1.0 if env.is_terminal else 0.0)
                else:
                    agent.memory.store_transition(env.current_state, env.current_action, env.reward, env.next_state, 1 if env.is_terminal else 0)
                agent.learn(is_reward_ascent=False)
        plt.ioff()

    if TEST:
        pass