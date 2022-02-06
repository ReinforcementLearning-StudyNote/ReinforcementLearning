# import random
# import torch.nn as nn
# import torch.nn.functional as func
# import torch
from environment.config.xml_write import xml_cfg
from common.common import *
import pandas as pd

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = False
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class CriticNetWork(nn.Module):
    def __init__(self, beta, state_dim, fc1_dims, fc2_dims, action_dim, name, chkpt_dir):
        super(CriticNetWork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_ddpg'

        self.fc1 = nn.Linear(self.state_dim, fc1_dims)  # state -> hidden1
        self.batch_norm1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)  # hidden1 -> hidden2
        self.batch_norm2 = nn.LayerNorm(fc2_dims)

        self.action_value = nn.Linear(self.action_dim, fc2_dims)  # action -> hidden2
        self.q = nn.Linear(fc2_dims, 1)  # hidden2 -> output action value

        # self.initialization()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)  # forward
        state_value = self.batch_norm1(state_value)  # batch normalization
        state_value = func.relu(state_value)  # relu

        state_value = self.fc2(state_value)
        state_value = self.batch_norm2(state_value)

        action_value = func.relu(self.action_value(action))
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

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self,
                 alpha,
                 state_dim1, fc1_dims1, fc2_dims1, fc3_dims1,
                 state_dim2, fc1_dims2, fc2_dims2, fc3_dims2,
                 fc_combine_dims,
                 action_dim, name, chkpt_dir):
        super(ActorNetwork, self).__init__()
        self.state_dim1 = state_dim1
        self.state_dim2 = state_dim2
        self.action_dim = action_dim

        self.checkpoint_file = chkpt_dir + name + '_ddpg'

        self.linear11 = nn.Linear(self.state_dim1, fc1_dims1)  # 第一部分网络第一层
        self.batch_norm11 = nn.LayerNorm(fc1_dims1)
        self.linear12 = nn.Linear(fc1_dims1, fc2_dims1)  # 第一部分网络第二层
        self.batch_norm12 = nn.LayerNorm(fc2_dims1)
        self.linear13 = nn.Linear(fc2_dims1, fc3_dims1)
        self.batch_norm13 = nn.LayerNorm(fc3_dims1)  # 第一部分网络第三层

        self.linear21 = nn.Linear(self.state_dim2, fc1_dims2)  # 第二部分网络第一层
        self.batch_norm21 = nn.LayerNorm(fc1_dims2)
        self.linear22 = nn.Linear(fc1_dims2, fc2_dims2)  # 第二部分网络第二层
        self.batch_norm22 = nn.LayerNorm(fc2_dims2)
        self.linear23 = nn.Linear(fc2_dims2, fc3_dims2)
        self.batch_norm23 = nn.LayerNorm(fc3_dims2)  # 第二部分网络第三层

        self.mu = nn.Linear(fc3_dims1 + fc3_dims2, self.action_dim)

        # self.combine = nn.Linear(fc3_dims1 + fc3_dims2, fc_combine_dims)  # 第三层，合并
        # self.mu = nn.Linear(fc_combine_dims, self.action_dim)  # 第四层，直接输出

        # self.initialization()

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

    def forward(self, state1, state2):
        """
        :param state1:      first part of the data
        :param state2:      second part of teh data
        :return:            output of the net
        """
        x1 = self.linear11(state1)
        x1 = self.batch_norm11(x1)
        x1 = func.relu(x1)

        x1 = self.linear12(x1)
        x1 = self.batch_norm12(x1)
        x1 = func.relu(x1)

        x1 = self.linear13(x1)
        x1 = self.batch_norm13(x1)
        x1 = func.relu(x1)  # 该合并了

        x2 = self.linear21(state2)
        x2 = self.batch_norm21(x2)
        x2 = func.relu(x2)

        x2 = self.linear22(x2)
        x2 = self.batch_norm22(x2)
        x2 = func.relu(x2)

        x2 = self.linear23(x2)
        x2 = self.batch_norm23(x2)
        x2 = func.relu(x2)  # 该合并了

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

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class DDPG2:
    def __init__(self,
                 gamma: float = 0.9,
                 actor_learning_rate: float = 1e-3,
                 critic_learning_rate: float = 1e-3,
                 actor_soft_update: float = 1e-2,
                 critic_soft_update: float = 1e-2,
                 memory_capacity: int = 5000,
                 batch_size: int = 64,
                 modelFileXML: str = '',
                 path: str = ''
                 ):
        """
        :param gamma:                   discount factor
        :param actor_learning_rate:     learning rate of actor net
        :param critic_learning_rate:    learning rate of critic net
        :param actor_soft_update:       soft update rate of actor
        :param critic_soft_update:      soft update rate of critic
        :param memory_capacity:         capacity of the replay memory
        :param batch_size:              batch size
        :param modelFileXML:            model file
        """
        '''From rl_base'''
        # DDPG 要求智能体状态必须是连续的，动作必须连续的
        self.agentName, self.state_dim_nn, self.action_dim_nn, self.action_range = \
            self.get_RLBase_from_XML(modelFileXML)
        # agentName:            the name of the agent
        # state_dim_nn:         the dimension of the neural network input
        # action_dim_nn:        the dimension of the neural network output
        # action_range:         the range of physical action
        '''From rl_base'''

        '''DDPG'''
        self.gamma = gamma
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.actor_tau = actor_soft_update
        self.critic_tau = critic_soft_update
        self.memory = ReplayBuffer(memory_capacity, batch_size, self.state_dim_nn, self.action_dim_nn)
        '''DDPG'''

        '''obstacle2'''
        self.state_dim_nn1 = 10
        self.state_dim_nn2 = self.state_dim_nn - self.state_dim_nn1

        '''network'''
        # self.actor = ActorNetwork(self.actor_lr,
        #                           self.state_dim_nn1, 64, 32, 32,     # 非激光雷达
        #                           self.state_dim_nn2, 256, 128, 64,     # 激光雷达
        #                           64,
        #                           self.action_dim_nn, name='Actor', chkpt_dir=path)
        # self.target_actor = ActorNetwork(self.actor_lr,
        #                                  self.state_dim_nn1, 64, 32, 32,
        #                                  self.state_dim_nn2, 256, 128, 64,
        #                                  64,
        #                                  self.action_dim_nn, name='TargetActor', chkpt_dir=path)
        #
        # self.critic = CriticNetWork(self.critic_lr, self.state_dim_nn, 128, 64, self.action_dim_nn, name='Critic', chkpt_dir=path)
        # self.target_critic = CriticNetWork(self.critic_lr, self.state_dim_nn, 128, 64, self.action_dim_nn, name='TargetCritic', chkpt_dir=path)   # 第一次的

        self.actor = ActorNetwork(self.actor_lr,
                                  self.state_dim_nn1, 128, 64, 64,     # 非激光雷达
                                  self.state_dim_nn2, 128, 64, 32,     # 激光雷达
                                  64,
                                  self.action_dim_nn, name='Actor', chkpt_dir=path)
        self.target_actor = ActorNetwork(self.actor_lr,
                                         self.state_dim_nn1, 128, 64, 64,
                                         self.state_dim_nn2, 128, 64, 32,
                                         64,
                                         self.action_dim_nn, name='TargetActor', chkpt_dir=path)

        self.critic = CriticNetWork(self.critic_lr, self.state_dim_nn, 128, 64, self.action_dim_nn, name='Critic', chkpt_dir=path)
        self.target_critic = CriticNetWork(self.critic_lr, self.state_dim_nn, 128, 64, self.action_dim_nn, name='TargetCritic', chkpt_dir=path)
        '''network'''
        '''obstacle2'''

        self.noise_OU = OUActionNoise(mu=np.zeros(self.action_dim_nn))
        self.noise_gaussian = GaussianNoise(mu=np.zeros(self.action_dim_nn))
        self.update_network_parameters()

        self.episode = 0
        self.reward = 0

        self.save_episode = []
        self.save_reward = []
        self.save_step = []
        self.save_stepreward = []

    def choose_action_random(self):
        """
        :brief:     因为该函数与choose_action并列，所以输出也必须是[-1, 1]之间
        :return:    random action
        """
        return np.random.uniform(low=-1, high=1, size=self.action_dim_nn)

    def choose_action(self, state, is_optimal=False, sigma=1 / 3):
        self.actor.eval()  # 切换到测试模式
        t_state = torch.tensor(state, dtype=torch.float).to(self.actor.device)  # get the tensor of the state
        [t_state1, t_state2] = torch.split(t_state, [self.state_dim_nn1, self.state_dim_nn2])
        mu = self.actor(t_state1, t_state2).to(self.actor.device)  # choose action
        if is_optimal:
            mu_prime = mu
        else:
            mu_prime = mu + torch.tensor(self.noise_gaussian(sigma=sigma), dtype=torch.float).to(self.actor.device)  # action with gaussian noise
            # mu_prime = mu + torch.tensor(self.noise_OU(), dtype=torch.float).to(self.actor.device)             # action with OU noise
        self.actor.train()  # 切换回训练模式
        mu_prime_np = mu_prime.cpu().detach().numpy()
        return np.clip(mu_prime_np, -1, 1)  # 将数据截断在[-1, 1]之间

    def learn(self, is_reward_ascent=True):
        if self.memory.mem_counter < self.memory.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(is_reward_ascent=is_reward_ascent)
        state = torch.tensor(state, dtype=torch.float).to(self.critic.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.critic.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.critic.device)
        done = torch.tensor(done, dtype=torch.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        # print(new_state.size())
        [new_state1, new_state2] = torch.split(new_state, [self.state_dim_nn1, self.state_dim_nn2], dim=1)

        target_actions = self.target_actor.forward(new_state1, new_state2)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.memory.batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * done[j])
        target = torch.tensor(target).to(self.critic.device)
        target = target.view(self.memory.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = func.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        [state1, state2] = torch.split(state, [self.state_dim_nn1, self.state_dim_nn2], dim=1)

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state1, state2)  # 是个动作
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)  # 是个评价
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self):
        """
        :return:        None
        """
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.critic_tau) + param.data * self.critic_tau)  # soft update
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.actor_tau) + param.data * self.actor_tau)  # soft update

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self, path):
        """
        :brief:         only for test
        :param path:    file path
        :return:
        """
        print('...loading checkpoint...')
        self.actor.load_state_dict(torch.load(path + 'Actor_ddpg'))
        self.target_actor.load_state_dict(torch.load(path + 'TargetActor_ddpg'))
        self.critic.load_state_dict(torch.load(path + 'Critic_ddpg'))
        self.target_critic.load_state_dict(torch.load(path + 'TargetCritic_ddpg'))

    def load_actor_optimal(self, path, file):
        print('...loading optimal...')
        self.actor.load_state_dict(torch.load(path + file))

    def get_RLBase_from_XML(self, filename):
        rl_base, agentName = self.load_rl_basefromXML(filename=filename)
        state_dim_nn = int(rl_base['state_dim'])  # input dimension of NN
        action_dim_nn = int(rl_base['action_dim'])
        action_range = str2list(rl_base['action_range'])
        return agentName, state_dim_nn, action_dim_nn, action_range

    @staticmethod
    def load_rl_basefromXML(filename: str) -> (dict, str):
        """
        :brief:             从模型文件中加载数据到DQN中
        :param filename:    模型文件
        :return:            数据字典
        """
        root = xml_cfg().XML_Load(filename)
        return xml_cfg().XML_GetTagValue(node=xml_cfg().XML_FindNode(nodename='RL_Base', root=root)), root.attrib['name']

    def DDPG_info(self):
        print('agent name：', self.agentName)
        print('state_dim:', self.state_dim_nn)
        print('action_dim:', self.action_dim_nn)
        print('action_range:', self.action_range)
        print(self.actor)
        print(self.critic)

    def action_linear_trans(self, action):
        # the action output
        linear_action = []
        for i in range(self.action_dim_nn):
            a = min(max(action[i], -1), 1)
            maxa = self.action_range[i][1]
            mina = self.action_range[i][0]
            k = (maxa - mina) / 2
            b = (maxa + mina) / 2
            linear_action.append(k * a + b)
        return linear_action

    def saveData_Step_Reward(self,
                             step,
                             reward,
                             is2file=False,
                             filename='StepReward.csv',
                             filepath=''):
        if is2file:
            data = pd.DataFrame({
                'step:': self.save_step,
                'stepreward': self.save_stepreward,
            })
            data.to_csv(filepath + filename, index=False, sep=',')
        else:
            self.save_step.append(step)
            self.save_stepreward.append(reward)

    def saveData_EpisodeReward(self,
                               episode,
                               reward,
                               is2file=False,
                               filename='EpisodeReward.csv',
                               filepath=''):
        if is2file:
            data = pd.DataFrame({
                'episode:': self.save_episode,
                'reward': self.save_reward,
            })
            data.to_csv(filepath + filename, index=False, sep=',')
        else:
            self.save_episode.append(episode)
            self.save_reward.append(reward)
