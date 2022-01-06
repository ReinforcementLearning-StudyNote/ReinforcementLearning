# import random
import torch.nn as nn
import torch.nn.functional as func
import torch
from environment.config.xml_write import xml_cfg
from common.common import *
import pandas as pd

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = True
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class CriticNetWork(nn.Module):
    def __init__(self, beta, state_dim, fc1_dims, fc2_dims, action_dim, name, chkpt_dir):
        super(CriticNetWork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')
        self.checkpoint_file = chkpt_dir + name + '_ddpg'

        # print('嘤嘤嘤', self.state_dim, fc1_dims)
        self.fc1 = nn.Linear(self.state_dim, fc1_dims)
        self.batch_norm1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.batch_norm2 = nn.LayerNorm(fc2_dims)

        self.action_value = nn.Linear(self.action_dim, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.initialization()

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

    def save_checkpoint(self):
        print('...saving checkpoint...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, fc1_dims, fc2_dims, action_dim, name, chkpt_dir):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')
        self.checkpoint_file = chkpt_dir + name + '_ddpg'

        self.fc1 = nn.Linear(self.state_dim, fc1_dims)
        self.batch_norm1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.batch_norm2 = nn.LayerNorm(fc2_dims)

        self.mu = nn.Linear(fc2_dims, self.action_dim)

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

    def save_checkpoint(self):
        print('...saving checkpoint...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class DDPG:
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

        '''network'''
        self.actor = ActorNetwork(self.actor_lr, self.state_dim_nn, 128, 128, self.action_dim_nn, name='Actor', chkpt_dir=path)
        self.target_actor = ActorNetwork(self.actor_lr, self.state_dim_nn, 128, 128, self.action_dim_nn, name='TargetActor', chkpt_dir=path)

        self.critic = CriticNetWork(self.critic_lr, self.state_dim_nn, 128, 128, self.action_dim_nn, name='Critic', chkpt_dir=path)
        self.target_critic = CriticNetWork(self.critic_lr, self.state_dim_nn, 128, 128, self.action_dim_nn, name='TargetCritic', chkpt_dir=path)
        '''network'''

        self.noise = OUActionNoise(mu=np.zeros(self.action_dim_nn))
        self.noise_gaussian = GaussianNoise(mu=np.zeros(self.action_dim_nn))
        self.update_network_parameters()

        self.episode = 0
        self.reward = 0

        self.save_episode = [self.episode]
        self.save_reward = [self.reward]

    def choose_action(self, state, is_optimal):
        self.actor.eval()  # 切换到测试模式
        t_state = torch.tensor(state, dtype=torch.float).to(self.actor.device)  # get the tensor of the state
        mu = self.actor(t_state).to(self.actor.device)  # choose action
        # mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float).to(self.actor.device)             # action with OU noise
        if is_optimal:
            mu_prime = mu
        else:
            mu_prime = mu + torch.tensor(self.noise_gaussian(), dtype=torch.float).to(self.actor.device)  # action with gaussian noise
        self.actor.train()  # 切换回训练模式
        mu_prime_np = mu_prime.cpu().detach().numpy()
        return np.clip(mu_prime_np, -1, 1)  # 将数据截断在[-1, 1]之间

    def learn(self):
        if self.memory.mem_counter < self.memory.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer()
        state = torch.tensor(state, dtype=torch.float).to(self.critic.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.critic.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.critic.device)
        done = torch.tensor(done, dtype=torch.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(new_state)  # 256 4
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

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
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

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()

    def actor_load_optimal(self, path, file):
        """
        :brief:         only for test
        :param path:    file path
        :param file:    file name
        :return:
        """
        print('...loading checkpoint...')
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

    def action_linear_trans(self, action):
        # the action output
        linear_action = []
        for i in range(self.action_dim_nn):
            a = action[i]
            maxa = self.action_range[i][1]
            mina = self.action_range[i][0]
            k = (maxa - mina) / 2
            b = (maxa + mina) / 2
            linear_action.append(k * a + b)
        return linear_action

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
