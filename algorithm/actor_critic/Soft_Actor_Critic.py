import torch

from environment.config.xml_write import xml_cfg
from common.common_func import *
from common.common_cls import *
import pandas as pd

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = False
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class Soft_Actor_Critic:
    def __init__(self,
                 gamma: float = 0.9,
                 actor_soft_update: float = 1e-2,
                 critic_soft_update: float = 1e-2,
                 alpha: float = 0.2,
                 alpha_lr: float = 3e-4,
                 alpha_learning: bool = True,
                 memory_capacity: int = 5000,
                 batch_size: int = 64,
                 modelFileXML: str = '',
                 path: str = ''):
        """
        @note  brief:				class initialization
        @param gamma:				discount factor
        @param actor_soft_update:
        @param critic_soft_update:
        @param alpha:				factor for actor loss
        @param alpha_learning:		adaptively tune alpha or not
        @param memory_capacity:		capacity of replay memory
        @param batch_size:			bath size
        @param modelFileXML:		model description file
        @param path:				path to load 'modelFileXML'
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

        '''SAC'''
        self.device = device
        self.gamma = gamma
        # self.actor_lr = actor_learning_rate
        # self.critic_lr = critic_learning_rate
        self.actor_tau = actor_soft_update
        self.critic_tau = critic_soft_update
        self.memory = ReplayBuffer(memory_capacity, batch_size, self.state_dim_nn, self.action_dim_nn)

        self.path = path
        self.alpha_learning = alpha_learning
        self.alpha_lr = alpha_lr
        if self.alpha_learning:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -self.action_dim_nn
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1).to(self.device)
            self.log_alpha.requires_grad_(True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = alpha
        '''SAC'''

        '''network'''
        self.actor = ProbActor(1e-4, self.state_dim_nn, self.action_dim_nn, name='Actor', chkpt_dir=self.path)
        # The output of the actor is the mean of a gaussian distribution and a log_pi
        self.critic = DualCritic(1e-3, self.state_dim_nn, self.action_dim_nn, name='Critic', chkpt_dir=self.path)
        # This critic contains double Q-net structure. No difference, just merge the two nets.
        self.target_critic = DualCritic(1e-3, self.state_dim_nn, self.action_dim_nn, name='TargetCritic', chkpt_dir=self.path)
        '''network'''

        self.episode = 0
        self.reward = 0

        self.save_episode = []  # 保存的每一个回合的回合数
        self.save_reward = []  # 保存的每一个回合的奖励
        self.save_time = []
        self.save_average_reward = []  # 保存的每一个回合的平均时间的奖励
        self.save_successful_rate = []
        self.save_step = []  # 保存的每一步的步数
        self.save_stepreward = []  # 保存的每一步的奖励

    def choose_action_random(self):
        """
        @note brief:	因为该函数与choose_action并列，所以输出也必须是[-1, 1]之间
        @return:		random actions
        """
        return np.random.uniform(low=-1, high=1, size=self.action_dim_nn)

    def choose_action(self, state, deterministic=False):
        """
        @note brief:			choose actions deterministically of randomly
        @param state:
        @param deterministic:
        @return: actions
        """
        t_state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0).to(self.device)  # get the tensor of the state
        # self.actor.eval()
        act, _ = self.actor(t_state, deterministic)
        # self.actor.train()
        return act.cpu().detach().numpy().flatten()

    def evaluate(self, state):
        t_state = torch.tensor(state, dtype=torch.float).to(self.device)  # get the tensor of the state
        act, _ = self.actor(t_state, True)
        return act.cpu().detach().numpy()

    def learn(self, is_reward_ascent=False):
        if self.memory.mem_counter < self.memory.batch_size:
            return

        '''第一步：取数据'''
        state, action, reward, new_state, done = self.memory.sample_buffer(is_reward_ascent=is_reward_ascent)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.float).to(self.device)
        '''第一步：取数据'''

        with torch.no_grad():
            action_, log_pi_ = self.actor(new_state)  # 从当前 policy 中得到 a'
            # Compute target Q
            target_q1, target_q2 = self.target_critic(new_state, action_)   # 用的双Q网络，但是放在一个critic里面到了
            '''
            target critic里面出来的是targetQ，critic里面出来的是Q
            '''
            # print('new_state:', new_state.device)
            # print('action_:', action_.device)
            # print('log_pi_:', log_pi_.device)
            # print('target_q1:', target_q1.device)
            # print('target_q2:', target_q2.device)
            # print('done:', done.device)
            # print('reward:', reward.device)
            target_q = reward + self.gamma * (1 - done) * (torch.min(target_q1, target_q2) - self.alpha * log_pi_)
            # target_q = target_q.to(self.critic.device)

        '''current Q'''
        # self.critic.eval()
        current_q1, current_q2 = self.critic(state, action)
        '''current Q'''

        '''critic learn'''
        critic_loss = func.mse_loss(current_q1, target_q) + func.mse_loss(current_q2, target_q)  # 两个 Q 需要同时学习，两个TD误差加起来
        # self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        # self.critic.eval()
        '''critic learn'''

        '''compute actor loss'''
        a, log_pi = self.actor(state)
        q1, q2 = self.critic(state, a)
        q = torch.min(q1, q2)
        actor_loss = torch.mean(self.alpha * log_pi - q)
        '''compute actor loss'''

        '''actor learn'''
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        '''actor learn'''

        if self.alpha_learning:
            alpha_loss = -torch.mean(self.log_alpha.exp() * (log_pi + self.target_entropy).detach())
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        for par, target_par in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_par.data.copy_(self.critic_tau * par.data + (1 - self.critic_tau) * target_par.data)

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

    def SAC_info(self):
        print('agent name：', self.agentName)
        print('state_dim:', self.state_dim_nn)
        print('action_dim:', self.action_dim_nn)
        print('action_range:', self.action_range)

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

    def saveData_Step_Reward(self, step, reward, is2file=False, filename='StepReward.csv'):
        if is2file:
            data = pd.DataFrame({
                'step:': self.save_step,
                'stepreward': self.save_stepreward,
            })
            data.to_csv(self.path + filename, index=False, sep=',')
        else:
            self.save_step.append(step)
            self.save_stepreward.append(reward)

    def saveData_EpisodeReward(self, episode, time, reward, average_reward, successrate, is2file=False, filename='EpisodeReward.csv'):
        if is2file:
            data = pd.DataFrame({
                'episode': self.save_episode,
                'time': self.save_time,
                'reward': self.save_reward,
                'average_reward': self.save_average_reward,
                'success_rate': self.save_successful_rate
            })
            data.to_csv(self.path + filename, index=False, sep=',')
        else:
            self.save_episode.append(episode)
            self.save_time.append(time)
            self.save_reward.append(reward)
            self.save_average_reward.append(average_reward)
            self.save_successful_rate.append(successrate)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def save_models_all(self):
        self.actor.save_all_net()
        self.critic.save_all_net()
        self.target_critic.save_all_net()

    def load_models(self, path):
        """
        :brief:         only for test
        :param path:    file path
        :return:
        """
        print('...loading checkpoint...')
        self.actor.load_state_dict(torch.load(path + 'Actor_sac'))
        self.critic.load_state_dict(torch.load(path + 'Critic_sac'))
        self.target_critic.load_state_dict(torch.load(path + 'TargetCritic_sac'))

    def load_actor_optimal(self, path, file):
        print('...loading optimal...')
        self.actor.load_state_dict(torch.load(path + file))
