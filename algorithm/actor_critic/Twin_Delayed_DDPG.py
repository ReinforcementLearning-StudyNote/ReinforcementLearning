import pandas as pd

from common.common import *
from environment.config.xml_write import xml_cfg

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = False
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class Twin_Delayed_DDPG:
    def __init__(self,
                 gamma: float = 0.9,
                 noise_clip: float = 1 / 2,
                 noise_policy: float = 1 / 4,
                 policy_delay: int = 5,
                 critic1_soft_update: float = 1e-2,
                 critic2_soft_update: float = 1e-2,
                 actor_soft_update: float = 1e-2,
                 memory_capacity: int = 5000,
                 batch_size: int = 64,
                 modelFileXML: str = '',
                 path: str = ''):
        """
        :param gamma:                       discount factor
        :param noise_clip:                  upper limit of noise in policy for training
        :param noise_policy:                variance of noise in policy for training
        :param policy_delay:                update delay
        :param critic1_soft_update:         update rate of critic1
        :param critic2_soft_update:         update rate of critic2
        :param actor_soft_update:           update rate of actor
        :param memory_capacity:             capacity of replay buffer
        :param batch_size:                  batch size
        :param modelFileXML:                model description file
        :param path:                        saving path
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

        '''Twin-Delay-DDPG'''
        self.gamma = gamma
        # for target policy smoothing regularization
        self.noise_clip = noise_clip
        self.noise_policy = noise_policy
        self.action_regularization = GaussianNoise(mu=np.zeros(self.action_dim_nn))
        # for target policy smoothing regularization
        self.policy_delay = policy_delay
        self.policy_delay_iter = 0
        self.critic1_tau = critic1_soft_update
        self.critic2_tau = critic2_soft_update
        self.actor_tau = actor_soft_update
        self.memory = ReplayBuffer(memory_capacity, batch_size, self.state_dim_nn, self.action_dim_nn)
        self.path = path
        '''Twin-Delay-DDPG'''

        '''network'''
        self.actor = ActorNetwork(1e-4, self.state_dim_nn, self.action_dim_nn, name='Actor', chkpt_dir=self.path)
        self.target_actor = ActorNetwork(1e-4, self.state_dim_nn, self.action_dim_nn, name='TargetActor', chkpt_dir=self.path)

        self.critic1 = CriticNetWork(1e-3, self.state_dim_nn, self.action_dim_nn, name='Critic1', chkpt_dir=self.path)
        self.target_critic1 = CriticNetWork(1e-3, self.state_dim_nn, self.action_dim_nn, name='TargetCritic1', chkpt_dir=self.path)

        self.critic2 = CriticNetWork(1e-3, self.state_dim_nn, self.action_dim_nn, name='Critic2', chkpt_dir=self.path)
        self.target_critic2 = CriticNetWork(1e-3, self.state_dim_nn, self.action_dim_nn, name='TargetCritic2', chkpt_dir=self.path)
        self.actor_replace_iter = 0
        '''network'''

        self.noise_OU = OUActionNoise(mu=np.zeros(self.action_dim_nn))
        self.noise_gaussian = GaussianNoise(mu=np.zeros(self.action_dim_nn))
        self.update_network_parameters()

        self.episode = 0
        self.reward = 0

        self.save_episode = []          # 保存的每一个回合的回合数
        self.save_reward = []           # 保存的每一个回合的奖励
        self.save_time = []
        self.save_average_reward = []   # 保存的每一个回合的平均时间的奖励
        self.save_successful_rate = []
        self.save_step = []             # 保存的每一步的步数
        self.save_stepreward = []       # 保存的每一步的奖励

    def choose_action_random(self):
        """
        :brief:     因为该函数与choose_action并列，所以输出也必须是[-1, 1]之间
        :return:    random action
        """
        return np.random.uniform(low=-1, high=1, size=self.action_dim_nn)

    def choose_action(self, state, is_optimal=False, sigma=1 / 3):
        self.actor.eval()  # 切换到测试模式
        t_state = torch.tensor(state, dtype=torch.float).to(self.actor.device)  # get the tensor of the state
        mu = self.actor(t_state).to(self.actor.device)  # choose action
        if is_optimal:
            mu_prime = mu
        else:
            mu_prime = mu + torch.tensor(self.noise_gaussian(sigma=sigma), dtype=torch.float).to(self.actor.device)  # action with gaussian noise
            # mu_prime = mu + torch.tensor(self.noise_OU(), dtype=torch.float).to(self.actor.device)             # action with OU noise
        self.actor.train()  # 切换回训练模式
        mu_prime_np = mu_prime.cpu().detach().numpy()
        return np.clip(mu_prime_np, -1, 1)  # 将数据截断在[-1, 1]之间

    def learn(self, is_reward_ascent=True, critic_random=True):     # TODO
        if self.memory.mem_counter < self.memory.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(is_reward_ascent=is_reward_ascent)
        state = torch.tensor(state, dtype=torch.float).to(self.critic1.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic1.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.critic1.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.critic1.device)
        done = torch.tensor(done, dtype=torch.float).to(self.critic1.device)

        state = torch.tensor(state, dtype=torch.float).to(self.critic2.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic2.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.critic2.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.critic2.device)
        done = torch.tensor(done, dtype=torch.float).to(self.critic2.device)

        self.target_actor.eval()            # PI'
        self.target_critic1.eval()          # Q1'
        self.critic1.eval()                 # Q1
        self.target_critic2.eval()          # Q2'
        self.critic2.eval()                 # Q2

        target_actions = self.target_actor.forward(new_state).to(self.critic1.device)                   # a' = PI'(s')
        '''动作正则化'''
        action_noise = torch.clip(torch.tensor(self.action_regularization(sigma=self.noise_policy)), -self.noise_clip, self.noise_clip).to(self.critic1.device)
        target_actions += action_noise
        '''动作正则化'''
        critic_value1_ = self.target_critic1.forward(new_state, target_actions)
        critic_value1 = self.critic1.forward(state, action)
        critic_value2_ = self.target_critic2.forward(new_state, target_actions)
        critic_value2 = self.critic2.forward(state, action)

        target = []
        '''取较小的Q'''
        for j in range(self.memory.batch_size):
            target.append(reward[j] + self.gamma * torch.minimum(critic_value1_[j], critic_value2_[j]) * done[j])

        target1 = torch.tensor(target).to(self.critic1.device)
        target1 = target1.view(self.memory.batch_size, 1)

        target2 = torch.tensor(target).to(self.critic2.device)
        target2 = target2.view(self.memory.batch_size, 1)       # target1 and target2 are identical

        '''critic1 training'''
        self.critic1.train()
        self.critic1.optimizer.zero_grad()
        critic_loss = func.mse_loss(target1, critic_value1)
        critic_loss.backward()
        self.critic1.optimizer.step()
        '''critic1 training'''

        '''critic2 training'''
        self.critic2.train()
        self.critic2.optimizer.zero_grad()
        critic_loss = func.mse_loss(target2, critic_value2)
        critic_loss.backward()
        self.critic2.optimizer.step()
        '''critic2 training'''

        self.policy_delay_iter += 1

        '''actor training, choose critic1 or critic2 randomly'''
        '''延迟更新'''
        if self.policy_delay_iter % self.policy_delay == 0:
            if critic_random:
                if random.randint(1, 2) == 1:
                    self.critic1.eval()
                    self.actor.optimizer.zero_grad()
                    mu = self.actor.forward(state)
                    self.actor.train()
                    actor_loss = -self.critic1.forward(state, mu)
                else:
                    self.critic2.eval()
                    self.actor.optimizer.zero_grad()
                    mu = self.actor.forward(state)
                    self.actor.train()
                    actor_loss = -self.critic2.forward(state, mu)
            else:
                self.critic1.eval()
                self.actor.optimizer.zero_grad()
                mu = self.actor.forward(state)
                self.actor.train()
                actor_loss = -self.critic1.forward(state, mu)
            actor_loss = torch.mean(actor_loss)
            actor_loss.backward()
            self.actor.optimizer.step()
        '''actor training, choose critic1 or critic2 randomly'''

        self.update_network_parameters()

    def update_network_parameters(self):
        """
        :return:        None
        """
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.critic1_tau) + param.data * self.critic1_tau)  # soft update
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.critic2_tau) + param.data * self.critic2_tau)  # soft update

        if self.policy_delay_iter % self.policy_delay == 0:
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.actor_tau) + param.data * self.actor_tau)  # soft update

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic1.save_checkpoint()
        self.target_critic2.save_checkpoint()

    def save_models_all(self):
        self.actor.save_all_net()
        self.critic1.save_all_net()
        self.critic2.save_all_net()
        self.target_actor.save_all_net()
        self.target_critic1.save_all_net()
        self.target_critic2.save_all_net()

    def load_models(self, path):
        """
        :brief:         only for test
        :param path:    file path
        :return:
        """
        print('...loading checkpoint...')
        self.actor.load_state_dict(torch.load(path + 'Actor_ddpg'))
        self.target_actor.load_state_dict(torch.load(path + 'TargetActor_ddpg'))
        self.critic1.load_state_dict(torch.load(path + 'Critic1_ddpg'))
        self.target_critic1.load_state_dict(torch.load(path + 'TargetCritic1_ddpg'))
        self.critic2.load_state_dict(torch.load(path + 'Critic2_ddpg'))
        self.target_critic2.load_state_dict(torch.load(path + 'TargetCritic2_ddpg'))

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
