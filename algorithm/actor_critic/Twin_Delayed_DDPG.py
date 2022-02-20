from environment.config.xml_write import xml_cfg
from common.common import *
import pandas as pd

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = False
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class Twin_Delayed_DDPG:
    def __init__(self,
                 gamma: float = 0.9,
                 actor_update_freq: int = 5,
                 critic1_soft_update: float = 1e-2,
                 critic2_soft_update: float = 1e-2,
                 memory_capacity: int = 5000,
                 batch_size: int = 64,
                 modelFileXML: str = '',
                 path: str = ''):
        """
        :param gamma:                       折扣因子
        :param actor_update_freq:           actor更新频率，critic每更新actor_update_freq次，actor直接hard更新一次
        :param critic1_soft_update:         critic1网络更新率
        :param critic2_soft_update:         critic2网络更新率
        :param memory_capacity:             经验池容量
        :param batch_size:                  batch大小
        :param modelFileXML:                模型描述文件
        :param path:                        保存路径
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
        # self.actor_lr = actor_learning_rate
        # self.critic_lr = critic_learning_rate
        self.actor_update_freq = actor_update_freq
        self.critic1_soft_update = critic1_soft_update
        self.critic2_soft_update = critic2_soft_update
        self.memory = ReplayBuffer(memory_capacity, batch_size, self.state_dim_nn, self.action_dim_nn)
        self.path = path
        '''DDPG'''

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

    def learn(self, is_reward_ascent=True):     # TODO
        if self.memory.mem_counter < self.memory.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(is_reward_ascent=is_reward_ascent)
        state = torch.tensor(state, dtype=torch.float).to(self.critic.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.critic.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.critic.device)
        done = torch.tensor(done, dtype=torch.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic1.eval()
        self.critic1.eval()
        self.target_critic2.eval()
        self.critic2.eval()

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