from common.common_func import *
from common.common_cls import *
from environment.config.xml_write import xml_cfg
from tensorboardX import SummaryWriter

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = True
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class Proximal_Policy_Optimization:
	def __init__(self,
				 actor_lr: float = 3e-4,
				 critic_lr: float = 1e-3,
				 gamma: float = 0.99,
				 K_epochs: int = 10,
				 eps_clip: float = 0.2,
				 action_std_init: float = 0.6,
				 buffer_size: int = 1200,
				 modelFileXML: str = '',
				 path: str = ''):
		"""
		@note:
		@param actor_lr:			actor learning rate
		@param critic_lr:			critic learning rate
		@param gamma:				discount factor
		@param K_epochs:			update policy for K epochs in one PPO update
		@param eps_clip:			clip parameter for PPO
		@param action_std_init:		starting std for action distribution (Multivariate Normal)
		@param modelFileXML:		model description file
		@param path:				path
		"""
		'''From rl_base'''
		# 这里使用连续动作，连续状态
		self.agentName, self.state_dim_nn, self.action_dim_nn, self.action_range = self.get_RLBase_from_XML(modelFileXML)
		'''From rl_base'''

		'''PPO'''
		self.gamma = gamma  # discount factor
		self.K_epochs = K_epochs  # 每隔 timestep_num 学习一次
		self.eps_clip = eps_clip
		self.action_std = action_std_init
		self.path = path
		# self.buffer = RolloutBuffer()
		self.buffer = RolloutBuffer(buffer_size, self.state_dim_nn, self.action_dim_nn)
		self.actor_lr = actor_lr
		self.critic_lr = critic_lr
		'''PPO'''

		'''networks'''
		self.policy = PPOActorCritic(self.state_dim_nn, self.action_dim_nn, action_std_init, name='PPOActorCritic', chkpt_dir=self.path)
		self.policy_old = PPOActorCritic(self.state_dim_nn, self.action_dim_nn, action_std_init, name='PPOActorCritic_old', chkpt_dir=self.path)
		self.policy_old.load_state_dict(self.policy.state_dict())

		self.optimizer = torch.optim.Adam([
			{'params': self.policy.actor.parameters(), 'lr': self.actor_lr},
			{'params': self.policy.critic.parameters(), 'lr': self.critic_lr}
		])
		self.loss = nn.MSELoss()
		self.device = device  # 建议使用 CPU 训练
		'''networks'''

		self.episode = 0
		self.reward = 0

		self.save_episode = []  # 保存的每一个回合的回合数
		self.save_reward = []  # 保存的每一个回合的奖励
		self.save_time = []
		self.save_step = []  # 保存的每一步的步数
		# self.writer = SummaryWriter(path)

	def set_action_std(self, new_action_std):
		self.action_std = new_action_std
		self.policy.set_action_std(new_action_std)
		self.policy_old.set_action_std(new_action_std)

	def decay_action_std(self, action_std_decay_rate, min_action_std):
		self.action_std = self.action_std - action_std_decay_rate
		self.action_std = round(self.action_std, 4)
		if self.action_std <= min_action_std:
			self.action_std = min_action_std
			print("setting actor output action_std to min_action_std : ", self.action_std)
		else:
			print("setting actor output action_std to : ", self.action_std)
		self.set_action_std(self.action_std)

	def choose_action_random(self):
		"""
		:brief:     因为该函数与choose_action并列，所以输出也必须是[-1, 1]之间
		:return:    random action
		"""
		return np.random.uniform(low=-1, high=1, size=self.action_dim_nn)

	def choose_action(self, state):
		with torch.no_grad():
			t_state = torch.FloatTensor(state).to(device)
			action, action_log_prob, state_val = self.policy_old.act(t_state)

		return action, t_state, action_log_prob, state_val

	def evaluate(self, state):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(device)
			action_mean = self.policy.actor(state)
		return action_mean.detach()

	def learn(self):
		"""
		@note: 	 network update
		@return: None
		"""
		'''1. Monte Carlo estimate of returns'''
		rewards = []
		discounted_reward = 0
		for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + self.gamma * discounted_reward
			rewards.insert(0, discounted_reward)

		'''2. Normalizing the rewards'''
		rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

		'''3. convert numpy to tensor'''
		with torch.no_grad():
			old_states = torch.FloatTensor(self.buffer.states).detach().to(self.device)
			old_actions = torch.FloatTensor(self.buffer.actions).detach().to(self.device)
			old_log_probs = torch.FloatTensor(self.buffer.log_probs).detach().to(self.device)
			old_state_values = torch.FloatTensor(self.buffer.state_values).detach().to(self.device)

		'''4. calculate advantages'''
		advantages = rewards.detach() - old_state_values.detach()

		'''5. Optimize policy for K epochs'''
		for _ in range(self.K_epochs):
			'''5.1 Evaluating old actions and values'''
			log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

			'''5.2 match state_values tensor dimensions with rewards tensor'''
			state_values = torch.squeeze(state_values)

			'''5.3 Finding the ratio (pi_theta / pi_theta__old)'''
			ratios = torch.exp(log_probs - old_log_probs.detach())

			'''5.4 Finding Surrogate Loss'''
			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

			'''5.5 final loss of clipped objective PPO'''
			loss = -torch.min(surr1, surr2) + 0.5 * self.loss(state_values, rewards) - 0.01 * dist_entropy

			'''5.6 take gradient step'''
			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()

		'''6. Copy new weights into old policy'''
		self.policy_old.load_state_dict(self.policy.state_dict())

	def save_models(self):
		self.policy.save_checkpoint()
		self.policy_old.save_checkpoint()

	def save_models_all(self):
		self.policy.save_all_net()
		self.policy_old.save_all_net()

	def load_models(self, path):
		"""
		:brief:         only for test
		:param path:    file path
		:return:
		"""
		print('...loading checkpoint...')
		self.policy.load_state_dict(torch.load(path + 'Policy_ppo'))
		self.policy_old.load_state_dict(torch.load(path + 'Policy_old_ppo'))

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

	def PPO_info(self):
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
