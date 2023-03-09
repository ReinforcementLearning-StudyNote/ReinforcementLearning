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
			self.log_alpha = torch.zeros(1, requires_grad=True)
			self.alpha = self.log_alpha.exp()
			self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
		else:
			self.alpha = alpha
		'''SAC'''

		'''network'''
		self.actor = ProbActor(1e-4, self.state_dim_nn, self.action_dim_nn, name='Actor', chkpt_dir=self.path)
		self.critic = DualCritic(1e-3, self.state_dim_nn, self.action_dim_nn, name='Critic', chkpt_dir=self.path)
		# This critic contains double Q-net structure
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
