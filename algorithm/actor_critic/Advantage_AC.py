import pandas as pd
import torch

from common.common_func import *
from common.common_cls import *
from environment.config.xml_write import xml_cfg

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = False
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class Advantage_AC:
	def __init__(self, gamma: float = 0.99, trajectory_num: int = 10, modelFileXML: str = '', path: str = ''):
		'''From rl_base'''
		# DDPG 要求智能体状态必须是连续的，动作必须连续的
		self.agentName, self.state_dim_nn, self.action_dim_nn, self.action_range = \
			self.get_RLBase_from_XML(modelFileXML)
		# agentName:            the name of the agent
		# state_dim_nn:         the dimension of the neural network input
		# action_dim_nn:        the dimension of the neural network output
		# action_range:         the range of physical action
		'''From rl_base'''

		'''A2C'''
		self.gamma = gamma						# discount factor
		self.trajectory_num = trajectory_num
		self.path = path
		'''A2C'''

		'''networks'''
		self.actor = DiscreteActor(1e-4, self.state_dim_nn, self.action_dim_nn, name='DiscreteActor', chkpt_dir=self.path)
		self.critic = Critic(1e-4, self.state_dim_nn, self.action_dim_nn, name='Critic', chkpt_dir=self.path)
		'''networks'''

		self.episode = 0
		self.reward = 0

		self.save_episode = []  # 保存的每一个回合的回合数
		self.save_reward = []  # 保存的每一个回合的奖励
		self.save_time = []
		self.save_step = []  # 保存的每一步的步数
		self.save_stepreward = []  # 保存的每一步的奖励

	def action_index_find(self, action):
		"""
		@param action:		the index of the action space
		@return:			real physical action
		"""
		linear_action = []
		for i in range(self.action_dim_nn):
			a = self.
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

	def saveData_EpisodeReward(self, episode, time, reward, is2file=False, filename='EpisodeReward.csv'):
		if is2file:
			data = pd.DataFrame({
				'episode': self.save_episode,
				'time': self.save_time,
				'reward': self.save_reward
			})
			data.to_csv(self.path + filename, index=False, sep=',')
		else:
			self.save_episode.append(episode)
			self.save_time.append(time)
			self.save_reward.append(reward)
