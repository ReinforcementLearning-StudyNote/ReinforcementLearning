import numpy as np
import pandas as pd
import torch

from common.common_func import *
from common.common_cls import *
from environment.config.xml_write import xml_cfg

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = True
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


class Advantage_AC:
	def __init__(self, gamma: float = 0.99, timestep_num: int = 10, modelFileXML: str = '', path: str = ''):
		"""
		@param gamma:
		@param timestep_num:
		@param modelFileXML:
		@param path:
		"""
		'''From rl_base'''
		# A2C 要求智能体状态必须是连续的，动作必须离散的
		self.agentName, self.state_dim_nn, self.action_dim_nn, self.action_num, self.action_space_nn, self.action_range = \
			self.get_RLBase_from_XML(modelFileXML)
		'''From rl_base'''

		'''A2C'''
		self.gamma = gamma  # discount factor
		self.timestep_num = timestep_num  # 每隔 timestep_num 学习一次
		self.path = path
		'''A2C'''

		'''networks'''
		self.actor = SofemaxActor(1e-4, self.state_dim_nn, self.action_dim_nn, name='SofemaxActor', chkpt_dir=self.path)
		self.critic = Critic(1e-4, self.state_dim_nn, self.action_dim_nn, name='Critic', chkpt_dir=self.path)
		self.device = device  # 建议使用 CPU 训练
		'''networks'''

		self.episode = 0
		self.reward = 0

		self.save_episode = []  # 保存的每一个回合的回合数
		self.save_reward = []  # 保存的每一个回合的奖励
		self.save_time = []
		self.save_step = []  # 保存的每一步的步数
		self.save_stepreward = []  # 保存的每一步的奖励

	def choose_action_random(self):
		"""
		@note:		随机选择每一维动作的索引，记住是索引
		@return:	random action
		"""
		_a = []
		for _num in self.action_num:
			_a.append(np.random.randint(low=0, high=_num))
		return np.array(_a)

	def choose_action(self, state, deterministic=False):
		t_state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0).to(self.device)
		p = torch.squeeze(self.actor(t_state).cpu().detach()).numpy()  # probability distribution(numpy), n 个分布列
		p = np.split(p, [np.sum(self.action_num[0:i]) for i in range(self.action_dim_nn - 1)])
		_a = []
		_lg_prob = []
		if deterministic:
			for i in range(self.action_dim_nn):
				index = np.argmax(p[i])
				_a.append(index)  # Select the action with the highest probability
				_lg_prob.append(np.log(p[i][index]))
			return np.array(_a), np.array(_lg_prob)
		else:
			for i in range(self.action_dim_nn):
				index = np.random.choice(range(self.action_num[i]), p=p[i] / np.sum(p[i]))
				_a.append(index)
				_lg_prob.append(np.log(p[i][index]))
			return np.array(_a), np.array(_lg_prob)

	def evaluate(self, state):
		return self.choose_action(state, True)

	def learn(self, r: np.ndarray, lg_prob: np.ndarray, vs: np.ndarray):
		t_vs = torch.FloatTensor(vs).squeeze().to(self.device)  # 这里的a是索引号，必须是int
		t_lg_prob = torch.FloatTensor(lg_prob).to(self.device)
		R = 0
		traj_r = []
		for _r in r[::-1]:
			R = _r + self.gamma * R
			traj_r.insert(0, R)
		with torch.no_grad():
			traj_r = torch.FloatTensor(traj_r).squeeze().to(self.device)
			advantages = traj_r - t_vs

		# Update actor
		t_lg_prob.requires_grad_()
		t_vs.requires_grad_()
		actor_loss = -torch.sum(t_lg_prob * advantages)
		self.actor.optimizer.zero_grad()
		actor_loss.backward()
		self.actor.optimizer.step()

		# Update critic
		critic_loss = func.smooth_l1_loss(t_vs, traj_r)
		self.critic.optimizer.zero_grad()
		critic_loss.backward()
		self.critic.optimizer.step()

	def save_models(self):
		self.actor.save_checkpoint()
		self.critic.save_checkpoint()

	def save_models_all(self):
		self.actor.save_all_net()
		self.critic.save_all_net()

	def load_models(self, path):
		"""
		:brief:         only for test
		:param path:    file path
		:return:
		"""
		print('...loading checkpoint...')
		self.actor.load_state_dict(torch.load(path + 'Actor_A2C'))
		self.critic.load_state_dict(torch.load(path + 'Critic_A2C'))

	def load_actor_optimal(self, path, file):
		print('...loading optimal...')
		self.actor.load_state_dict(torch.load(path + file))

	def A2C_info(self):
		print('agent name:', self.agentName)
		print('state_dim:', self.state_dim_nn)
		print('action_dim:', self.action_dim_nn)
		print('action_num', self.action_num)
		print('action_space_nn', self.action_space_nn)
		print('action_range:', self.action_range)

	def get_RLBase_from_XML(self, filename):
		"""
		@note:	A2C 是一个离散动作空间算法，所以其参数定义稍有不同
			state_dim_nn: 		状态的维度，与其余算法一样
			action_dim_nn:		动作的维度，并不是动作空间的数量，比如双轮小车，action_dim_nn就是2，分别是左轮和右轮的转速
			action_num:			每一维度动作空间的数量
			action_space_nn:	动作空间，对于离散动作，每一维动作都需要有一个取值的集合，比如力输出从0到10，步长0.5，那么这一维度的动作空间就是[0 : 0.5 : 10]，这个概念要区分开
			action_range:		每一维动作的取值上下界，对于A2C算法，这个值是没有用的，不过为了保持完整，予以保留
		@param filename: name of the *.xml file
		@return:
		"""
		rl_base, agentName = self.load_rl_basefromXML(filename=filename)
		state_dim_nn = int(rl_base['state_dim'])
		action_dim_nn = int(rl_base['action_dim'])
		action_num = str2list2(rl_base['action_num'])
		action_space_nn = str2list(rl_base['action_space'])
		action_range = str2list(rl_base['action_range'])
		return agentName, state_dim_nn, action_dim_nn, action_num, action_space_nn, action_range

	@staticmethod
	def load_rl_basefromXML(filename: str) -> (dict, str):
		"""
		@note:				从模型文件中加载数据到 A2C 中
		@param filename:	模型文件
		@return:			数据字典
		"""
		root = xml_cfg().XML_Load(filename)
		return xml_cfg().XML_GetTagValue(node=xml_cfg().XML_FindNode(nodename='RL_Base', root=root)), root.attrib['name']

	def action_index_find(self, action):
		"""
		@param action:		the index of the action space
		@return:			real physical action
		"""
		_a = []
		for i in range(self.action_dim_nn):
			_a.append(self.action_space_nn[i][action[i]])
		return np.array(_a)

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
