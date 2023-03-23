import cv2 as cv
import numpy as np
from common.common_cls import *
import torch.multiprocessing as mp
from multiprocessing import shared_memory
import pandas as pd

"""
	Note: CPU is recommended for DPPO.
"""


class Worker(mp.Process):
	def __init__(self,
				 env,
				 name: str,
				 index: int,
				 global_policy: PPOActorCritic,
				 global_permit: mp.Value,
				 share_memory: shared_memory,
				 buffer_size: int,
				 gamma:float,
				 action_std_decay_freq: int,
				 action_std_decay_rate: float,
				 min_action_std: float
				 ):
		"""
		@param env:				RL environment
		@param name:			name of the worker
		@param index:			index of the worker
		@param global_flag:		forget
		@param global_permit:	permission receives from the chief, different workers need different permissions
		@param share_memory:	shared buffer
		@param buffer_size:		buffer size
		@param gamma:			discount factor
		"""
		super(Worker, self).__init__()
		self.env = env
		self.index = index
		self.name = name
		self.gamma = gamma
		self.buffer_size = buffer_size
		self.global_policy = global_policy
		self.global_permit = global_permit
		'''share_buffer 是一个 buffer_size 行, (状态维度 + 动作维度 + log_prob + sv + r) 列的 numpy 矩阵'''
		buffer = np.zeros((buffer_size, self.env.state_dim + self.env.action_dim + 3), dtype=np.float)
		self.share_buffer = np.ndarray(buffer.shape, self.buffer.dtype, share_memory.buf)
		self.local_policy = PPOActorCritic(self.env.state_dim, self.env.action_dim, 0.8, '', '')
		self.action_std = 0.8
		self.timestep = 0
		self.action_std_decay_freq = action_std_decay_freq
		self.action_std_decay_rate = action_std_decay_rate
		self.min_action_std = min_action_std

	def choose_action(self, state):
		"""
		@param state:	state of the env
		@return:		action (tensor), state (tensor), the log-probability of action (tensor), and state value (tensor)
		"""
		with torch.no_grad():
			t_state = torch.FloatTensor(state).to(device)
			action, action_log_prob, state_val = self.local_policy.act(t_state)
		return action, t_state, action_log_prob, state_val

	def action_linear_trans(self, action):
		# the action output
		linear_action = []
		for i in range(self.env.action_dim):
			a = min(max(action[i], -1), 1)
			maxa = self.env.action_range[i][1]
			mina = self.env.action_range[i][0]
			k = (maxa - mina) / 2
			b = (maxa + mina) / 2
			linear_action.append(k * a + b)
		return linear_action

	def set_action_std(self, new_action_std):
		self.action_std = new_action_std
		self.local_policy.set_action_std(new_action_std)

	def decay_action_std(self, action_std_decay_rate, min_action_std):
		self.action_std = self.action_std - action_std_decay_rate
		self.action_std = round(self.action_std, 4)
		if self.action_std <= min_action_std:
			self.action_std = min_action_std
			print("setting actor output action_std to min_action_std : ", self.action_std)
		else:
			print("setting actor output action_std to : ", self.action_std)
		self.set_action_std(self.action_std)

	def load_global_policy(self):
		self.local_policy.load_state_dict(self.global_policy.state_dict())	# 加载网络参数
		self.local_policy.to('cpu')		# 将探索放在cpu里面
		self.set_action_std(self.action_std)	# 设置探索方差

	def package_rewards(self, buffer_r, buffer_done):
		"""
		@note:					r 和 done 需要对应起来
		@param buffer_r:		the immediate reward of trajectories
		@param buffer_done:		the done flag of trajectories
		@return:				the cumulative reward of trajectories
		"""
		# rewards = []
		_l = len(buffer_r)
		# rewards = np.zeros(_l)
		discounted_reward = 0
		_i = 0
		for reward, is_terminal in zip(reversed(buffer_r), reversed(buffer_done)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + (self.gamma * discounted_reward)
			self.share_buffer[_l - i - 1, -1] = discounted_reward
			# rewards.insert(0, discounted_reward)

	def run(self):
		"""
		@note:		DPPO2 中，worker 只负责产生数据，并不负责训练
				global_permit 为1时，开始收集数据，手机完成时，置 0。主进程检测到所有标志位都为0时，开始学习，学习完毕，将该标志位置 1。
				换言之，子进程只有将 global_permit 置 0 的权限；主进程只有将 global_permit 置 1 的权限
		@return:
		"""
		while True:
			if self.global_permit.value == 1:
				index = 0
				self.load_global_policy(global_polocy)
				'''开始搜集数据'''
				while index < self.buffer_size:
					env.reset_random()
					while not env.is_terminal:
						self.env.current_state = self.env.next_state.copy()
						action_from_actor, s, a_log_prob, s_value = self.choose_action(self.env.current_state)  # 返回三个没有梯度的 tensor
						action_from_actor = action_from_actor.numpy()
						action = self.action_linear_trans(action_from_actor.flatten())	# flatten 也有去掉多余括号的功能
						self.env.step_update(action)
						'''状态维度 + 动作维度 + log_prob + sv + r'''
						self.share_buffer[index, 0: self.env.state_dim] = self.env.current_state.copy()
						self.share_buffer[index, self.env.state_dim: self.env.state_dim + self.env.action_dim] = action_from_actor.copy()
						self.share_buffer[index, -3] = a_log_prob.item()	# 因为就 1 个数，所以可以用 item
						self.share_buffer[index, -2] = s_value.item()		# 因为就 1 个数，所以可以用 item
						index += 1
						self.timestep += 1
						if self.timestep % self.action_std_decay_freq == 0:
							self.decay_action_std(self.action_std_decay_rate, self.min_action_std)
						if index == self.buffer_size:
							break
				self.package_rewards()		# 将 reward 放到 buffer 的最后一列
				self.global_permit.value = 0		# 数据采集结束，将标志置 0
			else:
				'''等待 chief 进程发送允许标志'''
				pass


class Distributed_PPO2:
	def __init__(self,
				 env,
				 actor_lr: float = 3e-4,
				 critic_lr: float = 1e-3,
				 num_of_pro: int = 5,
				 path: str = '',
				 action_std_decay_freq: int = int(5e4),
				 action_std_decay_rate: float = 0.05,
				 min_action_std: float = 0.1,
				 total_tr_cnt: int = 5000,
				 k_epo: int = 250
				 ):
		self.env = env

		'''PPO'''
		self.actor_lr = actor_lr
		self.critic_lr = critic_lr
		self.global_policy = PPOActorCritic(env.state_dim, env.action_dim, 0, 'Global_policy_ppo', path)
		self.global_policy.share_memory()		# 全局
		self.optimizer = torch.optim.Adam([
			{'params': self.global_policy.actor.parameters(), 'lr': self.actor_lr},
			{'params': self.global_policy.critic.parameters(), 'lr': self.critic_lr}
		])
		self.loss = nn.MSELoss()
		self.device = 'cuda:0'		# 建议使用 gpu
		self.action_std_decay_freq = action_std_decay_freq
		self.action_std_decay_rate = action_std_decay_rate
		self.min_action_std = min_action_std
		self.total_tr_cnt = g_tr_cnt		# 网络一共训练多少轮
		self.g_tr_cnt = 0
		self.k_epo = k_epo				# 每一轮训练多少次
		self.gamma = 0.99
		'''PPO'''

		'''multi process'''
		self.num_of_pro = num_of_pro
		self.buffer_size = int(env.timeMax / env.dt) * 2
		ref_buffer = np.zeros((self.buffer_size, self.env.state_dim + self.env.action_dim + 3), dtype=np.float)		# 共享内存的参考变量
		self.g_buf = []
		self.share_memory = []
		self.global_permit = []
		for _ in range(self.num_of_pro):
			share_mem = shared_memory.SharedMemory(create=True, name='buffer' + str(i), size=ref_buffer.nbytes)		# 创建共享内存
			buffer = np.ndarray(ref_buffer.shape, ref_buffer.dtype, share_mem.buf)		# 创建一个变量，指向共享内存
			self.g_buf.append(buffer)				# 共享内存
			self.share_memory.append(share_mem)				# 指向共享内存的变量
			self.global_permit.append(mp.Value('i', 0))		# 全局标志位
		self.processes = [mp.Process(target=self.global_learn, args=())]	# training process
		'''multi process'''

		'''data buffer'''
		self.d_buf_s = np.zeros((self.buffer_size * self.num_of_pro, self.env.state_dim))
		self.d_buf_a = np.zeros((self.buffer_size * self.num_of_pro, self.env.action_dim))
		self.d_buf_lg_prob = np.zeros(self.buffer_size * self.num_of_pro)
		self.d_buf_vs = np.zeros(self.buffer_size * self.num_of_pro)
		self.d_buf_r = np.zeros(self.buffer_size * self.num_of_pro)
		'''data buffer'''

	def add_worker(self):
		"""
		@return:	none
		"""
		'''
			env,
			name: str,
			index: int,
			global_policy: PPOActorCritic,
			global_permit: mp.Value,
			share_memory: shared_memory,
			buffer_size: int,
			gamma:float,
			action_std_decay_freq: int,
			action_std_decay_rate: float,
			min_action_std: float
		'''
		for i in range(self.num_of_pro):
			worker = Worker(env=env,
							name='worker' + str(i),
							index=i,
							global_policy=self.global_policy,
							global_permit=self.global_permit[i],
							share_memory=self.share_memory[i],
							buffer_size=self.buffer_size,
							gamma=self.gamma,
							action_std_decay_freq=self.action_std_decay_freq,
							action_std_decay_rate=self.action_std_decay_rate,
							min_action_std=self.min_action_std
							)
			self.processes.append(worker)

	def start_multi_process(self):
		for p in self.processes:
			p.start()
			p.join(0.5)

	def train_permit(self):
		for i in range(self.num_of_pro):
			if self.global_permit[i].value == 1:
				return False
		return True

	def permit_exploration(self):
		for i in range(self.num_of_pro):
			self.global_permit[i].value = 1		# 允许子进程开始搜集数据

	def evaluate(self, state):
		with torch.no_grad():
			t_state = torch.FloatTensor(state).to(self.device)
			action_mean = self.global_policy.actor(t_state)
		return action_mean.detach()

	def action_linear_trans(self, action):
		# the action output
		linear_action = []
		for i in range(self.env.action_dim):
			a = min(max(action[i], -1), 1)
			maxa = self.env.action_range[i][1]
			mina = self.env.action_range[i][0]
			k = (maxa - mina) / 2
			b = (maxa + mina) / 2
			linear_action.append(k * a + b)
		return linear_action

	def global_learn(self):
		while self.g_tr_cnt < self.total_tr_cnt:
			if self.train_permit():
				'''1. 先把数据整理'''
				for i in range(self.num_of_pro):
					# s, a, lg, sv, r
					self.d_buf_s[i * self.buffer_size: (i + 1) * self.buffer_size] = self.g_buf[i][:, 0:self.env.state_dim]
					self.d_buf_a[i * self.buffer_size: (i + 1) * self.buffer_size] = self.g_buf[i][:, self.env.state_dim, self.env.state_dim + self.env.action_dim]
					self.d_buf_lg_prob[i * self.buffer_size: (i + 1) * self.buffer_size] = self.g_buf[i][:, -3]
					self.d_buf_vs[i * self.buffer_size: (i + 1) * self.buffer_size] = self.g_buf[i][:, -2]
					self.d_buf_r[i * self.buffer_size: (i + 1) * self.buffer_size] = self.g_buf[i][:, -1]
				'''2. 把数据放到 tensor 中'''
				with torch.no_grad():
					old_states = torch.FloatTensor(self.d_buf_s).detach().to(self.device)
					old_actions = torch.FloatTensor(self.d_buf_a).detach().to(self.device)
					old_log_probs = torch.FloatTensor(self.d_buf_lg_prob).detach().to(self.device)
					old_state_values = torch.FloatTensor(self.d_buf_vs).detach().to(self.device)
					rewards = torch.FloatTensor(self.d_buf_r).detach().to(self.device)
					rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)	# 奖励归一化

				'''3. 开始学习'''
				advantages = rewards.detach() - old_state_values.detach()
				for _ in range(self.k_epo):
					'''5.1 Evaluating old actions and values'''
					self.global_policy.to(self.device)	# 现将模型放到 gpu
					log_probs, state_values, dist_entropy = self.global_policy.evaluate(old_states, old_actions)

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
				self.g_tr_cnt += 1			# 全局学习次数加一
				if self.g_tr_cnt % 10 == 0:
					self.save_models()
					test_num = 5
					print('evaluating...')
					for _ in range(test_num):
						self.env.reset_random()
						while not self.env.is_terminal:
							self.env.current_state = self.env.next_state.copy()
							action_from_actor = self.evaluate(self.env.current_state)
							action_from_actor = action_from_actor.numpy()
							action = self.action_linear_trans(action_from_actor.flatten())  # 将动作转换到实际范围上
							self.env.step_update(action)  # 环境更新的action需要是物理的action
							r += self.env.reward
							self.env.show_dynamic_image(isWait=False)  # 画图
					cv.destroyAllWindows()
				self.permit_exploration()	# 允许收集数据
			else:
				'''啥也不干，等着采集数据'''
				pass
		print('Training termination...')
		self.clean()

	def save_models(self):
		self.global_policy.save_checkpoint()

	def save_models_all(self):
		self.global_policy.save_all_net()

	@atexit.register
	def clean(self):
		for _share in self.share_memory:
			_share.close()
			_share.unlink()
