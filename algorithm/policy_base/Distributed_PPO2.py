import time

import numpy as np

from common.common_cls import *
import torch.multiprocessing as mp
from multiprocessing import shared_memory
import cv2 as cv

# import pandas as pd

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
				 collecting_count: mp.Value,
				 ref_np: tuple,
				 share_memory: shared_memory.SharedMemory,
				 buffer_size: int,
				 gamma: float,
				 action_std_decay_freq: int,
				 action_std_decay_rate: float,
				 min_action_std: float,
				 action_std_init: float):
		"""
		@param env:				RL environment
		@param name:			name of the worker
		@param index:			index of the worker
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
		self.collecting_count = collecting_count
		self.collect = 0
		self.sm = shared_memory.SharedMemory(name=share_memory.name)  # 共享内存的连接，不创建，因为能跑到这一行，就说明已经创建完了
		# self.share_buffer = np.ndarray(ref_np.shape, ref_np.dtype, self.sm.buf)
		self.share_buffer = np.ndarray(ref_np[0], ref_np[1], self.sm.buf)
		print('fuck',self.share_buffer)
		self.local_policy = PPOActorCritic(self.env.state_dim, self.env.action_dim, 0.8, '', '')
		self.action_std = action_std_init
		self.timestep = 0
		self.action_std_decay_freq = action_std_decay_freq
		self.action_std_decay_rate = action_std_decay_rate
		self.min_action_std = min_action_std
		self.device = 'cpu'

	def choose_action(self, state):
		"""
		@param state:	state of the env
		@return:		action (tensor), state (tensor), the log-probability of action (tensor), and state value (tensor)
		"""
		with torch.no_grad():
			t_state = torch.FloatTensor(state).to(self.device)
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
		self.local_policy.load_state_dict(self.global_policy.state_dict())  # 加载网络参数
		self.local_policy.to('cpu')  # 将探索放在cpu里面
		self.set_action_std(self.action_std)  # 设置探索方差

	def package_rewards(self, buffer_r, buffer_done):
		"""
		@note:					r 和 done 需要对应起来
		@param buffer_r:		the immediate reward of trajectories
		@param buffer_done:		the done flag of trajectories
		@return:				the cumulative reward of trajectories
		"""
		# rewards = []
		# _l = len(buffer_r)
		# rewards = np.zeros(_l)
		discounted_reward = 0
		# 倒过来存储，计算值函数是从末端往回推
		for reward, is_terminal, _index in zip(reversed(buffer_r), reversed(buffer_done), reversed(range(self.buffer_size))):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + self.gamma * discounted_reward
			self.share_buffer[_index, -1] = discounted_reward
		# rewards.insert(0, discounted_reward)

	def clean(self):
		self.sm.close()

	def run(self):
		"""
		@note:		DPPO2 中，worker 只负责产生数据，并不负责训练
				global_permit 为1时，开始收集数据，手机完成时，置 0。主进程检测到所有标志位都为0时，开始学习，学习完毕，将该标志位置 1。
				换言之，子进程只有将 global_permit 置 0 的权限；主进程只有将 global_permit 置 1 的权限
		@return:
		"""
		try:
			while True:
				if self.global_permit.value == 1:
					index = 0
					'''
						a). 从 global 中加载模型
						b). 将模型放到 cpu 里面
						c). 设置模型探索的 std
					'''
					self.load_global_policy()
					print('========= EPISODE START =========')
					print('Collecting data...')
					_immediate_r = np.ones(self.buffer_size)
					_done = np.ones(self.buffer_size)
					while index < self.buffer_size:
						self.env.reset_random()
						while not self.env.is_terminal:
							self.env.current_state = self.env.next_state.copy()
							action_from_actor, s, a_log_prob, s_value = self.choose_action(self.env.current_state)  # 返回三个没有梯度的 tensor
							action_from_actor = action_from_actor.numpy()
							action = self.action_linear_trans(action_from_actor.flatten())	# flatten 也有去掉多余括号的功能
							self.env.step_update(action)
							# '''状态, 动作, log_prob, sv, r, 余下两个是即时奖励和done'''
							# self.share_buffer[index, 0: self.env.state_dim] = self.env.current_state.copy()		# 状态
							# self.share_buffer[index, self.env.state_dim: self.env.state_dim + self.env.action_dim] = action_from_actor.copy()	# 动作
							# self.share_buffer[index, -3] = a_log_prob.item()	# 对数概率，因为就 1 个数，所以可以用 item
							# self.share_buffer[index, -2] = s_value.item()		# 状态值函数，因为就 1 个数，所以可以用 item
							# _immediate_r[index] = self.env.reward
							# _done[index] = 1.0 if self.env.is_terminal else 0.0
							# '''状态, 动作, log_prob, sv, r,  余下两个是即时奖励和done'''

							'''状态, 动作, log_prob, sv, r, 余下两个是即时奖励和done'''
							self.share_buffer[index, 0: self.env.state_dim] = [9, 8]
							self.share_buffer[index, self.env.state_dim: self.env.state_dim + self.env.action_dim] = [2]
							self.share_buffer[index, -3] = 1
							self.share_buffer[index, -2] = 1.1

							_immediate_r[index] = self.env.reward
							_done[index] = 1.0 if self.env.is_terminal else 0.0
							'''状态, 动作, log_prob, sv, r,  余下两个是即时奖励和done'''
							index += 1
							self.timestep += 1
							if self.timestep % self.action_std_decay_freq == 0:
								self.decay_action_std(self.action_std_decay_rate, self.min_action_std)
							if index == self.buffer_size:
								break
					print('艹', self.share_buffer)
					self.share_buffer = np.ndarray(self.share_buffer.shape, self.share_buffer.dtype, self.sm.buf)
					print('我 TMD 再改一下', self.share_buffer)
					print('Finish collecting data: {}'.format(self.name))
					print('----------')
					self.package_rewards(_immediate_r, _done)  # 将 reward 放到 buffer 的最后一列
					self.global_permit.value = 0  # 数据采集结束，将标志置 0
				else:
					'''等待 chief 进程发送允许标志'''
					if self.collect == self.collecting_count.value:
						self.clean()
						break
		except:
			self.clean()


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
				 action_std_init: float = 0.8,
				 eps_clip: float = 0.2,
				 total_tr_cnt: int = 5000,
				 k_epo: int = 250):
		self.env = env
		self.path = path

		'''PPO'''
		self.actor_lr = actor_lr
		self.critic_lr = critic_lr
		self.global_policy = PPOActorCritic(env.state_dim, env.action_dim, 0, 'Global_policy_ppo', path)
		self.global_policy.share_memory()  # 全局
		self.optimizer = torch.optim.Adam([
			{'params': self.global_policy.actor.parameters(), 'lr': self.actor_lr},
			{'params': self.global_policy.critic.parameters(), 'lr': self.critic_lr}
		])
		self.loss = nn.MSELoss()
		self.device = 'cuda:0'  # 建议使用 gpu
		self.action_std_decay_freq = action_std_decay_freq
		self.action_std_decay_rate = action_std_decay_rate
		self.min_action_std = min_action_std
		self.action_std_init = action_std_init
		self.eps_clip = eps_clip
		self.total_tr_cnt = total_tr_cnt  # 网络一共训练多少轮
		self.g_tr_cnt = 0
		self.k_epo = k_epo  # 每一轮训练多少次
		self.gamma = 0.99
		'''PPO'''

		'''multi process'''
		self.num_of_pro = num_of_pro
		# self.buffer_size = int(env.timeMax / env.dt) * 2
		self.buffer_size = 10	# TODO
		self.g_buf = []
		self.share_memory = []
		self.global_permit = []
		for i in range(self.num_of_pro):
			ref_buffer = np.zeros((self.buffer_size, self.env.state_dim + self.env.action_dim + 3), dtype=np.float32)  # 共享内存的参考变量
			share_mem = shared_memory.SharedMemory(create=True, size=ref_buffer.nbytes)  # 创建共享内存
			buffer = np.ndarray(ref_buffer.shape, ref_buffer.dtype, share_mem.buf)
			buffer[:] = ref_buffer[:]
			buffer[0, 0] = 21
			self.g_buf.append(buffer)  # 共享内存
			self.share_memory.append(share_mem)  # 指向共享内存的变量
			self.global_permit.append(mp.Value('i', 1))  # 全局标志位，初始模式设置为收集模式

		self.processes = [mp.Process(target=self.global_learn, args=())]  # training process
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
		for i in range(self.num_of_pro):
			print((self.g_buf[0].shape, self.g_buf[0].dtype))
			worker = Worker(env=self.env,
							name='worker' + str(i),
							index=i,
							global_policy=self.global_policy,
							global_permit=self.global_permit[i],
							collecting_count=self.total_tr_cnt,
							ref_np=(self.g_buf[0].shape, self.g_buf[0].dtype),
							share_memory=self.share_memory[i],
							buffer_size=self.buffer_size,
							gamma=self.gamma,
							action_std_decay_freq=self.action_std_decay_freq,
							action_std_decay_rate=self.action_std_decay_rate,
							min_action_std=self.min_action_std,
							action_std_init=self.action_std_init)
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
			self.global_permit[i].value = 1  # 允许子进程开始搜集数据

	def forbid_exploration(self):
		for i in range(self.num_of_pro):
			self.global_permit[i].value = 0  # 禁止子进程开始搜集数据

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

	def print_data_in_worker(self, i):
		print('worker ', i)
		print('state: ', self.d_buf_s[i * self.buffer_size: (i + 1) * self.buffer_size])
		print('action: ', self.d_buf_a[i * self.buffer_size: (i + 1) * self.buffer_size])
		print('lg_prob: ', self.d_buf_lg_prob[i * self.buffer_size: (i + 1) * self.buffer_size])
		print('vs: ', self.d_buf_vs[i * self.buffer_size: (i + 1) * self.buffer_size])
		print('r: ', self.d_buf_r[i * self.buffer_size: (i + 1) * self.buffer_size])

	def global_learn(self):
		try:
			while True:
				if self.train_permit():
					print('Starting training')
					time.sleep(5)
					'''1. 先把数据整理'''
					for i in range(self.num_of_pro):
						print('??', self.g_buf[i])
						# s, a, lg, sv, r
						self.d_buf_s[i * self.buffer_size: (i + 1) * self.buffer_size] = self.g_buf[i][:, 0:self.env.state_dim]
						self.d_buf_a[i * self.buffer_size: (i + 1) * self.buffer_size] = self.g_buf[i][:, self.env.state_dim: self.env.state_dim + self.env.action_dim]
						self.d_buf_lg_prob[i * self.buffer_size: (i + 1) * self.buffer_size] = self.g_buf[i][:, -3]
						self.d_buf_vs[i * self.buffer_size: (i + 1) * self.buffer_size] = self.g_buf[i][:, -2]
						self.d_buf_r[i * self.buffer_size: (i + 1) * self.buffer_size] = self.g_buf[i][:, -1]
						self.print_data_in_worker(i)

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
						self.global_policy.to(self.device)	# 现将模型放到 gpu
						log_probs, state_values, dist_entropy = self.global_policy.evaluate(old_states, old_actions)
						state_values = torch.squeeze(state_values)
						ratios = torch.exp(log_probs - old_log_probs.detach())
						surr1 = ratios * advantages
						surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
						loss = -torch.min(surr1, surr2) + 0.5 * self.loss(state_values, rewards) - 0.01 * dist_entropy
						self.optimizer.zero_grad()
						loss.mean().backward()
						self.optimizer.step()
					self.g_tr_cnt += 1			# 全局学习次数加一
					if self.g_tr_cnt == self.total_tr_cnt:
						break
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
								# r += self.env.reward
								self.env.show_dynamic_image(isWait=False)  # 画图
						cv.destroyAllWindows()
					print('========== EPISODE END ==========\n')
					self.permit_exploration()  # 允许收集数据
				else:
					'''啥也不干，等着采集数据'''
					pass
			print('Training termination...')
			self.forbid_exploration()
			torch.cuda.empty_cache()
			self.clean()
		except:		# TODO 这一行一定要加，因为使用到了共享内存
			print('Unexpected termination...')
			torch.cuda.empty_cache()
			self.clean()

	def save_models(self):
		self.global_policy.save_checkpoint()

	def save_models_all(self):
		self.global_policy.save_all_net()

	def DPPO2_info(self):
		with open(self.path + 'DPPO2_info.txt', 'w') as f:
			f.writelines('========== DPPO2 info ==========')
			f.writelines('number of process: {}'.format(self.num_of_pro))
			f.writelines('agent name: {}'.format(self.env.name))
			f.writelines('state_dim: {}'.format(self.env.state_dim))
			f.writelines('action_dim: {}'.format(self.env.action_dim))
			f.writelines('action_range: {}'.format(self.env.action_range))
			f.writelines('actor learning rate: {}'.format(self.actor_lr))
			f.writelines('critic learning rate: {}'.format(self.critic_lr))
			f.writelines('DPPO2 training device: {}'.format(self.device))
			f.writelines('action_std_init: {}'.format(self.action_std_init))
			f.writelines('action_std_decay_freq: {}'.format(self.action_std_decay_freq))
			f.writelines('action_std_decay_rate: {}'.format(self.action_std_decay_rate))
			f.writelines('min_action_std: {}'.format(self.min_action_std))
			f.writelines('total training count: {}'.format(self.total_tr_cnt))
			f.writelines('k_epoch: {}'.format(self.k_epo))
			f.writelines('gamma: {}'.format(self.gamma))
			f.writelines('========== DPPO2 info ==========')

	def clean(self):
		for i in range(self.num_of_pro):
			self.share_memory[i].close()
			self.share_memory[i].unlink()
		print('共享内存清理完毕')
