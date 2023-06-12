import cv2 as cv
import numpy as np

from common.common_cls import *
import torch.multiprocessing as mp
import pandas as pd

"""
	Note: CPU is recommended for DPPO.
"""


class Worker(mp.Process):
	def __init__(self,
				 g_pi: PPOActorCritic,
				 l_pi: PPOActorCritic,
				 g_opt: SharedAdam,
				 g_train_n: mp.Value,
				 _index: int,
				 _name: str,
				 _env,
				 _queue: mp.Queue,
				 _lock: mp.Lock,
				 _ppo_msg: dict):
		"""
		@param g_pi:		global policy
		@param l_pi:		local policy
		@param g_opt:		global optimizer
		@param g_train_n:	global training number
		@param _index:		process number
		@param _name:		name of the worker
		@param _env:		RL environment
		@param _queue:		flag
		@param _lock:		process lock
		@param _ppo_msg:	ppo information
		"""
		super(Worker, self).__init__()
		self.g_pi = g_pi
		self.l_pi = l_pi
		self.g_opt = g_opt
		self.global_training_num = g_train_n
		self.index = _index
		self.name = _name
		self.env = _env
		self.queue = _queue
		self.lock = _lock
		self.buffer = RolloutBuffer(int(self.env.timeMax / self.env.dt * 2), self.env.state_dim, self.env.action_dim)
		self.buffer2 = RolloutBuffer2(self.env.state_dim, self.env.action_dim)
		self.gamma = _ppo_msg['gamma']
		self.k_epo = _ppo_msg['k_epo']
		self.eps_c = _ppo_msg['eps_c']
		self.action_std = _ppo_msg['a_std']
		self.device = _ppo_msg['device']
		self.loss = _ppo_msg['loss']
		self.episode = 0

	def learn(self):
		rewards = []
		discounted_reward = 0
		for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + (self.gamma * discounted_reward)
			rewards.insert(0, discounted_reward)

		rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
		# rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

		with torch.no_grad():
			old_states = torch.FloatTensor(self.buffer.states).detach().to(self.device)
			old_actions = torch.FloatTensor(self.buffer.actions).detach().to(self.device)
			old_log_probs = torch.FloatTensor(self.buffer.log_probs).detach().to(self.device)
			old_state_values = torch.FloatTensor(self.buffer.state_values).detach().to(self.device)

		advantages = rewards.detach() - old_state_values.detach()

		for _ in range(self.k_epo):
			log_probs, state_values, dist_entropy = self.l_pi.evaluate(old_states, old_actions)
			state_values = torch.squeeze(state_values)
			ratios = torch.exp(log_probs - old_log_probs.detach())
			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1 - self.eps_c, 1 + self.eps_c) * advantages
			loss = -torch.min(surr1, surr2) + 0.5 * self.loss(state_values, rewards) - 0.01 * dist_entropy

			'''网络更新 and 全局梯度更新'''
			self.g_opt.zero_grad()
			loss.mean().backward()
			for lp, gp in zip(self.l_pi.parameters(), self.g_pi.parameters()):
				gp._grad = lp.grad
			self.g_opt.step()
			'''网络更新 and 全局梯度更新'''

	def choose_action(self, state):
		with torch.no_grad():
			t_state = torch.FloatTensor(state).to(self.device)
			action, action_log_prob, state_val = self.l_pi.act(t_state)

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
		return np.array(linear_action)

	def set_action_std(self, new_action_std):
		self.action_std = new_action_std
		self.l_pi.set_action_std(new_action_std)

	def decay_action_std(self, action_std_decay_rate, min_action_std):
		self.action_std = self.action_std - action_std_decay_rate
		self.action_std = round(self.action_std, 4)
		if self.action_std <= min_action_std:
			self.action_std = min_action_std
			print("setting actor output action_std to min_action_std : ", self.action_std)
		else:
			print("setting actor output action_std to : ", self.action_std)
		self.set_action_std(self.action_std)

	def run(self):
		max_training_timestep = int(self.env.timeMax / self.env.dt) * 20000  # 5000 最长回合的数据
		# max_training_timestep = 5000
		action_std_decay_freq = int(8e5)  # 每隔这么多个 timestep 把探索方差减小点
		action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
		min_action_std = 0.6  # 方差最小不能小于 0.4，不管啥时候，都得适当探索
		train_num = 0
		timestep = 0
		start_eps = 0
		while timestep <= max_training_timestep:
			index = 0
			sumr = 0
			self.l_pi.load_state_dict(self.g_pi.state_dict())  # 从全局 policy 中加载网络
			'''收集数据'''
			while index < self.buffer.batch_size:
				self.env.reset_random()
				# self.env.reset()
				while not self.env.is_terminal:
					self.env.current_state = self.env.next_state.copy()
					action_from_actor, s, a_log_prob, s_value = self.choose_action(self.env.current_state)  # 返回三个没有梯度的tensor
					action_from_actor = action_from_actor.numpy()
					action = self.action_linear_trans(action_from_actor.flatten())  # 将动作转换到实际范围上
					self.env.step_update(action.astype(np.float32))  # 环境更新的action需要是物理的action
					sumr += self.env.reward
					self.buffer.append(s=self.env.current_state,
									   a=action_from_actor,
									   log_prob=a_log_prob.numpy(),
									   r=self.env.reward,
									   sv=s_value.numpy(),
									   done=1.0 if self.env.is_terminal else 0.0,
									   index=index)
					index += 1
					timestep += 1
					if timestep % action_std_decay_freq == 0:
						self.decay_action_std(action_std_decay_rate, min_action_std)
					if index == self.buffer.batch_size:
						break
			'''收集数据'''
			'''学习'''
			# print('========== LEARN ' + self.name + '==========')
			# print('Num of learning: {}'.format(train_num))
			self.learn()
			train_num += 1
			# print('Average reward:', round(sumr / (self.episode + 1 - start_eps), 3))
			start_eps = self.episode
			with self.lock:
				self.global_training_num.value += 1
				# print('Global training: ', self.global_training_num.value)
			# print('========== LEARN ==========')
			'''学习'''
			self.episode += 1
			self.queue.put(round(sumr / (self.episode + 1 - start_eps), 3))

		self.queue.put(None)  # 这个进程结束了，就把None放进去，用于global判断

	'''这个run是给UGVBi用的'''
	# def run(self):
	# 	max_training_timestep = int(self.env.timeMax / self.env.dt) * 20000  # 5000 最长回合的数据
	# 	# max_training_timestep = 5000
	# 	action_std_decay_freq = int(2.5e5)  # 每隔这么多个 timestep 把探索方差减小点
	# 	action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
	# 	min_action_std = 0.4  # 方差最小不能小于 0.1，不管啥时候，都得适当探索
	# 	train_num = 0
	# 	timestep = 0
	# 	start_eps = 0
	# 	while timestep <= max_training_timestep:
	# 		sumr = 0
	# 		self.l_pi.load_state_dict(self.g_pi.state_dict())  # 从全局 policy 中加载网络
	# 		'''收集数据'''
	# 		_temp_s = np.atleast_2d([]).astype(np.float32)
	# 		_temp_a = np.atleast_2d([]).astype(np.float32)
	# 		_temp_log_prob = np.atleast_1d([]).astype(np.float32)
	# 		_temp_r = np.atleast_1d([]).astype(np.float32)
	# 		_temp_sv = np.atleast_1d([]).astype(np.float32)
	# 		_temp_done = np.atleast_1d([]).astype(np.float32)
	# 		_localTimeStep = 0
	# 		self.env.reset_random()
	# 		# self.env.reset()
	# 		while not self.env.is_terminal:
	# 			self.env.current_state = self.env.next_state.copy()
	# 			action_from_actor, s, a_log_prob, s_value = self.choose_action(self.env.current_state)  # 返回三个没有梯度的tensor
	# 			action_from_actor = action_from_actor.numpy()
	# 			action = self.action_linear_trans(action_from_actor.flatten())  # 将动作转换到实际范围上
	# 			self.env.step_update(action)  # 环境更新的action需要是物理的action
	# 			self.env.show_dynamic_image(False)
	# 			sumr += self.env.reward
	# 			'''临时存数'''
	# 			if len(_temp_done) == 0:
	# 				_temp_s = np.atleast_2d(self.env.current_state).astype(np.float32)
	# 				_temp_a = np.atleast_2d(action_from_actor).astype(np.float32)
	# 				_temp_log_prob = np.atleast_1d(a_log_prob.numpy()).astype(np.float32)
	# 				_temp_r = np.atleast_1d(self.env.reward).astype(np.float32)
	# 				_temp_sv = np.atleast_1d(s_value.numpy()).astype(np.float32)
	# 				_temp_done = np.atleast_1d(1.0 if self.env.is_terminal else 0.0).astype(np.float32)
	# 			else:
	# 				_temp_s = np.vstack((_temp_s, self.env.current_state))
	# 				_temp_a = np.vstack((_temp_a, action_from_actor))
	# 				_temp_log_prob = np.hstack((_temp_log_prob, a_log_prob.numpy()))
	# 				_temp_r = np.hstack((_temp_r, self.env.reward))
	# 				_temp_sv = np.hstack((_temp_sv, s_value.numpy()))
	# 				_temp_done = np.hstack((_temp_done, 1.0 if self.env.is_terminal else 0.0))
	# 			_localTimeStep += 1
	# 			'''临时存数'''
	# 		'''收集数据'''
	#
	# 		'''走到这，回合肯定结束了'''
	# 		if self.env.terminal_flag != 4:
	# 			self.episode += 1
	# 			can_train = self.buffer.append_traj(_temp_s, _temp_a, _temp_log_prob, _temp_r, _temp_sv, _temp_done)
	# 			if can_train:
	# 				timestep += _localTimeStep
	# 				'''学习'''
	# 				self.learn()
	# 				train_num += 1
	# 				start_eps = self.episode
	# 				with self.lock:
	# 					self.global_training_num.value += 1
	# 				'''学习'''
	# 				if timestep > action_std_decay_freq:
	# 					action_std_decay_freq *= 2
	# 					self.decay_action_std(action_std_decay_rate, min_action_std)
	# 				self.queue.put(round(sumr / (self.episode + 1 - start_eps), 3))
	#
	# 	self.queue.put(None)  # 这个进程结束了，就把None放进去，用于global判断


class Distributed_PPO:
	def __init__(self, env, actor_lr: float = 3e-4, critic_lr: float = 1e-3, num_of_pro: int = 5, path: str = ''):
		"""
		@param env:			RL environment
		@param actor_lr:	actor learning rate
		@param critic_lr:	critic learning rate
		@param num_of_pro:	number of training process
		"""
		'''RL env'''
		# Remind: 这里的 env 还有 env 的成员函数是找不到索引的，需要确保写的时候没有问题才行，为了方便设计，不得已把 env 集成到 DPPO 里面
		# 如果形成一种习惯，也未尝不可，嗨嗨嗨
		self.env = env
		self.state_dim_nn = env.state_dim
		self.action_dim_nn = env.action_dim
		self.action_range = env.action_range
		'''RL env'''

		'''DPPO'''
		self.actor_lr = actor_lr
		self.critic_lr = critic_lr
		self.num_of_pro = num_of_pro  # 进程数量
		self.queue = mp.Queue()
		self.global_training_num = mp.Value('i', 0)
		self.lock = mp.Lock()
		self.path = path
		'''DPPO'''

		'''global variable'''
		self.global_policy = PPOActorCritic(self.state_dim_nn, self.action_dim_nn, 0.6, name='Policy_ppo', chkpt_dir=self.path)
		self.eval_policy = PPOActorCritic(self.state_dim_nn, self.action_dim_nn, 0.6, name='eval_policy', chkpt_dir=self.path)

		self.optimizer = SharedAdam([
			{'params': self.global_policy.actor.parameters(), 'lr': self.actor_lr},
			{'params': self.global_policy.critic.parameters(), 'lr': self.critic_lr}
		])
		self.global_policy.share_memory()
		'''global variable'''
		self.device = 'cpu'
		'''multi process'''
		self.processes = [mp.Process(target=self.global_evaluate, args=())]	# evaluation process
		'''multi process'''

		self.evaluate_record = []
		self.training_record = []

	def global_evaluate(self):
		while True:
			training_r = self.queue.get()
			if training_r is None:
				break
			if self.global_training_num.value % 50 == 0:
				print('Training count:, ', self.global_training_num.value)
			if self.global_training_num.value % 300 == 0:
				# 	print('Training count:, ', self.global_training_num.value)
				'''主进程不停地测试，每次随机选择 500 个回合。保存每次记录开始时候的网络，直至循环完成或者强制停止'''
				training_num_temp = self.global_training_num.value 		# 记录一下当前的数字，因为测试和学习同时进行的，号码容易窜
				self.eval_policy.load_state_dict(self.global_policy.state_dict())  # 复制 global policy
				print('...saving check point... ', int(training_num_temp))
				self.global_policy.save_checkpoint(name='Policy_PPO', path=self.path, num=training_num_temp)
				# self.save_models()
				eval_num = 10
				r = 0
				error = []
				for i in range(eval_num):
					if i % 100 == 0:
						print('测试: ', i)
					self.env.reset_random()
					# self.env.reset()
					while not self.env.is_terminal:
						self.env.current_state = self.env.next_state.copy()
						action_from_actor = self.evaluate(self.env.current_state)
						action_from_actor = action_from_actor.numpy()
						action = self.action_linear_trans(action_from_actor.flatten())  # 将动作转换到实际范围上
						# print(action_from_actor, action)
						self.env.step_update(action.astype(np.float32))  # 环境更新的action需要是物理的action
						r += self.env.reward
						self.env.show_dynamic_image(isWait=False)  # 画图
					# error.append(np.linalg.norm(self.env.error))
				cv.destroyAllWindows()
			# error = np.array(error)
			# with open(self.path + str(training_num_temp) + '.txt', 'w') as f:
			# 	f.write('mean error: %.3f\n' % (error.mean()))
			# 	f.write('std error: %.3f\n' % (error.std()))
			# 	f.write('max error: %.3f\n' % (np.max(error)))
			# 	f.write('min error: %.3f\n' % (np.min(error)))
				# r /= eval_num
				# self.evaluate_record.append(r)
				# self.save_evaluation_record()
		print('...training end...')

	def add_worker(self, worker: Worker):
		self.processes.append(worker)

	def start_multi_process(self):
		for p in self.processes:
			p.start()
			p.join(0.5)

	def evaluate(self, state):
		with torch.no_grad():
			t_state = torch.FloatTensor(state).to(self.device)
			action_mean = self.eval_policy.actor(t_state)
		return action_mean.detach()

	def save_models(self):
		self.global_policy.save_checkpoint()

	def save_models_all(self):
		self.global_policy.save_all_net()

	def load_models(self, path):
		"""
		:brief:         only for test
		:param path:    file path
		:return:
		"""
		print('...loading checkpoint...')
		self.global_policy.load_state_dict(torch.load(path + 'Policy_ppo'))

	def DPPO_info(self):
		print('number of process:', self.num_of_pro)
		print('agent name：', self.env.name)
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
		return np.array(linear_action)

	def save_training_record(self):
		pd.DataFrame({'training record': self.training_record}).to_csv(self.path + 'train_record.csv', index=True, sep=',')

	def save_evaluation_record(self):
		pd.DataFrame({'evaluation record': self.evaluate_record}).to_csv(self.path + 'evaluate_record.csv', index=True, sep=',')
