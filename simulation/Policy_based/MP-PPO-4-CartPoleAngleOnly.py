import math
import os
import sys
import datetime
import time
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from environment.envs.cartpole.cartpole_angleonly import CartPoleAngleOnly
from algorithm.policy_base.Proximal_Policy_Optimization import Proximal_Policy_Optimization as PPO
from common.common_cls import *
import torch.multiprocessing as mp

cfgPath = '../../environment/config/'
cfgFile = 'CartPoleAngleOnly.xml'
optPath = '../../datasave/network/'
show_per = 1
timestep = 0
ENV = 'MP-PPO-CartPoleAngleOnly'


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


setup_seed(3407)


class PPOActorCritic(nn.Module):
	def __init__(self, _state_dim, _action_dim, _action_std_init, name='PPOActorCritic', chkpt_dir=''):
		super(PPOActorCritic, self).__init__()
		self.checkpoint_file = chkpt_dir + name + '_ppo'
		self.checkpoint_file_whole_net = chkpt_dir + name + '_ppoALL'
		self.action_dim = _action_dim
		self.state_dim = _state_dim
		self.action_std_init = _action_std_init
		# 应该是初始化方差，一个动作就一个方差，两个动作就两个方差，std 是标准差
		self.action_var = torch.full((self.action_dim,), self.action_std_init * self.action_std_init)
		self.actor = nn.Sequential(
			nn.Linear(self.state_dim, 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh(),
			nn.Linear(64, self.action_dim),
			nn.Tanh()
		)
		self.critic = nn.Sequential(
			nn.Linear(self.state_dim, 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh(),
			nn.Linear(64, 1)
		)
		# self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		self.device = 'cpu'
		# torch.cuda.empty_cache()
		self.to(self.device)

	def set_action_std(self, new_action_std):
		self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

	def forward(self):
		raise NotImplementedError

	def act(self, s):
		action_mean = self.actor(s)
		cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
		dist = MultivariateNormal(action_mean, cov_mat)

		_a = dist.sample()
		action_logprob = dist.log_prob(_a)
		state_val = self.critic(s)

		return _a.detach(), action_logprob.detach(), state_val.detach()

	def evaluate(self, s, a):
		action_mean = self.actor(s)
		action_var = self.action_var.expand_as(action_mean)
		cov_mat = torch.diag_embed(action_var).to(self.device)
		dist = MultivariateNormal(action_mean, cov_mat)

		# For Single Action Environments.
		if self.action_dim == 1:
			a = a.reshape(-1, self.action_dim)

		action_logprobs = dist.log_prob(a)
		dist_entropy = dist.entropy()
		state_values = self.critic(s)

		return action_logprobs, state_values, dist_entropy

	def save_checkpoint(self, name=None, path='', num=None):
		print('...saving checkpoint...')
		if name is None:
			torch.save(self.state_dict(), self.checkpoint_file)
		else:
			if num is None:
				torch.save(self.state_dict(), path + name)
			else:
				torch.save(self.state_dict(), path + name + str(num))

	def save_all_net(self):
		print('...saving all net...')
		torch.save(self, self.checkpoint_file_whole_net)

	def load_checkpoint(self):
		print('...loading checkpoint...')
		self.load_state_dict(torch.load(self.checkpoint_file))


class PPOWorker(mp.Process):
	def __init__(self,
				 global_policy: PPOActorCritic,
				 global_old_policy: PPOActorCritic,
				 global_opt: SharedAdam,
				 _res_queue: mp.Queue,
				 name: str,
				 global_ep: mp.Value,
				 process_index:int,
				 env: CartPoleAngleOnly,
				 _actor_lr: float,
				 _critic_lr: float,
				 gamma: float,
				 K_epochs: int,
				 eps_clip: float,
				 action_std_init: float,
				 buffer_size: int,
				 modelFileXML: str,
				 path: str):
		"""
		@param global_policy:		global policy in SharedMemory
		@param global_old_policy:	global old policy in SharedMemory
		@param global_opt:			global optimizer in SharedMemory
		@param _res_queue:			global flag in SharedMemory
		@param name:				name of the worker
		@param env:					the training environment
		@param _actor_lr:			inherit from PPO
		@param _critic_lr:			inherit from PPO
		@param gamma:				inherit from PPO
		@param K_epochs:			inherit from PPO
		@param eps_clip:			inherit from PPO
		@param action_std_init:		inherit from PPO
		@param buffer_size:			inherit from PPO
		@param modelFileXML:		inherit from PPO
		@param path:				inherit from PPO
		"""
		super(PPOWorker, self).__init__()
		self.process_index = process_index
		self.global_ep = global_ep
		self.name = name
		self.global_policy = global_policy
		self.global_old_policy = global_old_policy
		self.global_opt = global_opt
		self.res_queue = _res_queue
		self.env = env
		self.agent = PPO(_actor_lr, _critic_lr, gamma, K_epochs, eps_clip, action_std_init, buffer_size, modelFileXML, path)
		self.agent.policy = PPOActorCritic(self.agent.state_dim_nn, self.agent.action_dim_nn, self.agent.action_std, 'Policy' + self.name, simulationPath)
		self.agent.policy_old = PPOActorCritic(self.agent.state_dim_nn, self.agent.action_dim_nn, self.agent.action_std, 'Policy_old' + self.name, simulationPath)
		self.agent.optimizer = global_opt  # 这里注意一定要使用全局的优化器，因为它在 SharedMemory 里面
		# self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		self.device = 'cpu'

	def agent_evaluate(self):
		test_num = 5
		for _ in range(test_num):
			self.env.reset_random()
			while not self.env.is_terminal:
				self.env.current_state = self.env.next_state.copy()
				_action_from_actor, _, _, _ = self.agent.choose_action(self.env.current_state)
				_action = self.agent.action_linear_trans(_action_from_actor.detach().cpu().numpy().flatten())  # 将动作转换到实际范围上
				self.env.step_update(_action)  # 环境更新的action需要是物理的action
				self.env.show_dynamic_image(isWait=False)  # 画图
		cv.destroyAllWindows()

	def mp_learn(self):
		rewards = []
		discounted_reward = 0
		for reward, is_terminal in zip(reversed(self.agent.buffer.rewards), reversed(self.agent.buffer.is_terminals)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + (self.agent.gamma * discounted_reward)
			rewards.insert(0, discounted_reward)

		rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

		with torch.no_grad():
			old_states = torch.FloatTensor(self.agent.buffer.states).detach().to(self.device)
			old_actions = torch.FloatTensor(self.agent.buffer.actions).detach().to(self.device)
			old_log_probs = torch.FloatTensor(self.agent.buffer.log_probs).detach().to(self.device)
			old_state_values = torch.FloatTensor(self.agent.buffer.state_values).detach().to(self.device)

		advantages = rewards.detach() - old_state_values.detach()

		for _ in range(self.agent.K_epochs):
			log_probs, state_values, dist_entropy = self.agent.policy.evaluate(old_states, old_actions)
			state_values = torch.squeeze(state_values)
			ratios = torch.exp(log_probs - old_log_probs.detach())
			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1 - self.agent.eps_clip, 1 + self.agent.eps_clip) * advantages
			loss = -torch.min(surr1, surr2) + 0.5 * self.agent.loss(state_values, rewards) - 0.01 * dist_entropy

			self.global_opt.zero_grad()
			loss.mean().backward()
			i = 0
			for lp, gp in zip(self.agent.policy.parameters(), self.global_policy.parameters()):		# TODO 很关键
				# TODO 问题是： global_policy 没有梯度
				print('====================')
				print('奥利给干了', i)
				print(gp._grad)
				print(lp.grad.shape)
				print('====================')
				gp._grad = lp.grad
			self.global_opt.step()

		'''6. Copy new weights into old policy'''
		self.agent.policy_old.load_state_dict(self.agent.policy.state_dict())		# 将 local  的 policy 复制给 policy_old
		self.global_old_policy.load_state_dict(self.global_policy.state_dict())		# 将 global 的 policy 复制给 policy_old

		self.agent.policy_old.load_state_dict(self.global_old_policy)
		self.agent.policy.load_state_dict(self.global_policy)

	def run(self):
		max_training_timestep = int(self.env.timeMax / self.env.dt) * 5000  # 5000 最长回合的数据
		action_std_decay_freq = int(2.5e5)
		action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
		min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)

		sumr = 0
		start_eps = 0
		train_num = 0
		index = 0
		timestep = 0
		while timestep <= max_training_timestep:
			self.env.reset_random()
			while not self.env.is_terminal:
				self.env.current_state = self.env.next_state.copy()
				action_from_actor, s, a_log_prob, s_value = self.agent.choose_action(self.env.current_state)  # 返回三个没有梯度的tensor
				action = self.agent.action_linear_trans(action_from_actor.detach().cpu().numpy().flatten())  # 将动作转换到实际范围上
				self.env.step_update(action)  # 环境更新的action需要是物理的action
				# env.show_dynamic_image(isWait=False)  # 画图
				sumr += self.env.reward
				'''存数'''
				self.agent.buffer.append(s=self.env.current_state,
										 a=action_from_actor.cpu().numpy(),
										 log_prob=a_log_prob.cpu().numpy(),
										 r=self.env.reward,
										 sv=s_value.cpu().numpy(),
										 done=1.0 if self.env.is_terminal else 0.0,
										 index=index)
				index += 1
				timestep += 1
				'''存数'''
				'''学习'''
				if timestep % self.agent.buffer.batch_size == 0:
					print('========== LEARN ==========')
					print('Episode: {}'.format(self.agent.episode))
					print('Num of learning: {}'.format(train_num))
					self.mp_learn()
					'''clear buffer'''
					# agent.buffer.clear()
					train_num += 1
					print('Average reward:', round(sumr / (self.agent.episode + 1 - start_eps), 3))
					start_eps = self.agent.episode
					sumr = 0
					index = 0
					if train_num % 50 == 0 and train_num > 0:
						if self.process_index == 0:
							self.agent_evaluate()
						# print('check point save')
						# temp = simulationPath + 'episode' + '_' + str(self.agent.episode) + '_save/'
						# os.mkdir(temp)
						# time.sleep(0.01)
						# self.agent.policy_old.save_checkpoint(name='Policy_PPO', path=temp, num=timestep)
					print('========== LEARN ==========')
				'''学习'''

				if timestep % action_std_decay_freq == 0:
					self.agent.decay_action_std(action_std_decay_rate, min_action_std)
			self.agent.episode += 1
			self.global_ep.value += 1
		self.res_queue.put(None)  # 这个进程结束了，就把None放进去，用于global判断


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)

	log_dir = '../../datasave/log/'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ENV + '/'
	os.mkdir(simulationPath)
	c = cv.waitKey(1)
	TRAIN = True  # 直接训练
	RETRAIN = False  # 基于之前的训练结果重新训练
	TEST = not TRAIN

	ref_env = CartPoleAngleOnly(0, False)  # 仅作为参考用，主进程不训练
	actor_lr = 3e-4
	critic_lr = 1e-3
	action_std = 0.6

	global_ploicy = PPOActorCritic(ref_env.state_dim, ref_env.state_dim, action_std, 'Policy', simulationPath)
	global_old_ploicy = PPOActorCritic(ref_env.state_dim, ref_env.state_dim, action_std, 'Policy_old', simulationPath)
	global_ploicy.share_memory()
	global_old_ploicy.share_memory()
	global_optimizer = SharedAdam([
		{'params': global_ploicy.actor.parameters(), 'lr': actor_lr},
		{'params': global_ploicy.critic.parameters(), 'lr': critic_lr}
	])
	res_queue = mp.Queue()
	global_episode = mp.Value('i', 0)
	# pro_num = mp.cpu_count()
	pro_num = 1

	'''
	global_policy: PPOActorCritic,
	global_old_policy: PPOActorCritic,
	global_opt: SharedAdam,
	_res_queue: mp.Value,
	name: str,
	process_index:int,
	env: CartPoleAngleOnly,
	_actor_lr: float,
	_critic_lr: float,
	gamma: float,
	K_epochs: int,
	eps_clip: float,
	action_std_init: float,
	buffer_size: int,
	modelFileXML: str,
	path: str
	'''

	workers = [PPOWorker(global_ploicy, global_old_ploicy, global_optimizer, res_queue, 'worker' + str(i), global_episode,
						 i, ref_env,
						 3e-4, 1e-3, 0.99, 80, 0.2, 0.6, int(ref_env.timeMax / ref_env.dt * 2),
						 cfgPath + cfgFile,
						 simulationPath) for i in range(pro_num)]
	[w.start() for w in workers]
	pre_ep = 0
	cur_ep = 0
	save_iter = 100
	while res_queue.get():
		if global_episode.value % save_iter <= 5:
			print('...save check point...')
			temp = simulationPath + 'episode' + '_' + str(global_episode.value) + '_save/'
			os.mkdir(temp)
			global_old_ploicy.save_checkpoint(name='Policy_PPO', path=temp, num=timestep)

	[w.join() for w in workers]
