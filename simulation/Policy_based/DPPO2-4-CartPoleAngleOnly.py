import os
import sys
import datetime
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from environment.envs.cartpole.cartpole_angleonly import CartPoleAngleOnly
from algorithm.policy_base.Distributed_PPO2 import Distributed_PPO2 as DPPO2
from common.common_cls import *
import torch.multiprocessing as mp

optPath = '../../datasave/network/'
show_per = 1
timestep = 0
ENV = 'DPPO2-CartPoleAngleOnly'


def setup_seed(seed):
	torch.manual_seed(seed)
	# torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


setup_seed(3407)
os.environ["OMP_NUM_THREADS"] = "1"


class PPOActorCritic(nn.Module):
	def __init__(self, _state_dim, _action_dim, _action_std_init, name='PPOActorCritic', chkpt_dir=''):
		super(PPOActorCritic, self).__init__()
		self.checkpoint_file = chkpt_dir + name + '_ppo'
		self.checkpoint_file_whole_net = chkpt_dir + name + '_ppoALL'
		self.action_dim = _action_dim
		self.state_dim = _state_dim
		self.action_std_init = _action_std_init
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
		self.device = 'cuda:0'
		torch.cuda.empty_cache()	# 清一下显存，不清暂时也没啥问题，不过应该没坏处
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


if __name__ == '__main__':
	log_dir = '../../datasave/log/'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ENV + '/'
	os.mkdir(simulationPath)
	c = cv.waitKey(1)
	TRAIN = True  # 直接训练
	RETRAIN = False  # 基于之前的训练结果重新训练
	TEST = not TRAIN

	env = CartPoleAngleOnly(0, False)

	if TRAIN:
		'''1. 启动多进程'''
		mp.set_start_method('spawn', force=True)

		'''2. 定义 DPPO2 基本参数'''
		actor_lr = 3e-4
		critic_lr = 1e-3
		process_num = 1
		action_std = 0.6
		action_std_decay_freq = int(5e4)
		action_std_decay_rate = 0.05
		min_action_std = 0.1
		action_std_init = 0.8
		eps_clip = 0.2
		total_tr_cnt = 1	# 5000
		k_epo = 250

		agent = DPPO2(env=env,
					  actor_lr=actor_lr,
					  critic_lr=critic_lr,
					  num_of_pro=process_num,
					  path=simulationPath,
					  action_std_decay_freq=action_std_decay_freq,
					  action_std_decay_rate=action_std_decay_rate,
					  min_action_std=min_action_std,
					  action_std_init=action_std_init,
					  eps_clip=eps_clip,
					  total_tr_cnt=total_tr_cnt,
					  k_epo=k_epo)

		'''3. 重新加载全局网络和优化器，这是必须的操作，因为考虑到不同的学习环境要设计不同的网络结构，在训练前，要重写 PPOActorCritic，并且重新加载优化器'''
		agent.global_policy = PPOActorCritic(agent.env.state_dim, agent.env.action_dim, action_std, 'GlobalPolicy', simulationPath)
		agent.global_policy.share_memory()
		agent.optimizer = torch.optim.Adam([
			{'params': agent.global_policy.actor.parameters(), 'lr': agent.actor_lr},
			{'params': agent.global_policy.critic.parameters(), 'lr': agent.critic_lr}
		])

		'''4. 添加进程'''
		agent.add_worker()
		agent.DPPO2_info()

		'''5. 启动多进程'''
		agent.start_multi_process()
	else:
		agent = DPPO2(env=env, actor_lr=3e-4, critic_lr=1e-3, num_of_pro=0, path=simulationPath)
		pass
