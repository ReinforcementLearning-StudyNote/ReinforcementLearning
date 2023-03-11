import os
import time
import numpy as np
import math
from environment.config.xml_write import xml_cfg
import random
import cv2 as cv
from common.common_func import *
import torch
import pandas as pd
import matplotlib.pyplot as plt
from environment.envs.pathplanning.bezier import Bezier
import gym
import torch.nn as nn
import collections
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
env = gym.make('CartPole-v1').unwrapped
'''
CartPole的环境状态特征量为推车的位置x、速度x_dot、杆子的角度theta、角速度theta_dot
状态是这四个状态特征所组成的，情况将是无限个，是连续的（即无限个状态），动作是推车向左+1，向右-1，（离散的，有限个，2个）
'''
state_number = env.observation_space.shape[0]
action_number = env.action_space.n
LR_A = 0.005  # learning rate for actor
LR_C = 0.01  # learning rate for critic
Gamma = 0.9
Switch = 1  # 训练、测试切换标志
'''AC第一部分 设计actor'''
'''第一步.设计actor和critic的网络部分'''


class ActorNet(nn.Module):
	def __init__(self):
		super(ActorNet, self).__init__()
		self.in_to_y1 = nn.Linear(state_number, 50)
		self.in_to_y1.weight.data.normal_(0, 0.1)
		self.y1_to_y2 = nn.Linear(50, 20)
		self.y1_to_y2.weight.data.normal_(0, 0.1)
		self.out = nn.Linear(20, action_number)
		self.out.weight.data.normal_(0, 0.1)

	def forward(self, x):
		x = self.in_to_y1(x)
		x = F.relu(x)
		x = self.y1_to_y2(x)
		x = torch.sigmoid(x)
		act = self.out(x)
		return F.softmax(act, dim=-1)


class CriticNet(nn.Module):
	def __init__(self):
		super(CriticNet, self).__init__()
		self.in_to_y1 = nn.Linear(state_number, 40)
		self.in_to_y1.weight.data.normal_(0, 0.1)
		self.y1_to_y2 = nn.Linear(40, 20)
		self.y1_to_y2.weight.data.normal_(0, 0.1)
		self.out = nn.Linear(20, 1)
		self.out.weight.data.normal_(0, 0.1)

	def forward(self, x):
		x = self.in_to_y1(x)
		x = F.relu(x)
		x = self.y1_to_y2(x)
		x = torch.sigmoid(x)
		act = self.out(x)
		return act


class Actor():
	def __init__(self):
		self.actor = ActorNet()
		self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_A)
	'''第二步.编写actor的选择动作函数'''

	def choose(self, x):
		x = torch.FloatTensor(x)
		probs = self.actor(x).detach().numpy()
		action = np.random.choice(np.arange(action_number), p=probs)
		return action

	'''第四步.根据td-error进行学习，编写公式log(p(s,a))*td_e的代码'''

	def learn(self, s, a, td):
		prob = self.actor(torch.FloatTensor(s))
		log_prob = torch.log(prob)
		actor_loss = -log_prob[a] * td
		self.optimizer.zero_grad()
		actor_loss.backward()
		self.optimizer.step()


'''第二部分 Critic部分'''
class Critic():
	def __init__(self):
		self.critic = CriticNet()
		self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_C)
		self.lossfunc = nn.MSELoss()  # 均方误差（MSE）

	'''第三步.编写td-error的计算代码（V现实减去V估计就是td-error）'''

	def learn(self, s, r, s_):
		'''当前的状态s计算当前的价值，下一个状态s_计算出下一状态的价值v_，然后v_乘以衰减γ再加上r就是v现实'''
		s = torch.FloatTensor(s)
		v = self.critic(s)  # 输入当前状态，有网络得到估计v
		r = torch.FloatTensor([r])  # .unsqueeze(0)#unsqueeze(0)在第一维度增加一个维度
		s_ = torch.FloatTensor(s_)
		reality_v = r + Gamma * self.critic(s_).detach()  # 现实v
		td_e = self.lossfunc(reality_v, v)
		self.optimizer.zero_grad()
		td_e.backward()
		self.optimizer.step()
		advantage = (reality_v - v).detach()
		return advantage  # pytorch框架独有的毛病：返回一定要用reality_v-v，但是误差反向传递一定要用td_e，不然误差传不了，不能收敛


'''训练'''
if Switch == 0:
	print('AC训练中...')
	actor = Actor()
	critic = Critic()
	for i in range(2000):
		r_totle = []
		s = env.reset()  # 环境重置
		while True:
			action = actor.choose(s)
			s_, r, done, info = env.step(action)
			if done:
				r = -50  # 稍稍修改奖励，让其收敛更快
			td_error = actor.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
			actor.learn(s, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
			s = s_
			r_totle.append(r)
			if done:
				break
		r_sum = sum(r_totle)
		print("\r回合数：{} 奖励：{}".format(i, r_sum), end=" ")
		if i % 50 == 0 and i > 300:  # 保存神经网络参数
			save_data = {'net': actor.actor.state_dict(), 'opt': actor.optimizer.state_dict(), 'i': i}
			torch.save(save_data, "model_actor.pth")
			save_data = {'net': critic.critic.state_dict(), 'opt': critic.optimizer.state_dict(), 'i': i}
			torch.save(save_data, "model_critic.pth")
else:
	print('AC测试中...')
	aa = Actor()
	cc = Critic()
	checkpoint_aa = torch.load("model_actor.pth")
	aa.actor.load_state_dict(checkpoint_aa['net'])
	checkpoint_cc = torch.load("model_critic.pth")
	cc.critic.load_state_dict(checkpoint_cc['net'])
	for j in range(100):
		state = env.reset()
		total_rewards = 0
		while True:
			env.render()
			state = torch.FloatTensor(state)
			action = aa.choose(state)
			new_state, reward, done, info = env.step(action)  # 执行动作
			total_rewards += reward
			if done:
				print("Score", total_rewards)
				break
			state = new_state
	env.close()
