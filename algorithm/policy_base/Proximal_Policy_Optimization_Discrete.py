from common.common_cls import *
import cv2 as cv


class Proximal_Policy_Optimization_Discrete:
    def __init__(self,
                 env,
                 actor_lr: float=3e-4,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.99,
                 K_epochs: int = 10,
                 eps_clip: float = 0.2,
                 buffer_size: int = 1200,
                 actor: SoftmaxActor = SoftmaxActor(),
                 critic: Critic = Critic(),
                 path: str = ''):
        """
        @param env:				environment
        @param gamma:			discount factor
        @param K_epochs:		PPO parameter
        @param eps_clip:		PPO parameter
        @param buffer_size:		buffer size
        @param actor:			actor net
        @param critic:			critic net
        @param path:			record path
        """
        self.env = env
        self.gamma = gamma

        '''PPO'''
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma  # discount factor
        self.K_epochs = K_epochs  # 每隔 timestep_num 学习一次
        self.eps_clip = eps_clip
        self.path = path
        self.buffer = RolloutBuffer(buffer_size, self.env.state_dim, self.env.action_dim)
        self.buffer2 = RolloutBuffer2(self.env.state_dim, self.env.action_dim)
        self.actor = actor
        self.critic = critic
        '''PPO'''

        dist = [torch.ones(self.env.action_num[i], dtype=torch.float32) / self.env.action_num[i] for i in range(self.env.action_dim)]
        self.random_policy = Categorical(probs=nn.utils.rnn.pad_sequence(dist).T)

        self.device = 'cpu'
        self.episode = 0

    def choose_action_random(self):
        """
        :brief:     因为该函数与choose_action并列，所以输出也必须是[-1, 1]之间
        :return:    random action
        """
        _a = self.random_policy.sample()
        return _a

    def choose_action(self, state, exploration=-1):
        with torch.no_grad():
            t_state = torch.FloatTensor(state).to(self.device)
            _s_value = self.critic(t_state).cpu()
            if 0 < exploration < 1:
                '''random action'''
                if np.random.uniform(0, 1) < exploration:
                    _a = self.choose_action_random()
                    _a_log_prob = torch.mean(self.random_policy.log_prob(_a))
                else:
                    _a, _a_log_prob, _ = self.actor.choose_action(t_state)
            else:
                _a, _a_log_prob, _ = self.actor.choose_action(t_state)
        return _a, t_state, _a_log_prob, _s_value

    def evaluate(self, state):
        with torch.no_grad():
            t_state = torch.FloatTensor(state).to(self.device)
            _a = self.actor.evaluate(t_state)
        return _a

    def train_evaluate(self, state, action):
        _dist = Categorical(probs=self.actor.forward(state))

        if self.env.action_dim == 1:
            action = action.reshape(-1, self.env.action_dim)

        _log_probs = torch.sum(_dist.log_prob(action), dim=1)
        _dist_entropy = torch.sum(_dist.entropy(), dim=1)
        _s_values = self.critic(state)
        return _log_probs, _s_values, _dist_entropy

    def agent_evaluate(self, show, _random=False, test_num=None):
        rr = []
        ee = []
        '''fixed initial position, velocity'''
        if _random:
            try:
                for _ in range(test_num):
                    self.env.reset_random()
                    r = 0
                    while not self.env.is_terminal:
                        self.env.current_state = self.env.next_state.copy()
                        _action_from_actor = self.evaluate(self.env.current_state)
                        _action = self.action_linear_trans(_action_from_actor.cpu().numpy().flatten())  # 将动作转换到实际范围上
                        self.env.step_update(_action)  # 环境更新的action需要是物理的action
                        r += self.env.reward
                        if show:
                            self.env.show_dynamic_image(isWait=False)  # 画图
                    rr.append(r)
                    ee.append(np.linalg.norm(self.env.error))
            except:
                print('Invalid input argument: test_num.')
        else:
            xx = np.arange(0.5, self.env.map_size[0], 0.5)
            yy = np.arange(0.5, self.env.map_size[1], 0.5)
            pts = []
            for _xx in xx:
                for _yy in yy:
                    pts.append([_xx, _yy])
            pts = np.array(pts).astype(np.float32)
            for _pt in pts:
                # self.env.reset_random()
                self.env.init_target = _pt
                self.env.reset()
                r = 0
                while not self.env.is_terminal:
                    self.env.current_state = self.env.next_state.copy()
                    _action_from_actor = self.evaluate(self.env.current_state)
                    _action = self.action_linear_trans(_action_from_actor.cpu().numpy().flatten())  # 将动作转换到实际范围上
                    self.env.step_update(_action)  # 环境更新的action需要是物理的action
                    r += self.env.reward
                    if show:
                        self.env.show_dynamic_image(isWait=False)  # 画图
                rr.append(r)
                ee.append(np.linalg.norm(self.env.error))
        cv.destroyAllWindows()
        return np.array(rr).astype(np.float32), np.array(ee).astype(np.float32)

    def agent_evaluate_once(self):
        self.env.reset()
        r = 0
        while not self.env.is_terminal:
            self.env.current_state = self.env.next_state.copy()
            _action_from_actor = self.evaluate(self.env.current_state)
            _action = self.action_linear_trans(_action_from_actor.cpu().numpy().flatten())  # 将动作转换到实际范围上
            self.env.step_update(_action)  # 环境更新的action需要是物理的action
            r += self.env.reward
            self.env.show_dynamic_image(isWait=False)  # 画图

    def action_linear_trans(self, action):
        """
        @param action:
        @return:
        """
        linear_action = []
        for _a, _action_space in zip(action, self.env.action_space):
            linear_action.append(_action_space[_a])
        return np.array(linear_action)

    def learn(self, adv_norm=False, lr_decay=False, decay_rate=None):
        """
        @param adv_norm:        advantage normalization
        @param lr_decay:
        @param decay_rate:
        @return:
        """
        '''1. 计算轨迹中每一个状态的累计回报'''
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        '''2. 奖励归一化'''
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        '''3. 将 numpy 数据转化为 tensor'''
        with torch.no_grad():
            old_states = torch.FloatTensor(self.buffer.states).detach().to(self.device)
            old_actions = torch.FloatTensor(self.buffer.actions).detach().to(self.device)
            old_log_probs = torch.FloatTensor(self.buffer.log_probs).detach().to(self.device)
            old_state_values = torch.FloatTensor(self.buffer.state_values).detach().to(self.device)

        '''4. 计算优势函数'''
        advantages = rewards.detach() - old_state_values.detach()
        if adv_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # self.actor.train()
        # self.critic.train()

        '''5. 学习 K 次'''
        for _ in range(self.K_epochs):
            '''5.1 Evaluating old actions and values'''
            log_probs, state_values, dist_entropy = self.train_evaluate(old_states, old_actions)

            '''5.2 match state_values tensor dimensions with rewards tensor'''
            state_values = torch.squeeze(state_values)

            '''5.3 Finding the ratio (pi_theta / pi_theta__old)'''
            ratios = torch.exp(log_probs - old_log_probs.detach())

            '''5.4 Finding Surrogate Loss'''
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            '''5.5 final loss of clipped objective PPO'''
            actor_loss = -torch.min(surr1, surr2) - 0.01 * dist_entropy
            self.actor.optimizer.zero_grad()
            actor_loss.mean().backward()
            self.actor.optimizer.step()

            critic_loss = 0.5 * func.mse_loss(state_values, rewards)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

        if lr_decay:
            for p in self.actor.optimizer.param_groups:
                p['lr'] = self.actor_lr * (1 - decay_rate)
            for p in self.critic.optimizer.param_groups:
                p['lr'] = self.critic_lr * (1 - decay_rate)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def save_models_all(self):
        self.actor.save_all_net()
        self.critic.save_all_net()

    def load_models(self, path='', actor_name='PPO_Actor', critic_name='PPO_Critic'):
        """
        @param path:
        @param actor_name:
        @param critic_name:
        @return:
        """
        print('...loading checkpoint...')
        self.actor.load_state_dict(torch.load(path + actor_name))
        self.critic.load_state_dict(torch.load(path + critic_name))

    def PPO_info(self):
        print('agent name：', self.env.name)
        print('state_dim:', self.env.state_dim)
        print('action_dim:', self.env.action_dim)
        print('action num:', self.env.action_num)
        print('action_space:', self.env.action_space)
