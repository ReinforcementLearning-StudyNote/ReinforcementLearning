from algorithm.value_base.DQN import DQN
import torch
import numpy as np

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")
"""use CPU or GPU"""


class Double_DQN(DQN):
    def __init__(self,
                 gamma,
                 epsilon,
                 learning_rate,
                 memory_capacity,
                 batch_size,
                 target_replace_iter,
                 modelFileXML):
        super(Double_DQN, self).__init__(gamma, epsilon, learning_rate, memory_capacity, batch_size, target_replace_iter, modelFileXML)

    def nn_training(self, saveNNPath=None):
        """
        :brief:             train the neural network
        :param saveNNPath:  path of the pkl file
        :return:            None
        """
        self.target_replace_count += 1
        if self.target_replace_count % self.target_replace_iter == 0:  # 满足这个条件，网络参数就更新一次
            self.target_net.load_state_dict(self.eval_net.state_dict())
            torch.save(self.target_net, saveNNPath + '/' + 'dqn.pkl')
            torch.save(self.target_net.state_dict(), saveNNPath + '/' + 'dqn_parameters.pkl')
            torch.save(self.eval_net, saveNNPath + '/' + 'eval_dqn.pkl')
            torch.save(self.eval_net.state_dict(), saveNNPath + '/' + 'eval_dqn_parameters.pkl')
            print('网络更新：', int(self.target_replace_count / self.target_replace_iter))
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        batch_memory = np.atleast_2d([self.replay_memory[i] for i in sample_index])
        '''得到s, a, r, s', boolend 索引的index'''
        index_s = [0, self.state_dim_nn]
        index_a = [index_s[1], index_s[1] + self.action_dim_physical]
        index_r = [index_a[1], index_a[1] + 1]
        index_s_ = [index_r[1], index_r[1] + self.state_dim_nn]
        index_b = [index_s_[1], index_s_[1] + 1]
        '''得到s, a, r, s', boolend 索引的index'''
        if self.state_dim_nn == 1:
            t_s = torch.unsqueeze(torch.from_numpy(batch_memory[:, index_s[0]: index_s[1]]).float(), dim=1).to(device)
            t_s_ = torch.unsqueeze(torch.from_numpy(batch_memory[:, index_s_[0]: index_s_[1]]).float(), dim=1).to(device)
        else:
            t_s = torch.squeeze(torch.from_numpy(batch_memory[:, index_s[0]: index_s[1]]).float()).to(device)
            t_s_ = torch.squeeze(torch.from_numpy(batch_memory[:, index_s_[0]: index_s_[1]]).float()).to(device)
        t_a = batch_memory[:, index_a[0]: index_a[1]]  # 是个numpy
        t_a_pos = self.torch_action2num(t_a).to(device)  # t_a是具体的物理动作，需要转换成动作编号作为索引值，是个tensor
        t_r = torch.unsqueeze(torch.from_numpy(np.squeeze(batch_memory[:, index_r[0]: index_r[1]])).float(), dim=1).to(device)
        t_bool = torch.unsqueeze(torch.from_numpy(np.squeeze(batch_memory[:, index_b[0]: index_b[1]])).float(), dim=1).to(device)
        q_next = torch.squeeze(self.target_net(t_s_).detach().float()).to(device)

        # '''Double DQN'''
        # ddqn_action_value = self.eval_net(t_s_).detach().cpu().numpy()
        # ddqn_num = np.argmax(ddqn_action_value, axis=1)
        # t_ddqn_num = torch.unsqueeze(torch.from_numpy(ddqn_num).long(), dim=1).to(device)
        # q_target = t_r + self.gamma * (torch.gather(q_next, 1, t_ddqn_num).mul(t_bool))
        # '''Double DQN'''
        '''Double DQN'''
        ddqn_action_value2 = self.eval_net(t_s_).detach()
        t_ddqn_num2 = torch.argmax(ddqn_action_value2, dim=1, keepdim=True)
        q_target = t_r + self.gamma * (torch.gather(q_next, 1, t_ddqn_num2).mul(t_bool))
        print(t_ddqn_num2.size())
        '''Double DQN'''

        for _ in range(1):
            q_eval = self.eval_net(t_s).gather(1, t_a_pos)
            loss = self.loss_func(q_eval, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.saveData_StepTDErrorNNLose(self.target_replace_count,
                                            (q_target - q_eval).sum().detach().cpu().numpy(),
                                            loss.detach().cpu().numpy())

    def DoubleDQN_info(self):
        print('This is Double DQN:')
        self.DQN_info()
