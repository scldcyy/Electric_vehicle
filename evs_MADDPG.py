import copy
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import rl_utils

import sys
from BSS_ENV import BSS_ENV, BSS_ENV2, BSS_ENV3


def onehot_from_logits(logits, eps=0.01):
    ''' 生成最优动作的独热（one-hot）形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
    # 正确地反传梯度
    return y


class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    ''' DDPG算法 '''

    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device):
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = TwoLayerFC(state_dim, action_dim,
                                       hidden_dim).to(device)
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(critic_input_dim, 1,
                                        hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

    def take_action(self, state, explore=False):
        action = self.actor(state)
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)


class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau):
        self.agents = []
        for i in range(len(env.agents)):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore):
        states = [
            torch.tensor(np.array([states[i]]), dtype=torch.float, device=self.device)
            for i in range(len(env.agents))
        ]
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def update(self, sample, i_agent):
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]

        cur_agent.critic_optimizer.zero_grad()
        all_target_act = [
            onehot_from_logits(pi(_next_obs))
            for pi, _next_obs in zip(self.target_policies, next_obs)
        ]
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(
            -1, 1) + self.gamma * cur_agent.target_critic(
            target_critic_input) * (1 - done[i_agent].view(-1, 1))
        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value,
                                            target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[i_agent])
        cur_act_vf_in = gumbel_softmax(cur_actor_out)
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(onehot_from_logits(pi(_obs)))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out ** 2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)


def evaluate(maddpg, n_episode=10, episode_length=25):
    # 对学习的策略进行评估,此时不会进行探索
    env = BSS_ENV3()
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        obs = env.reset()
        for t_i in range(episode_length):
            actions = maddpg.take_action(obs, explore=False)
            obs, rew, done = env.step(actions)
            rew = np.array(rew)
            returns += rew / n_episode
    return returns.tolist()


def calculate_cost(maddpg):
    for i, agent in enumerate(maddpg.agents):
        para_dict = torch.load(f"checkpoint/exp3/agent{i}.pth")
        agent.actor.load_state_dict(para_dict['actor'])
        agent.target_actor.load_state_dict(para_dict['target_actor'])
        agent.critic.load_state_dict(para_dict['critic'])
        agent.target_critic.load_state_dict(para_dict['target_critic'])
    env = BSS_ENV2()
    for i in range(10):
        path = f'init_Qs/init_Q{i}.json'
        obs = env.reset(json.load(open(path, 'r')))
        returns = np.zeros((len(env.agents), 100))
        for t_i in range(100):
            actions = maddpg.take_action(obs, explore=False)
            obs, rew, done = env.step(actions)
            # print([env.agents[n]["state"] for n in range(20)])
            returns[:, t_i] = np.array([env.agents[n]["f_n"] for n in range(len(env.agents))])
            # print(sum([env.agents[n]["f_n"] for n in range(len(env.agents))]))
        for c in returns:
            plt.plot(c)
        plt.title(f'init_Q{i}_MADDPG-sum_cost={sum([c[-1] for c in returns])}')
        plt.xlabel("iteration")
        plt.ylabel("cost")
        plt.ylim([10, 50])
        plt.savefig(f'NE_imgs/init_Q{i}_MADDPG.png')
        plt.close()
    # return returns.tolist()


if __name__ == "__main__":
    num_episodes = 5000
    episode_length = 25  # 每条序列的最大长度
    buffer_size = 100000
    hidden_dim = 64
    actor_lr = 1e-3
    critic_lr = 1e-3
    gamma = 0.99
    tau = 1e-2
    batch_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    update_interval = 100
    minimal_size = 4000

    env = BSS_ENV2()
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)

    state_dims = []
    action_dims = []
    for action_space in env.action_space:
        action_dims.append(action_space)
    for state_space in env.observation_space:
        state_dims.append(state_space)
    critic_input_dim = sum(state_dims) + sum(action_dims)

    maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims,
                    action_dims, critic_input_dim, gamma, tau)
    calculate_cost(maddpg)

    # return_list = []  # 记录每一轮的回报（return）
    # total_step = 0
    # for i_episode in range(num_episodes):
    #     state = env.reset()
    #     # ep_returns = np.zeros(len(env.agents))
    #     for e_i in range(episode_length):
    #         actions = maddpg.take_action(state, explore=True)
    #         next_state, reward, done = env.step(actions)
    #         replay_buffer.add(state, actions, reward, next_state, done)
    #         state = next_state
    #
    #         total_step += 1
    #         if replay_buffer.size() >= minimal_size and total_step % update_interval == 0:
    #             sample = replay_buffer.sample(batch_size)
    #
    #
    #             def stack_array(x):
    #                 rearranged = [[sub_x[i] for sub_x in x]
    #                               for i in range(len(x[0]))]
    #                 return [
    #                     torch.FloatTensor(np.vstack(aa)).to(device)
    #                     for aa in rearranged
    #                 ]
    #
    #
    #             sample = [stack_array(x) for x in sample]
    #             for a_i in range(len(env.agents)):
    #                 maddpg.update(sample, a_i)
    #             maddpg.update_all_targets()
    #
    #     if (i_episode + 1) % 100 == 0:
    #         for i, agent in enumerate(maddpg.agents):
    #             torch.save({'actor': agent.actor.state_dict(),
    #                         'target_actor': agent.target_actor.state_dict(),
    #                         'critic': agent.critic.state_dict(),
    #                         'target_critic': agent.target_critic.state_dict(), }, f'checkpoint/exp4/agent{i}.pth')
    #         ep_returns = evaluate(maddpg, n_episode=100)
    #         return_list.append(ep_returns)
    #         print(f"Episode: {i_episode + 1}, {ep_returns}")
    #
    # return_array = np.array(return_list)
    # for i, agent_name in enumerate([f"agent_{n}" for n in range(20)]):
    #     plt.figure()
    #     plt.plot(
    #         np.arange(return_array.shape[0]) * 100,
    #         rl_utils.moving_average(return_array[:, i], 9))
    #     plt.xlabel("Episodes")
    #     plt.ylabel("Returns")
    #     plt.title(f"{agent_name} by MADDPG")
    #     plt.savefig(f"imgs/exp4/{agent_name}_MADDPG.png")
    #     plt.show()
