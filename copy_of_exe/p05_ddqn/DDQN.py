import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import deepcopy

from Q_network import Q_network
from utils import from_tuple_to_tensor


class DDQN_agent:

    def __init__(self, env, rew_thre, buffer, lr=0.001, init_epsilon=0.5, batch_size=64):

        self.env = env
        self.network = Q_network(env, lr)
        self.target_network = deepcopy(self.network)
        self.buffer = buffer
        self.batch_size = batch_size
        self.window = 50
        self.reward_threshold = rew_thre
        self.step_count = 0
        self.episode = 0
        self.epsilo = init_epsilon

        self.initialize()

    def take_step(self, mode='exploit'):

        if mode == 'exploit':
            action = self.env.action_space.sample()
        else:
            action = self.network.greedy_action(torch.FloatTensor(self.s_0))

        s_1, r, done, _ = self.env.step(action)                 # SIMULATE
        self.buffer.append(self.s_0, action, r, done, s_1)      # BUFFERIZE
        self.rewards += r
        self.s_0 = s_1.copy()
        self.step_count += 1

        if done:
            self.s_0 = self.env.reset()
        return done

    def train(self, gamma=0.99, max_ep=10000, network_update_freq=10, network_sync_freq=200):

        self.gamma = gamma
        self.loss_funct = nn.MSELoss()
        self.s_0 = self.env.reset()

        while self.buffer.burn_in_capacity() < 1:  # POPULATE BUFFER
            self.take_step(mode='explore')

        ep = 0
        training = True
        self.populate = False

        while training:

            self.s_0 = self.env.reset()
            self.reward = 0
            done = False

            while not done:
                if (ep % 5) == 0:
                    self.env.render()

                p = np.random.random()
                if p < self.epsilon:
                    done = self.take_step(mode='explore')
                else:
                    done = self.take_step(mode='exploit')

                if self.step_count % network_update_freq == 0:
                    self.update()
                if self.step_count % network_sync_freq == 0:
                    self.target_network.loade_state_dict(self.network.state_dict())
                    self.sync_eps.append(ep)

                if done:
                    self.epsilon = max(0.05, self.epsilon * 0.7)
                    ep += 1

                    self.update_training_rewards()

                    if len(self.update_loss) == 0:
                        self.training_loss.append(0)
                    else:
                        self.training_loss.append(np.mean(self.update_loss))

                    mean_rewards = np.mean(self.training_rewards[-self.window:])
                    mean_loss = np.mean(self.training_loss[-self.window:])
                    self.mean_training_rewards.append(mean_rewards)

                    print("\rEpisode {:d} Mean Rewards {:.2f}  Episode reward = {:.2f}   mean loss = {:.2f}\t\t".format(
                        ep, mean_rewards, self.rewards, mean_loss), end="")

                    if ep >= max_ep:
                        training = False
                        print("\nmax ep reached")
                        break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print(f"\nEnvironment solved in {ep} episodes")

                    self.save_models()
                    self.plot_training_rewards()

    def save_models(self):
        torch.save(self.network, 'Q_net')

    def load_models(self):
        self.network = torch.load('Q_net')
        self.network.eval()

    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states = list(batch)

        rewards = torch.FloatTensor(rewards).reshape(-1, 1)
        actions = torch.LongTensor(actions).reshape(-1, 1)
        dones = torch.IntTensor(dones).reshape(-1, 1)
        states = from_tuple_to_tensor(states)
        next_states = from_tuple_to_tensor(next_states)

        qvals = self.network.get_qvals(states)
        qvals = torch.gather(qvals, 1, actions)

        next_qvals = self.target_network.get_qvals(states)
        next_qvals_max = torch.max(next_qvals, dim=-1)[0].reshape(-1, 1)
        target_qvals = rewards + (1 - dones) * self.gamma * next_qvals_max
        loss = self.loss_funct(qvals, target_qvals)

    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calcualate_loss(batch)

        loss.backward()
        self.network.optimizer.step()
        self.update_loss.append(loss.item)

    def update_training_rewards(self):
        if self.rewards > 2000:
            self.training_rewards.append(2000)
        elif self.rewards > 1000:
            self.training_rewards.append(1000)
        elif self.rewards > 500:
            self.training_rewards.append(500)
        else:
            self.training_rewards.append(self.rewards)
        if len(self.update_loss) == 0:
            self.training_loss.append(0)
        else:
            self.training_loss.append(np.mean(self.update_loss))

    def initialize(self):
        self.training_rewards = []
        self.training_loss = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0

    def plot_training_rewards(self):
        plt.plot(self.mean_training_rewards)
        plt.title('Mean training rewards')
        plt.ylabel('Reward')
        plt.xlabel('Episods')
        plt.show()
        plt.savefig('mean_training_rewards.png')
        plt.clf()

    def evaluate(self, eval_env):
        done = False
        s = eval_env.reset()
        rewards_eval = 0
        while not done:
            action = self.network.greedy_action(torch.FloatTensor(s))
            s, r, done, _, _ = eval_env.step(action)
            rewards_eval += r
        print(f'eval cumulative reward : {rewards_eval}')
