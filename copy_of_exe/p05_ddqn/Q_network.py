import torch
import numpy as np
import torch.nn as nn

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):

    def __init__(self, n_inputus, n_outputs, bias=True):
        super().__init__()

        self.activation = nn.Tanh()
        self.lin1 = nn.Linear(n_inputus, 64, bias=bias)
        self.lin2 = nn.Linear(64, 32, bias=bias)
        self.lin3 = nn.Linear(32, n_outputs, bias=bias)

    def forward(self, x):
        x = self.activation(self.lin1(x))
        x = self.activation(self.lin2(x))
        return self.lin3(x)


class Q_network(nn.Module):

    def __init__(self, env, lr=1e-4):
        super().__init__()
        self.network = Net(env.observation_space._shape[0], env.action_space.n)
        print(f'network shape: \n {self.network}')

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr)

    def greedy_action(self, state):
        qvals = self.get_qvals(state)
        greedy_a = torch.max(qvals, dim=-1)[1].item()
        return greedy_a

    def get_qvals(self, state):
        return self.network(state)
