from torch import nn
from torch.nn import functional as F
import torch


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MDNRNN(nn.Module):



    def __init__(self, z_size=32, act_size=3, n_hidden=256, n_gaussians=5, n_layers=1):
        super(MDNRNN, self).__init__()

        self.z_size = z_size
        self.act_size = act_size  # vect[sterzo, accelleratore, freno]
        self.input_size = z_size + act_size
        self.n_gaussians = n_gaussians
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(self.input_size, n_hidden, n_layers, batch_first=True)
        self.fc1 = nn.Linear(n_hidden, n_gaussians * self.input_size)
        self.fc2 = nn.Linear(n_hidden, n_gaussians * self.input_size)
        self.fc3 = nn.Linear(n_hidden, n_gaussians * self.input_size)

    def get_mixture_coef(self, y):
        rollout_length = y.size(1)
        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)

        pi = pi.view(-1, rollout_length, self.n_gaussians, self.input_size)
        mu = mu.view(-1, rollout_length, self.n_gaussians, self.input_size)
        sigma = sigma.view(-1, rollout_length, self.n_gaussians, self.input_size)

        pi = F.softmax(pi, 2)
        sigma = torch.exp(sigma)
        return pi, mu, sigma

    def forward(self, x, h):
        y, (h, c) = self.lstm(x, h)
        pi, mu, sigma = self.get_mixture_coef(y)
        return (pi, mu, sigma), (h, c)

    def init_hidden(self, batch):
        return (torch.zeros(self.n_layers, batch, self.n_hidden), # .to(device),
                torch.zeros(self.n_layers, batch, self.n_hidden)) # .to(device))
