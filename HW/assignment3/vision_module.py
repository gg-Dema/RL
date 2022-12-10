import torch
from torch import nn


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.shape[0], -1)


class UnFlatten(nn.Module):

    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):

    def __init__(self, image_channels=1, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )  # .to(device)

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def sampling_trick(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  #.to(device)
        esp = torch.randn(mu.size())  #.to(device)
        z = mu + std * esp
        return z

    def encode(self, x):
        h = self.encoder(x)                     # 256x4 per singola img
        # mu, log_var = self.fc1(h.view(-1)), self.fc2(h.view(-1)) # OK per singola img
        mu, log_var = self.fc1(h), self.fc2(h)  # batch mode
        z = self.sampling_trick(mu, log_var)
        return z, mu, log_var

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


