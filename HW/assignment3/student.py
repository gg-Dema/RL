import torch
import torch.nn as nn

from torchvision import transforms
from memory_module import MDNRNN
from vision_module import VAE, Flatten, UnFlatten


class Policy(nn.Module):


    continuous = True

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()

        self.device = device
        self.vision = VAE()
        self.memory = MDNRNN()
        self.controller = torch.nn.Sequential(
            torch.nn.Linear(256+32, 3),
            torch.nn.Tanh())

        self.load()

        self.last_act = torch.tensor([[0., 0., 0.]])
        self.h = self.memory.init_hidden(1)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize(64),
            transforms.ToTensor()
            ])

    def forward(self, x):
        z, _, _ = self.vision.encode(torch.unsqueeze(x, 0))  # z_shape([1, 32])
        temp = torch.unsqueeze( torch.cat( (z, self.last_act), -1 ), -2)
        _, self.h = self.memory(temp, self.h)
        return self.controller(torch.cat((z, self.h[0].view(1, -1)), -1))

    def act(self, state):
        # preprocessing
        state = self.transform(state)
        # predict
        self.last_act = self.forward(state)
        return self.last_act.view(-1).detach().numpy()

    def train(self):
        raise NotImplementedError('il train è stato eseguito su colab, vedere report')

    def save(self):
        torch.save(self, './model_weights/model.torch')
    def load(self):
        self = torch.load('./model_weights/model.torch', map_location='cpu')

    def save_2(self):
        torch.save(self.vision, './model_weights/vae.torch')
        torch.save(self.memory, './model_weights/rnn-mdn_14_ep.torch')
        torch.save(self.controller, './model_weights/controller.torch')

    def load_2(self):
        self.vision = torch.load('./model_weights/vae.torch', map_location='cpu')
        self.memory = torch.load('./model_weights/rnn-mdn_14_ep.torch', map_location='cpu')
        self.controller = torch.load('./model_weights/controller.torch', map_location='cpu')

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
