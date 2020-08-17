import numpy as np
import torch
import torch.nn as nn


def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, 0.0, 1.0 / np.sqrt(m.weight.shape[0]))
        nn.init.constant_(m.bias.data, 0.0)


class Linear_Generator(nn.Module):
    INPUT_SIZE = 64
    NUM_NEURONS_HIDDEN = 32

    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(self.INPUT_SIZE, self.NUM_NEURONS_HIDDEN), nn.ReLU(),
            nn.Linear(self.NUM_NEURONS_HIDDEN, self.NUM_NEURONS_HIDDEN), nn.ReLU(),
            nn.Linear(self.NUM_NEURONS_HIDDEN, self.NUM_NEURONS_HIDDEN), nn.ReLU(),
            nn.Linear(self.NUM_NEURONS_HIDDEN, self.NUM_NEURONS_HIDDEN), nn.ReLU(),
            nn.Linear(self.NUM_NEURONS_HIDDEN, 2),
            nn.Sigmoid()
        )
        self.apply(weights_init)
    
    def forward(self, x):
        return self.main(x)


class Linear_Discriminator(nn.Module):
    NUM_NEURONS_HIDDEN = 32

    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(2, self.NUM_NEURONS_HIDDEN), nn.ReLU(),
            nn.Linear(self.NUM_NEURONS_HIDDEN, self.NUM_NEURONS_HIDDEN), nn.ReLU(),
            nn.Linear(self.NUM_NEURONS_HIDDEN, self.NUM_NEURONS_HIDDEN), nn.ReLU(),
            nn.Linear(self.NUM_NEURONS_HIDDEN, self.NUM_NEURONS_HIDDEN), nn.ReLU(),
            nn.Linear(self.NUM_NEURONS_HIDDEN, 1),
            nn.Sigmoid()
        )
        self.apply(weights_init)
    
    def forward(self, x):
        return self.main(x)


def log_loss_generator(DGz: torch.Tensor) -> torch.Tensor:
    loss = - torch.log(DGz)
    return loss.mean()

def log_loss_discriminator(Dx: torch.Tensor,
                           DGz: torch.Tensor) -> torch.Tensor:
    loss = - torch.log(Dx) - torch.log(1 - DGz)
    return loss.mean()

