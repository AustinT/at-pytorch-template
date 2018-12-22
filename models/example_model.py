import torch
from torch import nn

class SimpleFeedForward(nn.Module):
    """
    This class implements a simple feed forward network with 1 hidden layer.
    It is meant to be an example of how to define a model in an external file, then call it in the main program
    """
    def __init__(self, configs):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1, configs["n_hidden"]),
            nn.ReLU(),
            nn.Linear(configs["n_hidden"], 1))

    def forward(self, x):
        return self.seq(x)

