import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

#the FCN encoder, it can take in either actions or actions + states, I left it flexible just in case we decide to try something
#bottleneck, same rank, rank increase are just dependent on what you set out_dim to be


class FCNEncoder(nn.Module): #basic FCN encoder, I just made the number of hidden layers flexible for now
    def __init__(self, action_dim, obs_dim, hidden_sizes, out_dim):
        super(FCNEncoder, self).__init__()
        if type(hidden_sizes) == int:
            hidden_sizes = [hidden_sizes]
        num_hidden_layers = len(hidden_sizes)
        self.input_dim = action_dim #+ obs_dim #if we want to go that route
        self.layers = nn.ModuleList([nn.Linear(self.input_dim, hidden_sizes[0]), nn.ReLU()])
        for i in range(1, num_hidden_layers):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_sizes[-1], out_dim))

    def forward(self, x): #x could be just the action batch, or the batch of actions and states concatenated
        for layer in self.layers:
            x = layer(x)
        return x




