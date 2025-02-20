import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.ReLU, init_type='kaiming'):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(input_dim, output_dim))

            if i < len(dims)-2: # NO activation after final layer
                layers.append(activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)