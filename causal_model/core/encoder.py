import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class HiddenStateEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.hidden_state_dim = params.hidden_state_dim
        self.feature_dim = self.hidden_state_dim

        # Linear layer for embeddings
        self.hidden_embed_projection = nn.Linear(self.hidden_state_dim, desired_dim)
        # No embedding
        self.hidden_embed = nn.Identity()
        self.to(params.device)

    def forward(self, hidden_state, detach=False):
        return self.project(hidden_state)


def make_encoder(params):
    return HiddenStateEncoder(params)