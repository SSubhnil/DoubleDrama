import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import MLP

class CausalEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.hidden_state_dim = params.hidden_state_dim
        self.feature_dim = params.code_dim

        # Feature Projection Layers
        self.tr_proj = MLP(self.hidden_state_dim, self.feature_dim, [self.hidden_state_dim//2],
                           activation=nn.SiLU)
        self.re_proj = MLP(self.hidden_state_dim, self.feature_dim, [self.hidden_state_dim//2],
                           activation=nn.SiLU)

    def forward(self, h):
        # Extract Features
        return{'tr_features': self.tr_proj(h), 're_features': self.re_proj(h)}

# Initialized for confounder prior
class CausalEncoder_Confounder(CausalEncoder):
    def __init__(self, params):
        super().__init__(params)

        # Confounder prior is a Gaussian dist conditioned on the hidden state.
        # Confounder parameters (mi, sig) generator
        # Define Confounder Prior Network
        self.confounder_prior_mu = nn.Sequential(
            nn.Linear(self.hidden_state_dim, self.feature_dim),
            nn.Dropout(p=0.1)) # Removed tanh() for better adaptability

        self.confounder_prior_logvar = nn.Sequential(
            nn.Linear(self.hidden_state_dim, self.feature_dim),
            nn.Softplus(), # Smooth positive training
            nn.Linear(self.feature_dim, self.feature_dim) # Learned scale factor
        )
        nn.init.constant_(self.confounder_prior_logvar[-1].weight, 0.5) # Initial scale

    def forward(self, h: torch.Tensor, code_ids: torch.Tensor = None):
        base_features = super().forward(h)

        # Generate confounder prior distribution parameters
        mu = self.confounder_prior_mu(h)
        logvar = self.confounder_prior_logvar(h)

        return {**base_features,
                'confounder_mu': mu,
                'confounder_logvar': logvar,
                }

class CausalDecoder(nn.Module):
    def __init__(self, params, use_confounder=False):
        super().__init__()
        self.use_confounder = use_confounder
        input_dim = params.code_dim * 2 + (params.feature_dim if use_confounder else 0)

        self.transition_net = MLP(input_dim, params.hidden_state_dim,
                                  [params.hidden_state_dim *2], activation=nn.SiLU)

    def forward(self, tr_code, re_code, confounder=None):
        inputs = [tr_code, re_code]
        if self.use_confounder and confounder is not None:
            inputs.append(confounder)

        return self.transition_net(torch.cat(inputs, dim=-1))