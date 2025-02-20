import numpy as np
import torch.nn as nn
import torch.nn.init as init
from enc_dec import CausalEncoder, CausalEncoder_Confounder, CausalDecoder
from quantizer import DualVQQuantizer
from confounder_approx import ConfounderApproximator
from networks import MLP

class CausalInterface(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        # Core modules
        self.quantizer = DualVQQuantizer(code_dim=params.code_dim,
                                         num_codes_tr=params.num_codes_tr,
                                         num_codes_re=params.num_codes_re,
                                         coupling=True)

        self.encoder = self.create_causal_encoder(self.params)

        self.confounder_net = ConfounderApproximator(code_dim=params.code_dim,
                                                     conf_dim=params.conf_dim,
                                                     num_codes=params.num_codes_tr,
                                                     params=self.params)

        # Hidden state modulation network
        self.state_mod = nn.Sequential(nn.Linear(params.code_dim*2 + params.conf_dim,
                                                 params.hidden_dim), nn.SiLU(inplace=True),
                                       nn.Linear(params.hidden_dim, params.hidden_dim*2))




    def forward(self, h):
        enc_output = self.encoder(h)
        next_h =  self.decoder(enc_output['tr_features'],
                               enc_output['re_features'],
                               enc_output.get('confounder_sample')
                               )
        return next_h, enc_output

    def create_causal_encoder(self, params):
        if params.use_confounder:
            return CausalEncoder_Confounder(params)
        return CausalEncoder(params)

    def create_causal_decoder(self, params):
        return CausalDecoder(params, use_confounder=params.use_confounder)