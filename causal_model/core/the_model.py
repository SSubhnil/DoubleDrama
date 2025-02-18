import numpy as np
import torch.nn as nn
import torch.nn.init as init
from enc_dec import CausalEncoder, CausalEncoder_Confounder, CausalDecoder
class CausalModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.encoder = self.create_causal_encoder(params)
        self.decoder = self.create_causal decoder(params)

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