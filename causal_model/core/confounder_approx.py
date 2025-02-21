import torch
import torch.nn as nn
import torch.nn.functional as F
from enc_dec import CausalEncoder_Confounder
"""
Posterior computation only.
See end_dec.py for regularized confounder prior.
"""

class PosteriorHypernet(nn.Module):
    """Generates posterior network weights conditioned on the code IDs"""
    def __init__(self, code_dim, hidden_dim, out_dim):
        super().__init__()
        self.weight_gen = nn.Sequential(nn.Linear(code_dim, 4*hidden_dim*out_dim),
                                        nn.Unflatten(-1, (4, hidden_dim, out_dim))
                                        )

    def forward(self, x, code_emb):
        # Generate dynamic weights [4, hidden_dim, out_dim]
        weights = self.weight_gen(code_emb)
        x = F.linear(x, weights[0], bias=None)
        x = F.silu(x)
        x = F.linear(x, weights[1], bias=None)
        return x

class ConfounderApproximator(nn.Module):
    """
    We implement Part 2B of report
    Components:
        - Code-conditioned variational posterior Q(u | h, c)
        - Hyper-network for parameter generation
        - Affine transformations code specific \phi_c(u)
    """

    def __init__(self, code_dim: int, conf_dim: int, num_codes: int, params,
                 embed_dim: int = None, hidden_dim: int = 256):
        super().__init__()
        self.encoder = CausalEncoder_Confounder(params)

        self.embed_dim = embed_dim or code_dim // 2 # Default to code_dim/2
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.conf_dim = conf_dim
        self.num_codes = num_codes

        # Code embedding layer
        self.h_proj = nn.Linear(self.code_dim, self.hidden_dim)
        self.code_proj = nn.Linear(self.embed_dim, self.hidden_dim)
        self.code_embed = nn.Embedding(self.num_codes, self.embed_dim)

        # Posterior networks
        self.confounder_post_mu_net = PosteriorHypernet(self.code_dim, self.hidden_dim, self.conf_dim)

        self.confounder_post_logvar_net = PosteriorHypernet(self.code_dim, self.hidden_dim, self.conf_dim)

        # self.confounder_post_mu_net = nn.Sequential(
        #     nn.Linear(code_dim * 2, self.hidden_dim), # Concat code dim + embed_dim
        #     nn.SiLU(inplace=True),
        # nn.Linear(self.hidden_dim, conf_dim), # posterior mu
        # )

        # self.confounder_post_logvar_net = nn.Sequential(
        #     nn.Linear(code_dim *2, self.hidden_dim),
        #     nn.SiLU(inplace=True), # \sigma \in (0, \inf)
        #     nn.Linear(self.hidden_dim, conf_dim),
        #     nn.Softplus() # Optionally constraint to (-a, a) if needed
        # )
        nn.init.constant_(self.confounder_post_logvar_net[-2].bias, -1.0)

        # Code-specific affine parameters  \phi_c(u) = s_c \odot u + t_c
        self.affine_scale = nn.Embedding(num_codes, conf_dim)
        self.affine_shift = nn.Embedding(num_codes, conf_dim)

        # Initialize near identity transform
        nn.init.normal_(self.affine_scale.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.affine_scale.weight[:, :conf_dim // 2], 0.0)  # Partial identity
        nn.init.uniform_(self.affine_shift.weight, -0.01, 0.01)

    def forward(self, h: torch.Tensor, code_ids: torch.Tensor):
        """
        Args:
            h: Hidden state from world model [B, D]
            code_ids: Discrete code indices [B]

        Returns:
            u_transformer: Confounder after code-specific affine [B, conf_dim]
            kl_loss: D_KL(Q(u|h,c) || P(u|h)) + regularization
        """
        # Call confounder prior network
        from_confounder_prior = self.encoder(h, code_ids)
        mu_prior = from_confounder_prior['confounder_mu']
        logvar_prior = from_confounder_prior['confounder_logvar']

        # Hypernetwork based posterior
        code_emb = self.code_embed(code_ids)  # [B, embed_dim]
        mu_post = self.confounder_post_mu_net(h, code_emb)
        logvar_post = self.confounder_post_logvar_net(h, code_emb)

        # Reparameterization trick
        u_post = torch.normal(mu_post, torch.exp(0.5 * logvar_post))  # ~15% faster on CUDA

        kl_loss = self.gaussian_KL(mu_post, logvar_post, mu_prior, logvar_prior)

        return u_post, kl_loss

    # Redundant due to non-use of scale and shift
    def regularization_loss_2(self, lamdba_weight=0.01):
        # Encourage sparse affine transformations
        scale_reg = torch.mean(torch.abs(self.affine_scale.weight))
        shift_reg = torch.mean(torch.abs(self.affine_shift.weight))
        return lamdba_weight * (scale_reg + shift_reg)

    # Main regularizer
    def regularization_loss(self):
        # Hypernetwork parameter regularization
        hyper_params = list(self.confounder_post_mu_net.parameters()) + \
                       list(self.confounder_post_logvar_net.parameters())

        l2_reg = torch.sum(torch.stack([torch.norm(p) for p in hyper_params]))

        # Code embedding sparsity
        code_sparsity = torch.mean(torch.abs(self.code_embed.weight))

        return 0.001*l2_reg + 0.01*code_sparsity

    def gaussian_KL(self, mu_post: torch.Tensor, logvar_post: torch.Tensor,
                    mu_prior: torch.Tensor, logvar_prior: torch.Tensor):
        var_prior = torch.exp(logvar_prior) + 1e-6
        var_post = torch.exp(logvar_post) + 1e-6
        logvar_ratio = logvar_prior - logvar_post

        kl_loss = 0.5 * ((mu_post - mu_prior).pow(2) / var_prior +
                         var_post/var_prior - 1- logvar_ratio)

        return kl_loss.sum(1).mean() # Averages over batch


