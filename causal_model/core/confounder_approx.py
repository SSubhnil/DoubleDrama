import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Posterior computation only.
See end_dec.py for regularized confounder prior.
"""

class ConfounderApproximator(nn.Module):
    """
    We implement Part 2B of report
    Components:
        - Code-conditioned variational posterior Q(u | h, c)
        - Hyper-network for parameter generation
        - Affine transformations code specific \phi_c(u)
    """

    def __init__(self, code_dim: int, conf_dim: int, num_codes: int):
        super().__init__()
        self.conf_dim = conf_dim
        self.num_codes = num_codes

        # Code embedding layer
        self.code_embed = nn.Embedding(num_codes, 128)

        # Posterior networks
        self.confounder_post_mu_net = nn.Sequential(
            nn.Linear(128 + code_dim, 256), # Concat code emb + hidden state
            nn.SiLU(),
        nn.Linear(256, conf_dim), # posterior mu
        )

        self.confounder_post_logvar_net = nn.Sequential(
            nn.Linear(128 + code_dim, 256),
            nn.SiLU(),
            nn.Linear(256, conf_dim),
            nn.Hardtanh(min_val=-6, max_val=0) # For stability
        )

        # Code-specific affine parameters  \phi_c(u) = s_c \odot u + t_c
        self.affine_scale = nn.Embedding(num_codes, conf_dim)
        self.affine_shift = nn.Embedding(num_codes, conf_dim)

        # Initialize near identity transform
        nn.init.normal_(self.affine_scale.weight, mean=1.0, std=-0.1) # s_c \approx 1
        nn.init.zeros_(self.affine_shift.weight) # t_c \approx 0

    def forward(self, h: torch.Tensor, code_ids: torch.Tensor):
        """
        Args:
            h: Hidden state from world model [B, D]
            code_ids: Discrete code indices [B]

        Returns:
            u_transformer: Confounder after code-specific affine [B, conf_dim]
            kl_loss: D_KL(Q(u|h,c) || P(u|h)) + regularization
        """

        code_emb = self.code_embed(code_ids) # [B, 128]

        # Concat h with code emb for posterior conditioning
        h_code = torch.cat([h, code_emb], dim=1) # [B, D+128]

        # Compute posterior parameters
        mu_post = self.confounder_post_mu_net(h_code)
        logvar_post = self.confounder_post_logvar_net(h_code) # Hardtanh constrained

        # Reparameterization trick
        std = torch.exp(0.5 * logvar_post)
        eps = torch.randn_like(std)
        u = mu_post + eps * std # Differentiable sampling

        # Apply code-specific affine transformation
        scale = self.affine_scale(code_ids) # [B, conf_dim]
        shift = self.affine_shift(code_ids)
        u_transformed = scale * u + shift # Element-wise ops

        # KL-divergence
        kl_loss = -0.5 * torch.sum(1 + logvar_post - mu_post.pow(2) - logvar_post.exp())
        kl_loss = kl_loss / h.size(0) # Batch average

        return u_transformed, kl_loss

    def regularization_loss(self, lamdba_weight=0.01):
        """Encourage sparse affine transformations"""
        l1_shift = torch.norm(self.affine_shift.weight, p=1)
        l1_scale = torch.norm(self.affine_scale.weight - 1.0, p=1) #  Peanlize deviation from 1
        return lamdba_weight * (l1_shift + l1_scale)


