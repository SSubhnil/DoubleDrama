import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEGatingMLP(nn.Module):
    def __init__(self, d_hidden, num_tr_experts, num_re_experts, hidden_dim=128, tau=1.0):
        """
        d_hidden: Dimensionality of the input hidden state.
        num_tr_experts: Number of transition experts.
        num_re_experts: Number of reward experts.
        hidden_dim: Hidden dimension for the gating MLP layers.
        tau: Temperature parameter for the Gumbel-Softmax.
        """
        super(MoEGatingMLP, self).__init__()
        self.tau = tau

        # Gating MLP for transition experts
        self.mlp_tr = nn.Sequential(
            nn.Linear(d_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tr_experts)
        )

        # Gating MLP for reward experts
        self.mlp_re = nn.Sequential(
            nn.Linear(d_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_re_experts)
        )

    def forward(self, h):
        """
        h: Hidden states with shape (batch_size, seq_length, d_hidden).
        Returns:
            gating_tr: Gating probabilities for transition experts of shape (B, T, num_tr_experts).
            gating_re: Gating probabilities for reward experts of shape (B, T, num_re_experts).
        """
        # Compute gating logits from the hidden state
        alpha_tr = self.mlp_tr(h)  # (B, T, num_tr_experts)
        alpha_re = self.mlp_re(h)  # (B, T, num_re_experts)

        # Apply the Gumbel-Softmax to obtain differentiable discrete assignments
        gating_tr = F.gumbel_softmax(alpha_tr, tau=self.tau, hard=False, dim=-1)
        gating_re = F.gumbel_softmax(alpha_re, tau=self.tau, hard=False, dim=-1)

        return gating_tr, gating_re


###############################################
# Example Usage
###############################################
if __name__ == "__main__":
    batch_size = 8
    seq_length = 10
    d_hidden = 256
    num_tr_experts = 4  # e.g., 4 transition experts
    num_re_experts = 3  # e.g., 3 reward experts

    # Create dummy hidden states from a transformer (B, T, d_hidden)
    dummy_hidden = torch.randn(batch_size, seq_length, d_hidden)

    # Instantiate the MoE gating module
    moe_gating = MoEGatingMLP(d_hidden, num_tr_experts, num_re_experts, hidden_dim=128, tau=0.5)

    gating_tr, gating_re = moe_gating(dummy_hidden)

    print("Transition gating probabilities shape:", gating_tr.shape)  # Expected: (8, 10, 4)
    print("Reward gating probabilities shape:", gating_re.shape)  # Expected: (8, 10, 3)
