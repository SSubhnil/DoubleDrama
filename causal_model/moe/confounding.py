import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfounderModule(nn.Module):
    def __init__(self, d_hidden, d_u, num_tr_experts):
        """
        d_hidden: Dimensionality of the input hidden state (e.g., from the causal transformer).
        d_u: Dimensionality of the confounder variable u.
        num_tr_experts: Number of transition experts.
        """
        super(ConfounderModule, self).__init__()
        # Variational posterior network: outputs concatenated mu and logvar for u
        self.posterior_net = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 2 * d_u)  # outputs [mu, logvar]
        )

        # Learnable global prior parameters for p0(u) = N(mu0, sigma0)
        self.mu0 = nn.Parameter(torch.zeros(d_u))
        self.logvar0 = nn.Parameter(torch.zeros(d_u))  # log-variance

        # For each transition expert, an affine transform T_k(u) = A_k u + b_k
        self.num_tr_experts = num_tr_experts
        self.A = nn.Parameter(torch.eye(d_u).unsqueeze(0).repeat(num_tr_experts, 1, 1))
        self.b = nn.Parameter(torch.zeros(num_tr_experts, d_u))

    def forward(self, h):
        """
        h: Hidden state tensor of shape (B, T, d_hidden)
        Returns:
            u: Sampled confounder from q(u|h) of shape (B, T, d_u)
            u_affine: Affine-transformed confounder per expert of shape (B, T, num_tr_experts, d_u)
            kl: The KL divergence between q(u|h) and the base prior p0(u) (averaged over batch and time)
        """
        B, T, _ = h.size()
        # Compute posterior parameters: mu and logvar for u
        posterior = self.posterior_net(h)  # shape: (B, T, 2*d_u)
        mu, logvar = torch.chunk(posterior, 2, dim=-1)  # each: (B, T, d_u)
        sigma = torch.exp(0.5 * logvar)

        # Reparameterize: sample epsilon ~ N(0, I) and compute u = mu + sigma * eps
        eps = torch.randn_like(sigma)
        u = mu + sigma * eps  # shape: (B, T, d_u)

        # Apply per-expert affine transforms: u_affine[b,t,k] = A_k u[b,t] + b_k
        # Resulting shape: (B, T, num_tr_experts, d_u)
        u_affine = torch.einsum('btd,kde->btke', u, self.A) + self.b.unsqueeze(0).unsqueeze(0)

        # Compute KL divergence between q(u|h)=N(mu, sigma^2) and p0(u)=N(mu0, sigma0^2)
        sigma0 = torch.exp(0.5 * self.logvar0)
        mu0 = self.mu0.view(1, 1, -1)
        sigma0 = sigma0.view(1, 1, -1)
        logvar0 = self.logvar0.view(1, 1, -1)

        # KL divergence element-wise: log(sigma0/sigma) + (sigma^2 + (mu-mu0)^2)/(2*sigma0^2) - 0.5
        kl_element = logvar0 - logvar + (sigma ** 2 + (mu - mu0) ** 2) / (2 * sigma0 ** 2) - 0.5
        kl = torch.sum(kl_element, dim=-1)  # shape: (B, T)
        kl = torch.mean(kl)  # averaged over batch and time

        return u, u_affine, kl


###############################################
# Example Usage
###############################################
if __name__ == "__main__":
    batch_size = 8
    seq_length = 10
    d_hidden = 256
    d_u = 32
    num_tr_experts = 4

    # Dummy hidden states from the causal transformer: shape (B, T, d_hidden)
    dummy_hidden = torch.randn(batch_size, seq_length, d_hidden)

    confounder_module = ConfounderModule(d_hidden, d_u, num_tr_experts)
    u, u_affine, kl_div = confounder_module(dummy_hidden)

    print("Sampled confounder shape:", u.shape)  # Expected: (8, 10, 32)
    print("Affine-transformed confounder shape:", u_affine.shape)  # Expected: (8, 10, 4, 32)
    print("KL divergence:", kl_div.item())
