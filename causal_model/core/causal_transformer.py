import math
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################
# Multi-Head Causal Self-Attention Module
###############################################
class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        """
        d_model: model (embedding) dimension.
        n_heads: number of attention heads. Must divide d_model.
        """
        super(MultiHeadCausalSelfAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head

        # Linear projections for queries, keys, and values.
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        # Note: We omit the final projection to reduce overfitting as described.

    def forward(self, x):
        """
        x: Input tensor of shape (B, T, d_model) where B is the batch size,
           T is the sequence length.
        """
        B, T, _ = x.size()

        # Linear projections
        Q = self.q_linear(x)  # shape: (B, T, d_model)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # Reshape and transpose to get (B, n_heads, T, d_k)
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Compute scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores shape: (B, n_heads, T, T)

        # Create a causal mask (upper triangular with -infty in masked positions)
        # This mask has shape (T, T) and is broadcast to (B, n_heads, T, T)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        scores = scores + causal_mask

        # Softmax over the last dimension to get attention weights
        attn = torch.softmax(scores, dim=-1)

        # Multiply attention weights with V to get the output per head
        out = torch.matmul(attn, V)  # shape: (B, n_heads, T, d_k)

        # Concatenate all heads (transpose and reshape back to (B, T, d_model))
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return out

###############################################
# Transformer Layer (Core Block)
###############################################
class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        """
        A single transformer layer that applies multi-head causal self-attention,
        followed by a residual connection and layer normalization.
        """
        super(TransformerLayer, self).__init__()
        self.mha = MultiHeadCausalSelfAttention(d_model, n_heads)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Compute self-attention and add a residual connection followed by layer norm.
        attn_out = self.mha(x)
        out = self.layer_norm(x + attn_out)
        return out

###############################################
# Positional Encoding Module (Learnable)
###############################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        """
        Learnable positional embeddings.
        """
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        """
        x: Tensor of shape (B, T, d_model)
        Returns x with positional embeddings added.
        """
        B, T, _ = x.size()
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embedding(positions)
        return x + pos_emb

###############################################
# Token Generator (MLP) and Transformer Core
###############################################
class TransformerCore(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers, max_seq_len=512):
        """
        input_dim: Dimensionality of the concatenated input [z_t, a_t, r_{t-1}].
        d_model: Transformer embedding dimension.
        n_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        max_seq_len: Maximum sequence length for positional encoding.
        """
        super(TransformerCore, self).__init__()
        # Token generator: maps concatenated features to d_model embedding.
        self.token_mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        # Stack of transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads) for _ in range(num_layers)
        ])
        # Final layer normalization (optional)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, features):
        """
        features: Tensor of shape (B, T, input_dim) representing the concatenated [z_t, a_t, r_{t-1}].
        Returns:
            hidden_states: Tensor of shape (B, T, d_model) representing the transformer core outputs.
        """
        # Generate tokens from input features
        tokens = self.token_mlp(features)  # (B, T, d_model)
        # Add positional encoding
        tokens = self.pos_encoding(tokens)
        # Pass through transformer layers
        for layer in self.layers:
            tokens = layer(tokens)
        # Optionally, apply a final layer normalization
        hidden_states = self.final_layer_norm(tokens)
        return hidden_states

###############################################
# Example Usage
###############################################
if __name__ == "__main__":
    # Example dimensions (adjust as needed)
    batch_size = 8
    seq_len = 20
    input_dim = 64    # e.g., concatenated dimension of [z_t, a_t, r_{t-1}]
    d_model = 128
    n_heads = 8
    num_layers = 4

    # Create a dummy input: (B, T, input_dim)
    dummy_features = torch.randn(batch_size, seq_len, input_dim)

    # Instantiate the Transformer Core
    transformer_core = TransformerCore(input_dim, d_model, n_heads, num_layers, max_seq_len=seq_len)

    # Forward pass
    hidden_states = transformer_core(dummy_features)
    print("Output hidden states shape:", hidden_states.shape)
    # Expected shape: (batch_size, seq_len, d_model)
