import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        # Linear layers to produce Q, K, V
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        Q = self.W_Q(x)   # (batch, seq_len, embed_dim)
        K = self.W_K(x)   # (batch, seq_len, embed_dim)
        V = self.W_V(x)   # (batch, seq_len, embed_dim)

        # Compute attention scores: QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        # Weighted sum of values
        out = torch.matmul(attn_weights, V)  # (batch, seq_len, embed_dim)

        return out, attn_weights


# -------------------------
# 🔹 Example Usage
# -------------------------
batch_size = 1
seq_len = 4
embed_dim = 8

x = torch.rand((batch_size, seq_len, embed_dim))  # random input
attention = SelfAttention(embed_dim)
out, weights = attention(x)

print("Output shape:", out.shape)         # (1, 4, 8)
print("Attention Weights shape:", weights.shape)  # (1, 4, 4)
