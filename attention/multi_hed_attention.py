import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Final projection after concatenation
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections
        Q = self.W_q(Q)  # (batch, seq_len, d_model)
        K = self.W_k(K)
        V = self.W_v(V)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)  # (batch, heads, seq_len, d_k)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)  # (batch, heads, seq_len, d_k)

        # Concatenate heads
        output = output.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)  # (batch, seq_len, d_model)

        # Final linear layer
        return self.W_o(output)

# ---- Test ----
batch_size = 2
seq_len = 5
d_model = 16
num_heads = 4

mha = MultiHeadAttention(d_model, num_heads)
Q = torch.rand(batch_size, seq_len, d_model)
K = torch.rand(batch_size, seq_len, d_model)
V = torch.rand(batch_size, seq_len, d_model)

output = mha(Q, K, V)
print("Output shape:", output.shape)  # (2, 5, 16)
