# decoder_min.py
# Minimal Transformer Decoder (PyTorch) with masked self-attention + cross-attention

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Positional Encoding ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                       # [T, D]
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [T, 1]
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))              # [1, T, D]

    def forward(self, x):                                        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------- Multi-Head Attention ----------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask: torch.Tensor | None = None):
        # Q,K,V: [B, T, D]; attn_mask (optional): broadcastable to [B, H, T_q, T_k]; 1=keep, 0=mask
        B, T_q, D = Q.size()
        T_k = K.size(1)
        H, d_k = self.num_heads, self.d_k

        q = self.W_q(Q).view(B, T_q, H, d_k).transpose(1, 2)     # [B, H, T_q, d_k]
        k = self.W_k(K).view(B, T_k, H, d_k).transpose(1, 2)     # [B, H, T_k, d_k]
        v = self.W_v(V).view(B, T_k, H, d_k).transpose(1, 2)     # [B, H, T_k, d_k]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)      # [B, H, T_q, T_k]
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = attn @ v                                           # [B, H, T_q, d_k]
        ctx = ctx.transpose(1, 2).contiguous().view(B, T_q, D)   # [B, T_q, D]
        return self.W_o(ctx)


# ---------- Feed-Forward ----------
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# ---------- Decoder Block ----------
class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        # x: [B, T_tgt, D], enc_out: [B, T_src, D]
        # self_mask: [B, 1, T_tgt, T_tgt] (causal & pad), cross_mask: [B, 1, T_tgt, T_src]
        sa = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(sa))

        ca = self.cross_attn(x, enc_out, enc_out, cross_mask)
        x = self.norm2(x + self.dropout(ca))

        ff = self.ffn(x)
        x = self.norm3(x + self.dropout(ff))
        return x


# ---------- Full Decoder (stacked blocks) ----------
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, num_heads: int = 4,
                 d_ff: int = 256, num_layers: int = 2, max_len: int = 512,
                 dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    # ---- mask helpers ----
    def make_pad_mask(self, x: torch.Tensor, pad_idx: int):
        # x: [B, T]; returns [B, 1, 1, T] where 1=keep, 0=mask
        return (x != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_causal_mask(self, T: int, device):
        # [1, 1, T, T] with 1 in lower triangle (including diag)
        m = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
        return m

    def forward(self, tgt_tokens, enc_out, src_tokens=None):
        # tgt_tokens: [B, T_tgt], enc_out: [B, T_src, D], src_tokens: [B, T_src] (for pad mask)
        B, T_tgt = tgt_tokens.size()
        T_src = enc_out.size(1)

        # embeddings + positions
        h = self.tok_emb(tgt_tokens) * math.sqrt(self.d_model)
        h = self.pos_enc(h)
        h = self.dropout(h)
