# transformer_encoder_min.py
# Minimal, self-contained Transformer Encoder (PyTorch)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Positional Encoding (sinusoidal)
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)            # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)                           # [1, max_len, d_model]
        self.register_buffer("pe", pe)                 # not a parameter

    def forward(self, x):
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ----------------------------
# Multi-Head Attention
# ----------------------------
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

    def forward(self, Q, K, V, mask=None):
        # Q,K,V: [B, T, d_model]
        B, T, _ = Q.size()
        d_k = self.d_k

        # Linear projections
        Q = self.W_q(Q).view(B, T, self.num_heads, d_k).transpose(1, 2)  # [B, H, T, d_k]
        K = self.W_k(K).view(B, T, self.num_heads, d_k).transpose(1, 2)
        V = self.W_v(V).view(B, T, self.num_heads, d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)              # [B, H, T, T]
        if mask is not None:
            # mask: [B, 1, 1, T] where 0 means pad; fill -inf to ignore
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = attn @ V                                               # [B, H, T, d_k]

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)  # [B, T, d_model]
        return self.W_o(context)

# ----------------------------
# Position-wise Feed Forward
# ----------------------------
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

# ----------------------------
# Transformer Encoder Block
# ----------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + residual + norm
        attn_out = self.mha(x, x, x, mask)
        x = self.ln1(x + self.dropout(attn_out))
        # FFN + residual + norm
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))
        return x

# ----------------------------
# Full Transformer Encoder (stack of N blocks)
# ----------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, num_heads: int = 4,
                 d_ff: int = 256, num_layers: int = 2, max_len: int = 512,
                 dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.pad_idx = pad_idx

    def make_pad_mask(self, x):
        # x: [B, T] (token ids)
        # return shape: [B, 1, 1, T] (broadcastable for attention)
        mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return mask  # 1 for real tokens, 0 for pad

    def forward(self, x):
        # x: [B, T] (token ids)
        mask = self.make_pad_mask(x)
        h = self.token_emb(x) * math.sqrt(self.d_model)
        h = self.pos_enc(h)
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h, mask)
        return h  # [B, T, d_model]

# ----------------------------
# Tiny demo: sentence classification head
# ----------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, encoder: TransformerEncoder, num_classes: int = 2, use_cls_token: bool = False):
        super().__init__()
        self.encoder = encoder
        self.use_cls_token = use_cls_token
        self.pool = nn.AdaptiveAvgPool1d(1)  # for mean pooling
        self.head = nn.Linear(encoder.d_model, num_classes)

    def forward(self, x):
        # x: [B, T]
        enc = self.encoder(x)  # [B, T, d_model]
        if self.use_cls_token:
            # assume first token is a special [CLS]
            rep = enc[:, 0, :]                          # [B, d_model]
        else:
            # mean pooling over non-pad tokens
            mask = (x != self.encoder.pad_idx).float()  # [B, T]
            rep = (enc * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        logits = self.head(rep)
        return logits

# ----------------------------
# Quick runnable example
# ----------------------------
if __name__ == "__main__":
    # Dummy vocab + toy tokenization
    vocab = {"<pad>": 0, "<unk>": 1, "this": 2, "movie": 3, "was": 4, "great": 5, "terrible": 6, "not": 7}
    pad_idx = vocab["<pad>"]

    def encode(tokens, max_len=8):
        ids = [vocab.get(t, vocab["<unk>"]) for t in tokens.split()]
        ids = ids[:max_len]
        ids += [pad_idx] * (max_len - len(ids))
        return ids

    s1 = "this movie was great"
    s2 = "this movie was terrible"
    s3 = "this movie was not great"

    batch = torch.tensor([
        encode(s1), encode(s2), encode(s3)
    ])  # [B=3, T=8]

    encoder = TransformerEncoder(
        vocab_size=len(vocab),
        d_model=128,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        max_len=64,
        dropout=0.1,
        pad_idx=pad_idx
    )

    clf = TransformerClassifier(encoder, num_classes=2)  # binary demo
    logits = clf(batch)  # [3, 2]
    probs = torch.softmax(logits, dim=-1)

    print("Logits:\n", logits)
    print("Probs:\n", probs)
    print("Predicted classes:", probs.argmax(dim=-1).tolist())
