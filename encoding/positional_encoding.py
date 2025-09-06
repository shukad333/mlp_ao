import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Positional Encoding FOR ORIG
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ----------------------------
# Transformer Seq2Seq Model
# ----------------------------
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=2, num_layers=2, dim_feedforward=64, max_len=20):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.pos_encoder(self.embedding(src))
        tgt_emb = self.pos_encoder(self.embedding(tgt))
        out = self.transformer(src_emb, tgt_emb)
        return self.fc_out(out)


# ----------------------------
# Toy Dataset (sequence → shifted sequence)
# ----------------------------
def generate_data(batch_size, seq_len, vocab_size):
    x = torch.randint(1, vocab_size - 1, (batch_size, seq_len))
    y = torch.roll(x, shifts=-1, dims=1)
    return x, y


# ----------------------------
# Training
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 50
seq_len = 10
batch_size = 32
num_epochs = 30

model = TransformerSeq2Seq(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    src, tgt = generate_data(batch_size, seq_len, vocab_size)
    src, tgt = src.to(device), tgt.to(device)

    # The target input to decoder is shifted right (teacher forcing)
    tgt_input = torch.cat([torch.zeros(batch_size, 1, dtype=torch.long, device=device), tgt[:, :-1]], dim=1)

    optimizer.zero_grad()
    output = model(src, tgt_input)  # (batch, seq, vocab_size)
    loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ----------------------------
# Inference
# ----------------------------
model.eval()
src, tgt = generate_data(1, seq_len, vocab_size)
src = src.to(device)
print("SRC:", src.tolist())

tgt_input = torch.zeros(1, 1, dtype=torch.long, device=device)  # start token
preds = []
for _ in range(seq_len):
    out = model(src, tgt_input)
    next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
    preds.append(next_token.item())
    tgt_input = torch.cat([tgt_input, next_token], dim=1)

print("Predicted:", preds)
