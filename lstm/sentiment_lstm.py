import torch
import torch.nn as nn


# Define LSTM Model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        out, (h, c) = self.lstm(x)  # out: [batch, seq_len, hidden_dim]
        h_last = h[-1]  # take last layer hidden state
        return self.fc(h_last)


# Example usage
vocab_size = 5000
embed_dim = 128
hidden_dim = 256
output_dim = 2  # e.g. positive/negative

model = SentimentLSTM(vocab_size, embed_dim, hidden_dim, output_dim)

# Dummy input (batch_size=2, seq_len=5)
x = torch.randint(0, vocab_size, (2, 5))
logits = model(x)
print(logits.shape)  # torch.Size([2, 2])
