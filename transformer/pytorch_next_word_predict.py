import torch
import torch.nn as nn
import torch.optim as optim

# Sample sentences (toy dataset)
sentences = [
    "I like pizza",
    "I like pasta",
    "I eat pizza",
    "I eat pasta",
    "I love coding",
    "I love python"
]

# Tokenization
words = set(" ".join(sentences).split())
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}

vocab_size = len(words)
context_size = 2  # we will look at 2 previous words
embed_dim = 10

# Prepare training data (context → target)
data = []
for sentence in sentences:
    tokens = sentence.split()
    for i in range(len(tokens) - context_size):
        context = tokens[i:i + context_size]
        target = tokens[i + context_size]
        data.append((
            torch.tensor([word2idx[w] for w in context], dtype=torch.long),
            torch.tensor(word2idx[target], dtype=torch.long)
        ))

# Model definition
class NextWordPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size):
        super(NextWordPredictor, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(context_size * embed_dim, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(1, -1)
        out = self.linear1(embeds)
        out = self.relu(out)
        out = self.linear2(out)
        return out

model = NextWordPredictor(vocab_size, embed_dim, context_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    total_loss = 0
    for context, target in data:
        model.zero_grad()
        logits = model(context)
        loss = loss_fn(logits, target.view(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Prediction function
def predict_next_word(model, context_words):
    context_idxs = torch.tensor([word2idx[w] for w in context_words], dtype=torch.long)
    with torch.no_grad():
        logits = model(context_idxs)
        predicted_idx = torch.argmax(logits, dim=1).item()
    return idx2word[predicted_idx]

# Test the model
print("Prediction for ['I', 'like']:", predict_next_word(model, ["I", "like"]))
print("Prediction for ['I', 'eat']:", predict_next_word(model, ["I", "eat"]))
