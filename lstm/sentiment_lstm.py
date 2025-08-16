# lstm_imdb_sentiment.py
# Minimal LSTM Sentiment Classifier on IMDb using torch + torchtext
# Usage:
#   pip install torch torchtext
#   python lstm_imdb_sentiment.py --epochs 5 --batch-size 64

import argparse
import math
import os
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------
# Data & Vocab
# ----------------------------
def yield_tokens(data_iter, tokenizer):
    for label, line in data_iter:
        yield tokenizer(line)

def build_vocab(tokenizer, min_freq: int = 2):
    # IMDB returns iterators; we instantiate train split for vocab
    train_iter = IMDB(split='train')
    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter, tokenizer),
        specials=["<unk>", "<pad>"],
        min_freq=min_freq
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def encode(text: str, tokenizer, vocab):
    return [vocab[token] for token in tokenizer(text)]

def collate_batch(batch, tokenizer, vocab, device):
    labels, texts = [], []
    for label, text in batch:
        labels.append(1 if label == "pos" else 0)
        ids = torch.tensor(encode(text, tokenizer, vocab), dtype=torch.long)
        texts.append(ids)
    labels = torch.tensor(labels, dtype=torch.long)

    # Pad to the longest sequence in the batch
    texts = nn.utils.rnn.pad_sequence(
        texts, batch_first=True, padding_value=vocab["<pad>"]
    )
    return texts.to(device), labels.to(device)

# ----------------------------
# Model
# ----------------------------
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=256, output_dim=2,
                 num_layers=1, pad_idx=1, dropout=0.5, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        proj_in = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(proj_in, output_dim)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))            # [B, T, E]
        out, (h, c) = self.lstm(emb)                     # h: [L*num_dir, B, H]
        h_last = h[-1] if self.lstm.bidirectional is False else torch.cat([h[-2], h[-1]], dim=1)
        logits = self.fc(self.dropout(h_last))           # [B, C]
        return logits

# ----------------------------
# Training / Eval
# ----------------------------
def accuracy(preds, labels):
    return (preds.argmax(1) == labels).float().mean().item()

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for texts, labels in dataloader:
        optimizer.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += (logits.argmax(1) == labels).float().sum().item()
        n += bs
    return total_loss / n, total_acc / n

@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for texts, labels in dataloader:
        logits = model(texts)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += (logits.argmax(1) == labels).float().sum().item()
        n += bs
    return total_loss / n, total_acc / n

@torch.no_grad()
def predict_sentiment(model, sentence, tokenizer, vocab, device):
    model.eval()
    ids = torch.tensor(encode(sentence, tokenizer, vocab), dtype=torch.long).unsqueeze(0).to(device)
    logits = model(ids)
    label = logits.argmax(1).item()
    prob = torch.softmax(logits, dim=1)[0, label].item()
    return ("Positive" if label == 1 else "Negative"), prob

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="LSTM Sentiment on IMDb (PyTorch + TorchText)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-freq", type=int, default=2, help="min token frequency for vocab")
    parser.add_argument("--bidirectional", action="store_true", help="use BiLSTM")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--save-path", type=str, default="lstm_imdb.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = get_tokenizer("basic_english")
    print("Building vocab (this may take a moment)...")
    vocab = build_vocab(tokenizer, min_freq=args.min_freq)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    # Load datasets (convert to list because IMDB() returns iterables that are single-use)
    print("Loading datasets...")
    train_data = list(IMDB(split="train"))
    test_data  = list(IMDB(split="test"))

    # DataLoaders with lambda to capture tokenizer/vocab/device
    collate = lambda batch: collate_batch(batch, tokenizer, vocab, device)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Model / Optim / Loss
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        output_dim=2,
        num_layers=1,
        pad_idx=vocab["<pad>"],
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        te_loss, te_acc = evaluate(model, test_loader, criterion)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
              f"Test Loss {te_loss:.4f} Acc {te_acc:.4f}")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab.get_stoi(),  # store stoi mapping
                "config": {
                    "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "bidirectional": args.bidirectional,
                    "dropout": args.dropout,
                    "pad_idx": vocab["<pad>"],
                    "vocab_size": vocab_size,
                }
            }, args.save_path)
            print(f"✓ Saved best model to {args.save_path}")

    # Quick interactive predictions
    print("\nTry a few sentences (Ctrl+C to quit):")
    try:
        demo_sentences = [
            "This movie was absolutely fantastic!",
            "I hated every minute of this film.",
            "Plot was decent but the acting was terrible.",
            "The cinematography and soundtrack were breathtaking."
        ]
        for s in demo_sentences:
            label, prob = predict_sentiment(model, s, tokenizer, vocab, device)
            print(f"[{label:8s} | {prob:.2f}] {s}")

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
