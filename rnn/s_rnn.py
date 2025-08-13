import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.hidden_size = hidden_size
        self.lr = learning_rate

        # Parameters
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01  # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
        self.Why = np.random.randn(hidden_size, output_size) * 0.01  # hidden to output
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))

    def forward(self, inputs, h_prev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        for t in range(len(inputs)):
            xs[t] = np.zeros((1, self.Wxh.shape[0]))  # one-hot
            xs[t][0][inputs[t]] = 1
            hs[t] = np.tanh(xs[t] @ self.Wxh + hs[t-1] @ self.Whh + self.bh)
            ys[t] = hs[t] @ self.Why + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        return xs, hs, ys, ps

    def backward(self, xs, hs, ps, targets):
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])
        loss = 0

        for t in reversed(range(len(xs))):
            loss += -np.log(ps[t][0, targets[t]])
            dy = np.copy(ps[t])
            dy[0, targets[t]] -= 1
            dWhy += hs[t].T @ dy
            dby += dy
            dh = dy @ self.Why.T + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh
            dbh += dh_raw
            dWxh += xs[t].T @ dh_raw
            dWhh += hs[t-1].T @ dh_raw
            dh_next = dh_raw @ self.Whh.T

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # gradient clipping

        # Update
        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.Why -= self.lr * dWhy
        self.bh  -= self.lr * dbh
        self.by  -= self.lr * dby

        return loss

    def train(self, data, char_to_ix, ix_to_char, epochs=100):
        h_prev = np.zeros((1, self.hidden_size))
        for epoch in range(epochs):
            inputs = [char_to_ix[ch] for ch in data[:-1]]
            targets = [char_to_ix[ch] for ch in data[1:]]

            xs, hs, ys, ps = self.forward(inputs, h_prev)
            loss = self.backward(xs, hs, ps, targets)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        print("Training complete.")

# ---------------------------
# Example usage
# ---------------------------
data = "hello"
chars = list(set(data))
vocab_size = len(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

rnn = SimpleRNN(input_size=vocab_size, hidden_size=8, output_size=vocab_size, learning_rate=0.1, seed=42)
rnn.train(data, char_to_ix, ix_to_char, epochs=100)
