import numpy as np

# -------------------------
# Utility functions
# -------------------------
def one_hot(y, num_classes):
    y = np.array(y, dtype=int)
    oh = np.zeros((y.size, num_classes))
    oh[np.arange(y.size), y] = 1
    return oh

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=1, keepdims=True)

def cross_entropy_loss(probs, y_true_onehot):
    # average cross-entropy
    N = probs.shape[0]
    clipped = np.clip(probs, 1e-12, 1.0)
    return -np.sum(y_true_onehot * np.log(clipped)) / N

# -------------------------
# Activation classes
# -------------------------
class ReLU:
    @staticmethod
    def forward(x):
        return np.maximum(0, x)
    @staticmethod
    def backward(x, grad_out):
        grad = grad_out.copy()
        grad[x <= 0] = 0
        return grad

class Sigmoid:
    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def backward(x, grad_out):
        s = Sigmoid.forward(x)
        return grad_out * s * (1 - s)

# -------------------------
# MLP class
# -------------------------
class MLP:
    def __init__(self, layer_sizes, activation='relu', seed=None):
        """
        layer_sizes: list, e.g. [input_dim, hidden1, hidden2, output_dim]
        activation: 'relu' or 'sigmoid' for hidden layers
        """
        if seed is not None:
            np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        # params: weights and biases
        self.W = []
        self.b = []
        for i in range(self.L):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            # He init for ReLU, Xavier for Sigmoid
            if activation == 'relu':
                std = np.sqrt(2.0 / in_dim)
            else:
                std = np.sqrt(1.0 / in_dim)
            self.W.append(np.random.randn(in_dim, out_dim) * std)
            self.b.append(np.zeros((1, out_dim)))
        self.activation_name = activation
        self.activation = ReLU if activation == 'relu' else Sigmoid

    def forward(self, X):
        """
        Returns probabilities (after softmax) and caches needed for backprop.
        """
        caches = {}
        A = X
        caches['A0'] = A
        for i in range(self.L - 1):  # hidden layers
            Z = A @ self.W[i] + self.b[i]
            caches[f'Z{i+1}'] = Z
            A = self.activation.forward(Z)
            caches[f'A{i+1}'] = A
        # final linear layer (logits)
        ZL = A @ self.W[-1] + self.b[-1]
        caches[f'Z{self.L}'] = ZL
        probs = softmax(ZL)
        caches['probs'] = probs
        return probs, caches

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == np.array(y))

    def compute_gradients(self, X, y_onehot, caches):
        """
        Backprop to compute gradients for weights and biases.
        Returns dW, db (lists).
        """
        N = X.shape[0]
        dW = [None] * self.L
        db = [None] * self.L

        probs = caches['probs']
        # gradient of loss wrt logits: (probs - y)/N
        dZ = (probs - y_onehot) / N

        # last layer grads
        A_prev = caches[f'A{self.L-1}'] if self.L > 1 else caches['A0']
        dW[self.L-1] = A_prev.T @ dZ
        db[self.L-1] = np.sum(dZ, axis=0, keepdims=True)

        # propagate backwards through hidden layers
        dA_prev = dZ @ self.W[self.L-1].T
        for l in range(self.L-2, -1, -1):
            Z = caches[f'Z{l+1}']
            dZ_hidden = self.activation.backward(Z, dA_prev)
            A_prev = caches[f'A{l}']
            dW[l] = A_prev.T @ dZ_hidden
            db[l] = np.sum(dZ_hidden, axis=0, keepdims=True)
            if l > 0:
                dA_prev = dZ_hidden @ self.W[l].T

        return dW, db

    def update_params(self, dW, db, lr):
        for i in range(self.L):
            self.W[i] -= lr * dW[i]
            self.b[i] -= lr * db[i]

    def fit(self, X, y, epochs=2000, lr=0.1, batch_size=None, verbose=True):
        """
        Simple SGD (or batch GD if batch_size None or =N).
        """
        N = X.shape[0]
        num_classes = max(y) + 1
        y_oh = one_hot(y, num_classes)
        if batch_size is None:
            batch_size = N

        history = {'loss': []}
        for epoch in range(1, epochs+1):
            # shuffle
            idx = np.random.permutation(N)
            X_shuf = X[idx]
            y_shuf = y_oh[idx]
            epoch_loss = 0.0
            for start in range(0, N, batch_size):
                xb = X_shuf[start:start+batch_size]
                yb = y_shuf[start:start+batch_size]
                probs, caches = self.forward(xb)
                loss = cross_entropy_loss(probs, yb)
                epoch_loss += loss * xb.shape[0]
                dW, db = self.compute_gradients(xb, yb, caches)
                self.update_params(dW, db, lr)
            epoch_loss /= N
            history['loss'].append(epoch_loss)
            if verbose and (epoch % max(1, epochs//10) == 0 or epoch == 1):
                acc = self.accuracy(X, np.argmax(y_oh, axis=1))
                print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f} - acc: {acc:.4f}")
        return history

    @staticmethod
    def main():
        # Example: XOR
        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        y = np.array([0,1,1,0])

        mlp = MLP([2, 4, 2], activation='relu', seed=42)
        mlp.fit(X, y, epochs=2000, lr=0.1, batch_size=4, verbose=True)

        preds = mlp.predict(X)
        acc = mlp.accuracy(X, y)

        print("\nFinal predictions:", preds)
        print("Final accuracy:", acc)


if __name__ == "__main__":
    MLP.main()

