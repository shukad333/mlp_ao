import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
from scipy import ndimage
import time


class ConvolutionalLayer:
    """Convolutional layer implementation from scratch"""

    def __init__(self, num_filters, filter_size, input_shape, stride=1, padding=0):
        """
        Initialize convolutional layer

        Args:
            num_filters (int): Number of filters/kernels
            filter_size (int): Size of square filter (e.g., 3 for 3x3)
            input_shape (tuple): (channels, height, width)
            stride (int): Stride for convolution
            padding (int): Padding to add around input
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape

        # Initialize filters with Xavier initialization
        fan_in = input_shape[0] * filter_size * filter_size
        fan_out = num_filters * filter_size * filter_size
        limit = np.sqrt(6 / (fan_in + fan_out))

        self.filters = np.random.uniform(-limit, limit,
                                         (num_filters, input_shape[0], filter_size, filter_size))
        self.biases = np.zeros((num_filters, 1))

        # Calculate output dimensions
        self.output_height = (input_shape[1] + 2 * padding - filter_size) // stride + 1
        self.output_width = (input_shape[2] + 2 * padding - filter_size) // stride + 1
        self.output_shape = (num_filters, self.output_height, self.output_width)

        print(f"Conv Layer: {input_shape} -> {self.output_shape}")

    def add_padding(self, input_data):
        """Add zero padding to input"""
        if self.padding == 0:
            return input_data

        if len(input_data.shape) == 3:  # Single sample
            return np.pad(input_data, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                          mode='constant', constant_values=0)
        else:  # Batch
            return np.pad(input_data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                          mode='constant', constant_values=0)

    def forward(self, input_data):
        """Forward propagation"""
        self.input_data = input_data

        if len(input_data.shape) == 3:  # Single sample
            input_data = input_data.reshape(1, *input_data.shape)
            batch_size = 1
        else:
            batch_size = input_data.shape[0]

        # Add padding
        padded_input = self.add_padding(input_data)

        # Initialize output
        output = np.zeros((batch_size, self.num_filters, self.output_height, self.output_width))

        # Convolution operation
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(0, self.output_height):
                    for j in range(0, self.output_width):
                        # Extract region
                        start_i = i * self.stride
                        end_i = start_i + self.filter_size
                        start_j = j * self.stride
                        end_j = start_j + self.filter_size

                        region = padded_input[b, :, start_i:end_i, start_j:end_j]

                        # Convolution: element-wise multiply and sum
                        output[b, f, i, j] = np.sum(region * self.filters[f]) + self.biases[f, 0]

        return output

    def backward(self, gradient_output, learning_rate=0.001):
        """Backward propagation"""
        if len(self.input_data.shape) == 3:
            input_data = self.input_data.reshape(1, *self.input_data.shape)
            gradient_output = gradient_output.reshape(1, *gradient_output.shape)
            batch_size = 1
        else:
            input_data = self.input_data
            batch_size = input_data.shape[0]

        padded_input = self.add_padding(input_data)

        # Initialize gradients
        gradient_filters = np.zeros_like(self.filters)
        gradient_biases = np.zeros_like(self.biases)
        gradient_input = np.zeros_like(padded_input)

        # Compute gradients
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        start_i = i * self.stride
                        end_i = start_i + self.filter_size
                        start_j = j * self.stride
                        end_j = start_j + self.filter_size

                        # Gradient w.r.t. filters
                        gradient_filters[f] += gradient_output[b, f, i, j] * padded_input[b, :, start_i:end_i,
                                                                             start_j:end_j]

                        # Gradient w.r.t. input
                        gradient_input[b, :, start_i:end_i, start_j:end_j] += gradient_output[b, f, i, j] * \
                                                                              self.filters[f]

                # Gradient w.r.t. biases
                gradient_biases[f, 0] += np.sum(gradient_output[b, f])

        # Update parameters
        self.filters -= learning_rate * gradient_filters / batch_size
        self.biases -= learning_rate * gradient_biases / batch_size

        # Remove padding from gradient_input
        if self.padding > 0:
            gradient_input = gradient_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return gradient_input


class PoolingLayer:
    """Max pooling layer implementation"""

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_data):
        """Forward propagation"""
        self.input_data = input_data

        if len(input_data.shape) == 3:
            input_data = input_data.reshape(1, *input_data.shape)
            batch_size = 1
        else:
            batch_size = input_data.shape[0]

        channels, input_height, input_width = input_data.shape[1], input_data.shape[2], input_data.shape[3]

        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1

        output = np.zeros((batch_size, channels, output_height, output_width))

        # Store indices for backprop
        self.max_indices = np.zeros((batch_size, channels, output_height, output_width, 2), dtype=int)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        start_i = i * self.stride
                        end_i = start_i + self.pool_size
                        start_j = j * self.stride
                        end_j = start_j + self.pool_size

                        region = input_data[b, c, start_i:end_i, start_j:end_j]
                        max_val = np.max(region)
                        max_idx = np.unravel_index(np.argmax(region), region.shape)

                        output[b, c, i, j] = max_val
                        self.max_indices[b, c, i, j] = [start_i + max_idx[0], start_j + max_idx[1]]

        return output

    def backward(self, gradient_output):
        """Backward propagation"""
        if len(self.input_data.shape) == 3:
            gradient_output = gradient_output.reshape(1, *gradient_output.shape)
            batch_size = 1
        else:
            batch_size = gradient_output.shape[0]

        gradient_input = np.zeros_like(self.input_data if len(self.input_data.shape) == 4
                                       else self.input_data.reshape(1, *self.input_data.shape))

        channels, output_height, output_width = gradient_output.shape[1], gradient_output.shape[2], \
        gradient_output.shape[3]

        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        max_i, max_j = self.max_indices[b, c, i, j]
                        gradient_input[b, c, max_i, max_j] += gradient_output[b, c, i, j]

        return gradient_input


class ReLULayer:
    """ReLU activation layer"""

    def forward(self, input_data):
        self.input_data = input_data
        return np.maximum(0, input_data)

    def backward(self, gradient_output):
        return gradient_output * (self.input_data > 0)


class FlattenLayer:
    """Flatten layer to convert 2D feature maps to 1D vector"""

    def forward(self, input_data):
        self.input_shape = input_data.shape
        if len(input_data.shape) == 3:  # Single sample
            return input_data.flatten().reshape(1, -1)
        else:  # Batch
            return input_data.reshape(input_data.shape[0], -1)

    def backward(self, gradient_output):
        return gradient_output.reshape(self.input_shape)


class DenseLayer:
    """Fully connected layer"""

    def __init__(self, input_size, output_size):
        # Xavier initialization
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input_data = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, gradient_output, learning_rate=0.001):
        # Compute gradients
        gradient_weights = np.dot(self.input_data.T, gradient_output)
        gradient_biases = np.sum(gradient_output, axis=0, keepdims=True)
        gradient_input = np.dot(gradient_output, self.weights.T)

        # Update parameters
        self.weights -= learning_rate * gradient_weights / self.input_data.shape[0]
        self.biases -= learning_rate * gradient_biases / self.input_data.shape[0]

        return gradient_input


class SoftmaxLayer:
    """Softmax activation layer"""

    def forward(self, input_data):
        self.input_data = input_data
        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, gradient_output):
        # For softmax + cross-entropy, gradient simplifies
        return gradient_output


class CNN:
    """Complete CNN implementation"""

    def __init__(self):
        self.layers = []
        self.loss_history = []
        self.accuracy_history = []

    def add_layer(self, layer):
        """Add layer to the network"""
        self.layers.append(layer)
        return self

    def forward(self, X):
        """Forward propagation through all layers"""
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

    def predict(self, X):
        """Make predictions"""
        return self.forward(X)

    def predict_classes(self, X):
        """Predict class labels"""
        probabilities = self.predict(X)
        return np.argmax(probabilities, axis=1)

    def cross_entropy_loss(self, y_true, y_pred):
        """Compute cross-entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def accuracy(self, y_true, y_pred):
        """Compute accuracy"""
        true_classes = np.argmax(y_true, axis=1)
        pred_classes = np.argmax(y_pred, axis=1)
        return np.mean(true_classes == pred_classes)

    def train_step(self, X_batch, y_batch, learning_rate=0.001):
        """Single training step"""
        # Forward pass
        predictions = self.forward(X_batch)

        # Compute loss
        loss = self.cross_entropy_loss(y_batch, predictions)

        # Compute accuracy
        acc = self.accuracy(y_batch, predictions)

        # Backward pass
        gradient = predictions - y_batch  # Softmax + Cross-entropy gradient

        # Backpropagate through layers in reverse order
        for layer in reversed(self.layers):
            if isinstance(layer, (ConvolutionalLayer, DenseLayer)):
                gradient = layer.backward(gradient, learning_rate)
            else:
                gradient = layer.backward(gradient)

        return loss, acc

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=10, batch_size=32, learning_rate=0.001, verbose=True):
        """Train the CNN"""
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            num_batches = 0

            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]

                batch_loss, batch_acc = self.train_step(batch_X, batch_y, learning_rate)

                epoch_loss += batch_loss
                epoch_acc += batch_acc
                num_batches += 1

            # Average metrics
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches

            self.loss_history.append(avg_loss)
            self.accuracy_history.append(avg_acc)

            # Validation metrics
            if X_val is not None and y_val is not None:
                val_predictions = self.predict(X_val)
                val_loss = self.cross_entropy_loss(y_val, val_predictions)
                val_acc = self.accuracy(y_val, val_predictions)

                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {avg_loss:.4f} - Acc: {avg_acc:.4f} - "
                          f"Val_Loss: {val_loss:.4f} - Val_Acc: {val_acc:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {avg_loss:.4f} - Acc: {avg_acc:.4f}")

    def plot_training_history(self):
        """Plot training metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.loss_history)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cross-Entropy Loss')

        ax2.plot(self.accuracy_history)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')

        plt.tight_layout()
        plt.show()


def load_and_prepare_mnist():
    """Load and prepare MNIST dataset"""
    try:
        print("Loading MNIST dataset...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)

        # Take a subset for faster training
        subset_size = 5000
        indices = np.random.choice(len(X), subset_size, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]

        # Normalize and reshape
        X_subset = X_subset / 255.0
        X_subset = X_subset.reshape(-1, 1, 28, 28)  # (samples, channels, height, width)

        # One-hot encode labels
        y_one_hot = np.eye(10)[y_subset]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y_one_hot, test_size=0.2, random_state=42, stratify=y_subset)

        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"Could not load MNIST: {e}")
        return create_synthetic_data()


def create_synthetic_data():
    """Create synthetic image data for testing"""
    print("Creating synthetic data...")

    np.random.seed(42)
    n_samples = 1000

    # Create simple patterns: vertical lines, horizontal lines, diagonal lines
    X = np.zeros((n_samples, 1, 16, 16))
    y = np.zeros((n_samples, 3))

    for i in range(n_samples):
        pattern = i % 3
        if pattern == 0:  # Vertical line
            X[i, 0, :, 7:9] = 1.0
            y[i, 0] = 1
        elif pattern == 1:  # Horizontal line
            X[i, 0, 7:9, :] = 1.0
            y[i, 1] = 1
        else:  # Diagonal line
            for j in range(16):
                if j < 16:
                    X[i, 0, j, j] = 1.0
            y[i, 2] = 1

    # Add noise
    X += np.random.normal(0, 0.1, X.shape)
    X = np.clip(X, 0, 1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def build_simple_cnn():
    """Build a simple CNN architecture"""
    cnn = CNN()

    # Architecture: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> Dense -> Softmax
    cnn.add_layer(ConvolutionalLayer(num_filters=6, filter_size=3, input_shape=(1, 28, 28), padding=1))
    cnn.add_layer(ReLULayer())
    cnn.add_layer(PoolingLayer(pool_size=2, stride=2))

    cnn.add_layer(ConvolutionalLayer(num_filters=16, filter_size=3, input_shape=(6, 14, 14), padding=1))
    cnn.add_layer(ReLULayer())
    cnn.add_layer(PoolingLayer(pool_size=2, stride=2))

    cnn.add_layer(FlattenLayer())
    cnn.add_layer(DenseLayer(input_size=16 * 7 * 7, output_size=120))
    cnn.add_layer(ReLULayer())
    cnn.add_layer(DenseLayer(input_size=120, output_size=84))
    cnn.add_layer(ReLULayer())
    cnn.add_layer(DenseLayer(input_size=84, output_size=10))
    cnn.add_layer(SoftmaxLayer())

    print("\nCNN Architecture built successfully!")
    print("Layers:", [type(layer).__name__ for layer in cnn.layers])

    return cnn


def demonstrate_cnn():
    """Demonstrate CNN training"""
    print("=" * 60)
    print("🚀 CNN FROM SCRATCH DEMONSTRATION")
    print("=" * 60)

    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare_mnist()

    # Build model
    cnn = build_simple_cnn()

    print(f"\nStarting training...")
    start_time = time.time()

    # Train the model
    cnn.fit(X_train, y_train, X_test, y_test,
            epochs=5, batch_size=32, learning_rate=0.01, verbose=True)

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Final evaluation
    final_predictions = cnn.predict(X_test)
    final_accuracy = cnn.accuracy(y_test, final_predictions)
    print(f"Final Test Accuracy: {final_accuracy:.4f}")

    # Plot training history
    cnn.plot_training_history()

    # Show some predictions
    print("\nSample predictions:")
    for i in range(5):
        pred_class = np.argmax(final_predictions[i])
        true_class = np.argmax(y_test[i])
        confidence = final_predictions[i][pred_class]
        print(f"Sample {i + 1}: Predicted={pred_class} (conf: {confidence:.3f}), True={true_class}")

    return cnn


def visualize_filters(cnn, layer_idx=0):
    """Visualize learned filters"""
    if isinstance(cnn.layers[layer_idx], ConvolutionalLayer):
        filters = cnn.layers[layer_idx].filters

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.ravel()

        for i in range(min(6, filters.shape[0])):
            if filters.shape[1] == 1:  # Single channel
                axes[i].imshow(filters[i, 0], cmap='gray')
            else:  # Multiple channels - show first channel
                axes[i].imshow(filters[i, 0], cmap='gray')
            axes[i].set_title(f'Filter {i + 1}')
            axes[i].axis('off')

        plt.suptitle(f'Learned Filters in Layer {layer_idx + 1}')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run the demonstration
    cnn = demonstrate_cnn()

    # Visualize learned filters
    print("\n" + "=" * 50)
    print("📊 VISUALIZING LEARNED FEATURES")
    print("=" * 50)
    try:
        visualize_filters(cnn, layer_idx=0)
    except Exception as e:
        print(f"Visualization error: {e}")

    print("\n🎉 CNN TRAINING COMPLETE!")
    print("\n🔍 What you've built:")
    print("✅ Convolutional layers from scratch")
    print("✅ Pooling layers with backpropagation")
    print("✅ Complete CNN architecture")
    print("✅ Training loop with batching")
    print("✅ Gradient computation and updates")
    print("\n🚀 Next steps: Try different architectures, datasets, or optimizations!")