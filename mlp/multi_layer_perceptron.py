import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns


class ActivationFunctions:
    """Collection of activation functions and their derivatives"""

    @staticmethod
    def sigmoid(x):
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def softmax(x):
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class LossFunctions:
    """Collection of loss functions and their derivatives"""

    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mse_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]

    @staticmethod
    def cross_entropy(y_true, y_pred):
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def cross_entropy_derivative(y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]


class MultiLayerPerceptron:
    def __init__(self, layers, activation='relu', output_activation='softmax',
                 learning_rate=0.01, loss='cross_entropy'):
        """
        Initialize the Multi-Layer Perceptron

        Args:
            layers (list): List of integers representing neurons in each layer
                          e.g., [4, 10, 8, 3] means input=4, hidden=[10,8], output=3
            activation (str): Activation function for hidden layers
            output_activation (str): Activation function for output layer
            learning_rate (float): Learning rate for gradient descent
            loss (str): Loss function to use
        """
        self.layers = layers
        self.num_layers = len(layers)
        self.learning_rate = learning_rate

        # Set activation functions
        self.activation = getattr(ActivationFunctions, activation)
        self.activation_derivative = getattr(ActivationFunctions, f"{activation}_derivative")
        self.output_activation = getattr(ActivationFunctions, output_activation)

        # Set loss function
        self.loss_function = getattr(LossFunctions, loss)
        self.loss_derivative = getattr(LossFunctions, f"{loss}_derivative")

        # Initialize weights and biases
        self.initialize_parameters()

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def initialize_parameters(self):
        """Initialize weights using Xavier/Glorot initialization"""
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            # Xavier initialization
            limit = np.sqrt(6 / (self.layers[i] + self.layers[i + 1]))
            weight = np.random.uniform(-limit, limit,
                                       (self.layers[i], self.layers[i + 1]))
            bias = np.zeros((1, self.layers[i + 1]))

            self.weights.append(weight)
            self.biases.append(bias)

    def forward_propagation(self, X):
        """
        Forward propagation through the network

        Args:
            X (np.ndarray): Input data of shape (batch_size, input_features)

        Returns:
            tuple: (activations, z_values) for all layers
        """
        activations = [X]  # Store activations for each layer
        z_values = []  # Store pre-activation values

        current_input = X

        for i in range(self.num_layers - 1):
            # Linear transformation
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            z_values.append(z)

            # Apply activation function
            if i == self.num_layers - 2:  # Output layer
                if self.output_activation == ActivationFunctions.softmax:
                    activation = self.output_activation(z)
                else:
                    activation = self.output_activation(z)
            else:  # Hidden layers
                activation = self.activation(z)

            activations.append(activation)
            current_input = activation

        return activations, z_values

    def backward_propagation(self, X, y, activations, z_values):
        """
        Backward propagation to compute gradients

        Args:
            X (np.ndarray): Input data
            y (np.ndarray): True labels (one-hot encoded)
            activations (list): Activations from forward pass
            z_values (list): Pre-activation values from forward pass

        Returns:
            tuple: (weight_gradients, bias_gradients)
        """
        m = X.shape[0]  # batch size
        weight_gradients = []
        bias_gradients = []

        # Initialize gradients lists
        for i in range(self.num_layers - 1):
            weight_gradients.append(np.zeros_like(self.weights[i]))
            bias_gradients.append(np.zeros_like(self.biases[i]))

        # Start with output layer error
        if self.output_activation == ActivationFunctions.softmax:
            # For softmax + cross-entropy, derivative simplifies
            delta = activations[-1] - y
        else:
            # General case
            loss_grad = self.loss_derivative(y, activations[-1])
            output_grad = self.activation_derivative(z_values[-1])
            delta = loss_grad * output_grad

        # Backpropagate through each layer
        for i in reversed(range(self.num_layers - 1)):
            # Compute gradients
            weight_gradients[i] = np.dot(activations[i].T, delta)
            bias_gradients[i] = np.sum(delta, axis=0, keepdims=True)

            # Compute delta for next layer (if not input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(z_values[i - 1])

        return weight_gradients, bias_gradients

    def update_parameters(self, weight_gradients, bias_gradients):
        """Update weights and biases using gradients"""
        for i in range(self.num_layers - 1):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]

    def predict(self, X):
        """Make predictions on input data"""
        activations, _ = self.forward_propagation(X)
        return activations[-1]

    def predict_classes(self, X):
        """Predict class labels"""
        probabilities = self.predict(X)
        return np.argmax(probabilities, axis=1)

    def calculate_accuracy(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict_classes(X)
        true_classes = np.argmax(y, axis=1)
        return np.mean(predictions == true_classes)

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100,
            batch_size=32, verbose=True):
        """
        Train the neural network

        Args:
            X_train, y_train: Training data and labels
            X_val, y_val: Validation data and labels (optional)
            epochs (int): Number of training epochs
            batch_size (int): Size of mini-batches
            verbose (bool): Whether to print training progress
        """
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch training
            epoch_loss = 0
            num_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]

                # Forward propagation
                activations, z_values = self.forward_propagation(batch_X)

                # Compute loss
                batch_loss = self.loss_function(batch_y, activations[-1])
                epoch_loss += batch_loss
                num_batches += 1

                # Backward propagation
                weight_gradients, bias_gradients = self.backward_propagation(
                    batch_X, batch_y, activations, z_values)

                # Update parameters
                self.update_parameters(weight_gradients, bias_gradients)

            # Calculate average loss and accuracy
            avg_loss = epoch_loss / num_batches
            train_acc = self.calculate_accuracy(X_train, y_train)

            self.train_losses.append(avg_loss)
            self.train_accuracies.append(train_acc)

            # Validation metrics
            if X_val is not None and y_val is not None:
                val_predictions = self.predict(X_val)
                val_loss = self.loss_function(y_val, val_predictions)
                val_acc = self.calculate_accuracy(X_val, y_val)

                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {avg_loss:.4f} - Acc: {train_acc:.4f} - "
                          f"Val_Loss: {val_loss:.4f} - Val_Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {avg_loss:.4f} - Acc: {train_acc:.4f}")

    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        ax1.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot accuracy
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        if self.val_accuracies:
            ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.tight_layout()
        plt.show()


def load_and_prepare_iris():
    """Load and prepare the Iris dataset"""
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Convert to one-hot encoding
    y_one_hot = np.eye(len(np.unique(y)))[y]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42, stratify=y)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, iris.feature_names, iris.target_names


def load_and_prepare_mnist_small():
    """Load and prepare a small subset of MNIST for testing"""
    try:
        # Load MNIST
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)

        # Take a small subset for faster training
        subset_size = 5000
        indices = np.random.choice(len(X), subset_size, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]

        # Normalize pixel values
        X_subset = X_subset / 255.0

        # Convert to one-hot encoding
        y_one_hot = np.eye(10)[y_subset]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y_one_hot, test_size=0.2, random_state=42, stratify=y_subset)

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"Could not load MNIST: {e}")
        print("Falling back to Iris dataset...")
        return None


def demonstrate_iris():
    """Demonstrate the MLP on Iris dataset"""
    print("=== IRIS DATASET DEMONSTRATION ===")

    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names, target_names = load_and_prepare_iris()

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Create and train the model
    # Architecture: 4 inputs -> 8 hidden -> 6 hidden -> 3 outputs
    model = MultiLayerPerceptron(
        layers=[4, 8, 6, 3],
        activation='relu',
        output_activation='softmax',
        learning_rate=0.01,
        loss='cross_entropy'
    )

    print("\nTraining the model...")
    model.fit(X_train, y_train, X_test, y_test, epochs=200, batch_size=16, verbose=True)

    # Evaluate the model
    train_accuracy = model.calculate_accuracy(X_train, y_train)
    test_accuracy = model.calculate_accuracy(X_test, y_test)

    print(f"\nFinal Results:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot training history
    model.plot_training_history()

    # Show some predictions
    predictions = model.predict_classes(X_test[:5])
    true_classes = np.argmax(y_test[:5], axis=1)

    print(f"\nSample Predictions:")
    for i in range(5):
        print(f"Sample {i + 1}: Predicted={target_names[predictions[i]]}, "
              f"Actual={target_names[true_classes[i]]}")

    return model


def demonstrate_mnist():
    """Demonstrate the MLP on MNIST subset"""
    print("\n=== MNIST SUBSET DEMONSTRATION ===")

    # Try to load MNIST
    data = load_and_prepare_mnist_small()
    if data is None:
        return None

    X_train, X_test, y_train, y_test = data

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Create and train the model
    # Architecture: 784 inputs -> 128 hidden -> 64 hidden -> 10 outputs
    model = MultiLayerPerceptron(
        layers=[784, 128, 64, 10],
        activation='relu',
        output_activation='softmax',
        learning_rate=0.001,
        loss='cross_entropy'
    )

    print("\nTraining the model...")
    model.fit(X_train, y_train, X_test, y_test, epochs=50, batch_size=64, verbose=True)

    # Evaluate the model
    train_accuracy = model.calculate_accuracy(X_train, y_train)
    test_accuracy = model.calculate_accuracy(X_test, y_test)

    print(f"\nFinal Results:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot training history
    model.plot_training_history()

    return model


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Demonstrate on Iris dataset
    iris_model = demonstrate_iris()

    # Demonstrate on MNIST (if available)
    mnist_model = demonstrate_mnist()

    print("\n=== DEMONSTRATION COMPLETE ===")
    print("You now have a complete neural network implementation from scratch!")
    print("Try experimenting with different architectures, learning rates, and activation functions.")