import numpy as np


# Base Layer Class (Optional but Recommended)

class Layer:
    """
    Base class for all layers.
    Provides a consistent interface for forward and backward passes.
    """
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output, lr):
        raise NotImplementedError


# Dense (Fully Connected) Layer

class DenseLayer(Layer):
    def __init__(self, in_dim, out_dim, activation, activation_deriv):
        # initialization for ReLU-like activations
        self.w = np.random.randn(out_dim, in_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros(out_dim)

        self.activation = activation
        self.activation_deriv = activation_deriv

        # Cache for forward pass
        self.z = None
        self.a = None
        self.x = None

    def forward(self, x):
        """
        x: input vector (shape: in_dim)
        Returns: activation output
        """
        self.x = x
        self.z = self.w @ x + self.b
        self.a = self.activation(self.z)
        return self.a

    def backward(self, grad_output, lr):
        """
        grad_output: gradient from next layer (shape: out_dim)
        lr: learning rate
        Returns: gradient to pass to previous layer
        """
        # dL/dz = dL/da * da/dz
        local_grad = self.activation_deriv(self.z)
        delta = grad_output * local_grad  # shape: (out_dim,)

        # Gradients for weights and biases
        grad_w = np.outer(delta, self.x)  # (out_dim, in_dim)
        grad_b = delta                    # (out_dim,)

        # Update parameters
        self.w -= lr * grad_w
        self.b -= lr * grad_b

        # Return gradient for previous layer: dL/dx
        return self.w.T @ delta



class EmbeddingLayer(Layer):
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Small random initialization
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01

        # Cache for backprop
        self.last_input_indices = None

    def forward(self, input_indices):
        """
        input_indices: array of token indices (shape: sequence_length)
        Returns: embeddings for each token
        """
        self.last_input_indices = input_indices
        return self.embeddings[input_indices]

    def backward(self, grad_output, lr):
        """
        grad_output: gradient wrt embedding output (shape: seq_len, embedding_dim)
        """
        for i, idx in enumerate(self.last_input_indices):
            self.embeddings[idx] -= lr * grad_output[i]

        # Embedding layers do not propagate gradients backward to earlier layers
        return None
