import numpy as np


class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def apply_gradients(self, lr):
        pass


class DenseLayer(Layer):
    def __init__(self, in_dim, out_dim, activation, activation_deriv):
        # Kaiming for ReLU, Xavier for Sigmoid
        if activation == 'sigmoid':
            self.w = np.random.randn(out_dim, in_dim) * np.sqrt(1.0 / in_dim)
        else:
            self.w = np.random.randn(out_dim, in_dim) * np.sqrt(2.0 / in_dim)

        self.b = np.zeros((1, out_dim))

        self.activation = activation
        self.activation_deriv = activation_deriv

        self.grad_w_accum = np.zeros_like(self.w)
        self.grad_b_accum = np.zeros_like(self.b)

        self.x = None
        self.z = None
        self.a = None

    def forward(self, x):
        """
        x: (batch, in_dim)
        returns: (batch, out_dim)
        """
        self.x = x
        self.z = x @ self.w.T + self.b
        self.a = self.activation(self.z)
        return self.a

    def backward(self, grad_output):
        """
        grad_output: (batch, out_dim)
        returns: (batch, in_dim)
        """
        local_grad = self.activation_deriv(self.z)      # (batch, out_dim)
        delta = grad_output * local_grad                # (batch, out_dim)

        self.grad_w_accum += delta.T @ self.x          # (out_dim, in_dim)
        self.grad_b_accum += np.sum(delta, axis=0, keepdims=True)  # (1, out_dim)

        return delta @ self.w                           # (batch, in_dim)

    def apply_gradients(self, lr):
        self.w -= lr * self.grad_w_accum
        self.b -= lr * self.grad_b_accum
        self.grad_w_accum.fill(0)
        self.grad_b_accum.fill(0)



# Embedding Layer (Optional for Project 5, Useful Later)

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