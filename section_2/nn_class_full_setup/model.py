import numpy as np

class SequentialModel:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        """
        x: (batch, in_dim)
        returns: (batch, out_dim)
        """
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, grad_output):
        """
        grad_output: (batch, out_dim)
        """
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, X):
        """
        X: (N, in_dim)
        returns: (N, out_dim)
        """
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def summary(self):
        print("Model Summary:")
        print("==============")
        for i, layer in enumerate(self.layers):
            name = layer.__class__.__name__
            if hasattr(layer, "w"):
                print(f"Layer {i}: {name} | Weights: {layer.w.shape} | Biases: {layer.b.shape}")
            else:
                print(f"Layer {i}: {name}")
        print("==============")
