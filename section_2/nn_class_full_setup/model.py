import numpy as np

class SequentialModel:
    """
    A simple container that stacks layers in order.
    """

    def __init__(self, layers):
        self.layers = layers  # list of Layer instances

  
    # Forward Pass

    def forward(self, x):
        """
        x: input vector (numpy array)
        Returns: list of activations (including input)
        """
        activations = [x]

        for layer in self.layers:
            a = layer.forward(activations[-1])
            activations.append(a)

        return activations


    # Backward Pass
 
    def backward(self, grad_output, activations, lr):
        """
        grad_output: gradient from loss wrt final output
        activations: list returned from forward()
        """
        grad = grad_output

        # Traverse layers in reverse order
        for i in reversed(range(len(self.layers))):
            grad = self.layers[i].backward(grad, lr)

        return grad


    # Prediction (no gradient)

    def predict(self, X):
        """
        X: array of input samples (shape: num_samples x input_dim)
        Returns: predictions for each sample
        """
        preds = []
        for x in X:
            a = x
            for layer in self.layers:
                a = layer.forward(a)
            preds.append(a.item() if a.size == 1 else a)
        return np.array(preds)


    # Optional: Model Summary (like Keras)

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
