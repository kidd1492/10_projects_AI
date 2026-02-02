import numpy as np

from layers import DenseLayer
from activations import relu, relu_deriv, sigmoid, sigmoid_deriv, linear, linear_deriv
from model import SequentialModel
from saved_model import load_model


# Define the SAME model architecture used during training

model = SequentialModel([
    DenseLayer(2, 6, relu, relu_deriv),
    DenseLayer(6, 3, relu, relu_deriv),
    DenseLayer(3, 1, sigmoid, sigmoid_deriv)   # or linear for regression
])

# Load saved parameters
load_model(model, "model.pkl")


# Inference function

def predict(input_vector):
    x = np.array(input_vector, dtype=np.float32)
    pred = model.predict([x])[0]   # model.predict returns array of predictions
    return float(pred)


# Example usage

if __name__ == "__main__":
    samples = [[9, 6], [4, 8], [4, 3], [9, 8]]

    for s in samples:
        prediction = predict(s)
        print(f"Input: {s} -> Prediction: {prediction:.4f}")
