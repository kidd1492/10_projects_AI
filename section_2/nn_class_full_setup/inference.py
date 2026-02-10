import numpy as np

from layers import DenseLayer
from activations import relu, relu_deriv, sigmoid, sigmoid_deriv
from model import SequentialModel
from saved_model import load_model

# Define the SAME model architecture used during training
model = SequentialModel([
    DenseLayer(2, 4, relu, relu_deriv),
    DenseLayer(4, 1, sigmoid, sigmoid_deriv)
])

# Load saved parameters
load_model(model, "model.pkl")

# Inference function
def predict(input_vector):
    x = np.array(input_vector, dtype=np.float32)
    pred = model.predict([x])[0]   # model.predict returns array of predictions
    pred = np.squeeze(pred)        # ensure scalar
    return float(pred)

# Example usage
if __name__ == "__main__":
    samples = [[0,0], [0,1], [1,0], [1,1]]

    for s in samples:
        prediction = predict(s)
        print(f"Input: {s} -> Prediction: {prediction:.4f}")
