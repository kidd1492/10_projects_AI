import numpy as np

from layers import DenseLayer
from activations import relu, relu_deriv, sigmoid, sigmoid_deriv, linear, linear_deriv
from model import SequentialModel
from trainer import Trainer
from losses import binary_cross_entropy, binary_cross_entropy_deriv
from saved_model import save_model
from data import get_data

# Build Model
model = SequentialModel([
    DenseLayer(2, 4, relu, relu_deriv),
    DenseLayer(4, 1, sigmoid, sigmoid_deriv)   # sigmoid for binary classification
])

model.summary()

# Load Data
X, y = get_data()  # X: (4,2), y: (4,1)

trainer = Trainer(
    model=model,
    loss_fn=binary_cross_entropy,
    loss_deriv=binary_cross_entropy_deriv,
    lr=0.1
)

trainer.train(
    X,
    y,
    epochs=3000,
    batch_size=4,
    log_interval=200
)

loss, accuracy = trainer.evaluate(X, y, classification=True)
print(f"\nFinal Test Loss: {loss:.6f}")
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")

save_model(model, filepath="model.pkl")
print("\nModel saved successfully.")
