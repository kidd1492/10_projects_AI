import numpy as np

from layers import DenseLayer
from activations import relu, relu_deriv, sigmoid, sigmoid_deriv, linear, linear_deriv
from model import SequentialModel
from trainer import Trainer
from losses import mse, mse_deriv, binary_cross_entropy, binary_cross_entropy_deriv
from saved_model import save_model, load_model
from data import get_data   # You already have this in your project


# Build Model

model = SequentialModel([
    DenseLayer(2, 6, relu, relu_deriv),
    DenseLayer(6, 3, relu, relu_deriv),
    DenseLayer(3, 1, sigmoid, sigmoid_deriv)   # sigmoid for binary classification
])

model.summary()


# Load Data

X_train, X_test, y_train, y_test = get_data()

# Ensure shapes are correct
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


# Train Model


trainer = Trainer(
    model=model,
    loss_fn=binary_cross_entropy,
    loss_deriv=binary_cross_entropy_deriv,
    lr=0.01
)

trainer.train(
    X_train,
    y_train,
    epochs=2000,
    batch_size=4,
    log_interval=200
)


# Evaluate Model

loss, accuracy = trainer.evaluate(X_test, y_test, classification=True)
print(f"\nFinal Test Loss: {loss:.6f}")
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")


# Save Model
save_model(model, filepath="model.pkl")
print("\nModel saved successfully.")

