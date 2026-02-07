import numpy as np

class Trainer:
    """
    Handles the training loop for a SequentialModel.
    """

    def __init__(self, model, loss_fn, loss_deriv, lr=0.001):
        self.model = model
        self.loss_fn = loss_fn
        self.loss_deriv = loss_deriv
        self.lr = lr
        self.loss_history = []

    # Training Loop

    def train(self, X, y, epochs=1000, batch_size=1, log_interval=100):
        n = len(X)

        for epoch in range(epochs):

            # Shuffle indices for each epoch
            indices = np.random.permutation(n)

            # Mini-batch training
            for start in range(0, n, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                batch_loss = 0

                for i in range(len(X_batch)):
                    x_i = X_batch[i]
                    y_i = y_batch[i]

                    activations = self.model.forward(x_i)
                    y_hat = activations[-1]

                    # Compute loss
                    loss = self.loss_fn(y_hat, y_i)
                    batch_loss += loss

                    # Compute gradient wrt output
                    grad_output = self.loss_deriv(y_hat, y_i)

                    # Backprop
                    self.model.backward(grad_output, activations, self.lr)

                batch_loss /= len(X_batch)
                self.loss_history.append(batch_loss)

            if epoch % log_interval == 0:
                print(f"Epoch {epoch}: Loss = {float(batch_loss):.6f}")

    # Evaluation (Regression or Classification)

    def evaluate(self, X, y, classification=False):
        preds = self.model.predict(X)
        loss = np.mean([self.loss_fn(preds[i], y[i]) for i in range(len(y))])

        if classification:

            # Case 1: Binary classification, predictions shape (N,)
            if preds.ndim == 1:
                preds_class = (preds > 0.5).astype(int)

            # Case 2: Binary classification, predictions shape (N,1)
            elif preds.ndim == 2 and preds.shape[1] == 1:
                preds_class = (preds[:, 0] > 0.5).astype(int)

            # Case 3: Multi-class classification
            else:
                preds_class = np.argmax(preds, axis=1)

            accuracy = np.mean(preds_class.flatten() == y.flatten())
            return loss, accuracy

        return loss

    def get_loss_history(self):
        return np.array(self.loss_history)
