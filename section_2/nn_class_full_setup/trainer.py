import numpy as np

class Trainer:
    def __init__(self, model, loss_fn, loss_deriv, lr=0.001):
        self.model = model
        self.loss_fn = loss_fn
        self.loss_deriv = loss_deriv
        self.lr = lr
        self.loss_history = []

    def train(self, X, y, epochs=1000, batch_size=1, log_interval=100):
        n = len(X)

        for epoch in range(epochs):
            indices = np.random.permutation(n)

            for start in range(0, n, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                X_batch = X[batch_idx]      # (batch, in_dim)
                y_batch = y[batch_idx]      # (batch, 1)

                # forward
                y_hat = self.model.forward(X_batch)  # (batch, 1)

                # loss
                batch_loss = self.loss_fn(y_hat, y_batch)
                self.loss_history.append(batch_loss)

                # backward
                grad_output = self.loss_deriv(y_hat, y_batch)  # (batch, 1)
                self.model.backward(grad_output)

                # update
                for layer in self.model.layers:
                    if hasattr(layer, "apply_gradients"):
                        layer.apply_gradients(self.lr)

            if epoch % log_interval == 0:
                print(f"Epoch {epoch}: Loss = {float(batch_loss):.6f}")

    def evaluate(self, X, y, classification=False):
        preds = self.model.predict(X)  # (N, 1) for XOR

        loss = self.loss_fn(preds, y)

        if classification:
            # binary classification
            preds_class = (preds > 0.5).astype(int)
            accuracy = np.mean(preds_class.flatten() == y.flatten())
            return loss, accuracy

        return loss

    def get_loss_history(self):
        return np.array(self.loss_history)
