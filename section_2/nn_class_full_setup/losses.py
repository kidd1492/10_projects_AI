import numpy as np


# Mean Squared Error (Regression)

def mse(y_hat, y):
    """
    y_hat: scalar or array
    y: scalar or array
    """
    return 0.5 * np.mean((y_hat - y)**2)

def mse_deriv(y_hat, y):
    """
    d/dy_hat (0.5 * (y_hat - y)^2) = (y_hat - y)
    """
    return (y_hat - y)


# Binary Cross Entropy (Binary Classification)

def binary_cross_entropy(y_hat, y, eps=1e-10):
    """
    y_hat: predicted probability (0..1)
    y: true label (0 or 1)
    """
    y_hat = np.clip(y_hat, eps, 1 - eps) 
    loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) 
    return np.mean(loss) # ensures scalar

def binary_cross_entropy_deriv(y_hat, y, eps=1e-10):
    return (y_hat - y)



# Softmax Cross Entropy (Multi-Class Classification)

def softmax_cross_entropy(logits, y_true, eps=1e-10):
    """
    logits: raw scores (vector)
    y_true: integer class index
    """
    # shift for numerical stability
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    probs = exp_vals / np.sum(exp_vals)

    # clip to avoid log(0)
    probs = np.clip(probs, eps, 1 - eps)

    return -np.log(probs[y_true])

def softmax_cross_entropy_deriv(logits, y_true):
    """
    Derivative of softmax + cross entropy:
    dL/dlogits = softmax(logits) - one_hot(y_true)
    """
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    probs = exp_vals / np.sum(exp_vals)

    # one-hot vector
    one_hot = np.zeros_like(probs)
    one_hot[y_true] = 1.0

    return probs - one_hot
