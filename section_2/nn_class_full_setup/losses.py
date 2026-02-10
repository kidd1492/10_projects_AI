import numpy as np

def mse(y_hat, y):
    return 0.5 * np.mean((y_hat - y) ** 2)

def mse_deriv(y_hat, y):
    return (y_hat - y)


def binary_cross_entropy(y_hat, y, eps=1e-10):
    y_hat = np.clip(y_hat, eps, 1 - eps)
    loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return np.mean(loss)

def binary_cross_entropy_deriv(y_hat, y, eps=1e-10):
    return (y_hat - y)


def softmax_cross_entropy(logits, y_true, eps=1e-10):
    # logits: (batch, num_classes)
    # y_true: (batch,) integer labels
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    probs = np.clip(probs, eps, 1 - eps)
    correct_logprobs = -np.log(probs[np.arange(len(y_true)), y_true])
    return np.mean(correct_logprobs)

def softmax_cross_entropy_deriv(logits, y_true):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return probs - one_hot
