import numpy as np

# ---------------------------------------------------------
# Utility: numerical stability helpers
# ---------------------------------------------------------

def _clip(z, min_val=-500, max_val=500):
    return np.clip(z, min_val, max_val)

# ---------------------------------------------------------
# Identity / Linear
# ---------------------------------------------------------

def linear(z):
    return z

def linear_deriv(z):
    return np.ones_like(z)

# ---------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------

def sigmoid(z):
    z = _clip(z)
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

# ---------------------------------------------------------
# Tanh
# ---------------------------------------------------------

def tanh(z):
    return np.tanh(z)

def tanh_deriv(z):
    # derivative uses output of tanh(z)
    t = np.tanh(z)
    return 1 - t**2

# ---------------------------------------------------------
# ReLU
# ---------------------------------------------------------

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

# ---------------------------------------------------------
# Leaky ReLU
# ---------------------------------------------------------

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_deriv(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)

# ---------------------------------------------------------
# ELU
# ---------------------------------------------------------

def elu(z, alpha=1.0):
    return np.where(z >= 0, z, alpha * (np.exp(z) - 1))

def elu_deriv(z, alpha=1.0):
    return np.where(z >= 0, 1, alpha * np.exp(z))

# ---------------------------------------------------------
# Softplus
# ---------------------------------------------------------

def softplus(z):
    z = _clip(z)
    return np.log1p(np.exp(z))

def softplus_deriv(z):
    return sigmoid(z)

# ---------------------------------------------------------
# Softmax (for multi-class classification)
# ---------------------------------------------------------

def softmax(z):
    # subtract max for numerical stability
    shift = z - np.max(z)
    exp_vals = np.exp(shift)
    return exp_vals / np.sum(exp_vals)

def softmax_deriv(z):
    """
    Softmax derivative is a Jacobian matrix.
    For teaching simplicity, we return the softmax output.
    The Trainer will handle cross-entropy simplification.
    """
    s = softmax(z)
    return s  # placeholder; actual derivative handled in loss
