import numpy as np
from sklearn.model_selection import train_test_split

def get_data(test_size=0.2, random_state=42):
    """
    Generates a simple binary classification dataset.
    Each sample is a pair (i, j) with i, j in [1..8].
    Label = 1 if i * j > 20, else 0.
    """

    X = []
    y = []

    for i in range(1, 9):
        for j in range(1, 9):
            X.append([i, j])
            y.append(1 if i * j > 20 else 0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


