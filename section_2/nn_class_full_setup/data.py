import numpy as np

def get_data():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float32)

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=np.float32)

    return X, y
