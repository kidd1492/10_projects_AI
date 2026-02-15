import numpy as np

class RNNCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h  = np.random.randn(hidden_size)

    def forward(self, x_t, h_prev):
        raw = self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h
        h_t = np.tanh(raw)
        return h_t, raw


class RNNPredictor:
    def __init__(self, input_size, hidden_size):
        self.cell = RNNCell(input_size, hidden_size)

        self.W_hy = np.random.randn(1, hidden_size) * 0.01
        self.b_y  = np.random.randn()

    def forward_sequence(self, sequence):
        h = np.zeros(self.cell.hidden_size)
        hs = [h]      # store hidden states
        raws = []     # store raw pre-activations

        for x in sequence:
            x_t = np.array([x])
            h, raw = self.cell.forward(x_t, h)
            hs.append(h)
            raws.append(raw)

        y_pred = self.W_hy @ h + self.b_y
        return y_pred, hs, raws

    def train_step(self, sequence, target, lr=0.0001):
        y_pred, hs, raws = self.forward_sequence(sequence)

        # ----- Loss -----
        loss = (y_pred - target)**2

        # ----- Gradients -----
        dL_dy = 2 * (y_pred - target)  # scalar

        # Output layer grads
        dW_hy = dL_dy * hs[-1].reshape(1, -1)
        db_y  = dL_dy

        # Backprop into last hidden state
        dh_next = (self.W_hy.T * dL_dy).flatten()

        # Initialize grads for RNN cell
        dW_xh = np.zeros_like(self.cell.W_xh)
        dW_hh = np.zeros_like(self.cell.W_hh)
        db_h  = np.zeros_like(self.cell.b_h)

 

        # ----- BPTT -----
        for t in reversed(range(len(sequence))):
            raw = raws[t]
            h_prev = hs[t]
        
            dtanh = (1 - np.tanh(raw)**2) * dh_next
        
            x_t = np.array([sequence[t]])
            dW_xh += dtanh.reshape(-1,1) @ x_t.reshape(1,-1)
            dW_hh += dtanh.reshape(-1,1) @ h_prev.reshape(1,-1)
            db_h  += dtanh
        
            dh_next = self.cell.W_hh.T @ dtanh
        
        # ----- Gradient Clipping -----
        clip_value = 1.0
        dW_xh = np.clip(dW_xh, -clip_value, clip_value)
        dW_hh = np.clip(dW_hh, -clip_value, clip_value)
        db_h  = np.clip(db_h,  -clip_value, clip_value)
        dW_hy = np.clip(dW_hy, -clip_value, clip_value)
        db_y  = np.clip(db_y,  -clip_value, clip_value)
        
        # ----- Update weights -----
        self.W_hy -= lr * dW_hy
        self.b_y  -= lr * db_y
        self.cell.W_xh -= lr * dW_xh
        self.cell.W_hh -= lr * dW_hh
        self.cell.b_h  -= lr * db_h


        return loss, y_pred
