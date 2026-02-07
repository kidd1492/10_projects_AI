import torch
import torch.nn as nn
import torch.optim as optim

# data
X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])

y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 2),     # Input → Hidden
            nn.ReLU(),
            nn.Linear(2, 1),     # Hidden → Output
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    

model = XORNet() #assign the model class to model

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)


epochs = 5000

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

with torch.no_grad():
    preds = model(X)
    print("\nPredictions:")
    for inp, pred in zip(X, preds):
        print(f"Input: {inp.tolist()} -> Prediction: {pred.item():.4f}")