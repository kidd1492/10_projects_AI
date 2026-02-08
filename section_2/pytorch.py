import torch
import torch.nn as nn
import torch.optim as optim

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
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

        # Match your He initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)

        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='sigmoid')
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


model = XORNet()


criterion = nn.BCEWithLogitsLoss()  # stable version of BCE
optimizer = optim.SGD(model.parameters(), lr=0.01)  # matches your scratch trainer


epochs = 2000

for epoch in range(epochs):
    total_loss = 0.0

    for x_i, y_i in zip(X, y):
        optimizer.zero_grad()

        output = model(x_i)
        loss = criterion(output, y_i)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss:.6f}")


with torch.no_grad():
    logits = model(X)
    preds = torch.sigmoid(logits)

    print("\nPredictions:")
    for inp, pred in zip(X, preds):
        print(f"Input: {inp.tolist()} -> Prediction: {pred.item():.4f}")

