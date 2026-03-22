import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Define ANN model
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(10, 16)   # input layer → hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)    # hidden layer → output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Step 2: Create model
model = ANN()

# Step 3: Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Dummy dataset
X = torch.randn(100, 10)   # 100 samples, 10 features
y = torch.randn(100, 1)

# Step 5: Training loop
for epoch in range(10):
    model.train()

    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Step 6: Testing
model.eval()
test_input = torch.randn(1, 10)
prediction = model(test_input)

print("Prediction:", prediction)
