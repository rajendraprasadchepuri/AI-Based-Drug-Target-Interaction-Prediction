import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

# Define GNN Model
class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return torch.sigmoid(x)

# Sample Data (Random)
X_train, y_train = np.random.rand(100, 10), np.random.randint(0, 2, 100)

# Train Model
model = GNNModel(in_channels=10, hidden_channels=32, out_channels=16)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

for epoch in range(50):
    optimizer.zero_grad()
    predictions = model(torch.tensor(X_train, dtype=torch.float32), torch.tensor([[0,1], [1,2]], dtype=torch.long))
    loss = criterion(predictions.squeeze(), torch.tensor(y_train, dtype=torch.float32))
    loss.backward()
    optimizer.step()

# Evaluate Model
y_pred = predictions.detach().numpy().round()
print("Accuracy:", accuracy_score(y_train, y_pred))
print("AUC:", roc_auc_score(y_train, predictions.detach().numpy()))
