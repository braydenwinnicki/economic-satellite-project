import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNN(nn.Module):

    def __init__(self):
        super().__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        # fully connected nn layers
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)

        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)

        # Re-View to flatten it out
        X = X.view(-1, 16 * 54 * 54)  # negative one so that we can vary the batch size

        # Fully Connected Layers
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return X


torch.manual_seed(1)
print(ConvNN())
