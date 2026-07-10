from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn


class ResNetRegressor(nn.Module):

    def __init__(self):
        super().__init__()

        self.weights = ResNet18_Weights.DEFAULT

        self.model = resnet18(weights=self.weights)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            1
        )

    def forward(self, x):
        return self.model(x)