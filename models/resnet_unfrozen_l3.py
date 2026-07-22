from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn


class ResNetRegressorUnfrozen(nn.Module):

    def __init__(self):
        super().__init__()

        self.weights = ResNet18_Weights.DEFAULT

        self.model = resnet18(weights=self.weights)

        # freeeze all
        for param in self.model.parameters():
            param.requires_grad = False
        # unfreze l3
        for param in self.model.layer3.parameters():
            param.requires_grad = True
        # unfreeze l4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # replace classifier
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x, mask):

        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)

        x = self.model(x)

        x = x.view(B, T, 1)

        # ensure mask is float for multiplication
        mask = mask.unsqueeze(-1).float()

        x = x * mask

        prediction = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return prediction


if __name__ == "__main__":
    print(ResNetRegressorUnfrozen())
