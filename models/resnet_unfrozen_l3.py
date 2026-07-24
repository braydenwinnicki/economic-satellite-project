from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn


class ResNetRegressorUnfrozenl3(nn.Module):
    """
    Same as the frozen ResNet, but we unfreeze layers 3 and 4.

    The idea is that earlier layers learn generic features (edges, textures)
    that are useful for any vision task, while later layers learn more
    dataset-specific stuff. By unfreezing layers 3 and 4, we let the model
    adapt those higher-level features to satellite imagery while keeping
    the generic low-level features intact.

    Layer 1 and 2 stay frozen. Layer 3, 4, and the regression head learn.
    """

    def __init__(self):
        super().__init__()

        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights)

        # freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        # then selectively unfreeze layer3 and layer4 so they get gradient updates
        for param in self.model.layer3.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # replace the classifier head with a single regression output
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x, mask):
        """
        Same forward logic as the frozen model — flatten tiles, run through
        ResNet, then average tile predictions using the mask.
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)      # (B*T, C, H, W)
        x = self.model(x)                # (B*T, 1)
        x = x.view(B, T, 1)             # (B, T, 1)

        # zero out padded tiles and average over real ones
        mask = mask.unsqueeze(-1).float()
        x = x * mask
        prediction = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return prediction


if __name__ == "__main__":
    print(ResNetRegressorUnfrozenl3())