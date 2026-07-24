from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn


class ResNetRegressorUnfrozenl4(nn.Module):
    """
    Same as frozen, but we unfreeze layer 4 (the last convolutional block).

    This is a middle ground between fully frozen (resnet_multi) and the L3
    variant. Only layer 4 and the regression head get updated during training.
    Layers 1-3 stay frozen with their ImageNet weights.

    The idea: layer 4 captures the highest-level features before the
    classifier, so unfreezing just that lets the model specialize a bit
    without risking overfitting or destabilizing the lower layers.
    """

    def __init__(self):
        super().__init__()

        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights)

        # freeze everything first
        for param in self.model.parameters():
            param.requires_grad = False

        # then unfreeze just layer 4 (the final conv block) so it can adapt
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # replace the classifier head
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x, mask):
        """
        Identical forward logic to the frozen model — just with different
        parameters being trainable.
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)      # (B*T, C, H, W)
        x = self.model(x)                # (B*T, 1)
        x = x.view(B, T, 1)             # (B, T, 1)

        mask = mask.unsqueeze(-1).float()
        x = x * mask
        prediction = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return prediction


if __name__ == "__main__":
    print(ResNetRegressorUnfrozenl4())