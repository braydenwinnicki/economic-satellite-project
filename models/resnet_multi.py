from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn


class ResNetRegressor(nn.Module):
    """
    Frozen ResNet18 backbone with a custom regression head.

    Takes in one or more satellite tiles per tract, runs each through the
    pretrained ResNet (all weights frozen so they don't get updated during
    training), then aggregates the tile-level predictions into a tract-level
    prediction by averaging only the valid (non-padded) tiles.

    The "mask" input is what tells it which tiles are real vs. padding.
    """

    def __init__(self):
        super().__init__()

        # Load pretrained ResNet18 — these weights were trained on ImageNet
        # so the model already knows how to recognize basic visual features.
        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights)

        # Freeze every parameter so backprop doesn't touch them.
        # Only the final fully-connected layer we replace below will learn.
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the original 1000-class classification head with a single
        # neuron that outputs a continuous income prediction.
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x, mask):
        """
        x shape:    (B, T, C, H, W)  — batch of tracts, each with T tiles
        mask shape: (B, T)           — 1 where tile exists, 0 where padded
        Returns:    (B, 1)           — one income prediction per tract
        """
        # ResNet expects 4D input (B, C, H, W), but we have 5D with the tile
        # dimension. So we flatten the batch and tile dims together, run the
        # model on every tile, then reshape back.
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)         # (B*T, C, H, W)
        x = self.model(x)                   # (B*T, 1) — raw predictions per tile
        x = x.view(B, T, 1)                 # (B, T, 1)

        # Mask out padded tiles so they don't contribute to the average.
        # unsqueeze(-1) adds a dimension at the end so the mask aligns
        # with the (B, T, 1) shape for element-wise multiplication.
        mask = mask.unsqueeze(-1).float()   # (B, T, 1)
        x = x * mask                        # zero out padded tile predictions

        # Average over the real tiles only. clamp(min=1) prevents division
        # by zero if a tract somehow has zero valid tiles.
        prediction = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return prediction


if __name__ == "__main__":
    print(ResNetRegressor())