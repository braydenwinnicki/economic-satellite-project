from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn


class ResNetRegressor(nn.Module):
    """
    A ResNet18 with all weights frozen, swapping the classification head
    for a single regression output.

    This is the single-tile version. It takes one image at a time and
    predicts one income per image (no averaging over tiles).
    """

    def __init__(self):
        super().__init__()

        # ResNet18_Weights.DEFAULT loads the best available pretrained weights
        # (trained on ImageNet — 1000 object categories)
        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights)

        # Freeze all parameters so they don't get gradient updates during training.
        # requires_grad = False means backprop will skip these layers entirely.
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the final layer. model.fc originally outputs 1000 class scores
        # (for ImageNet). We swap it for a Linear layer with 1 output neuron
        # so it predicts a single continuous value (income).
        # in_features is the number of inputs coming from the previous layer.
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        # x shape: (batch_size, 3, 224, 224) — one image per sample
        # The model runs each image through ResNet's conv layers, then the
        # new fc layer, and outputs (batch_size, 1).
        return self.model(x)