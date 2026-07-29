from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn


class ResNetRegressor(nn.Module):
    """
    A ResNet18 with configurable freeze mode, swapping the classification head
    for a single regression output.

    freeze_mode options:
      "frozen" — freeze all conv layers (only train the final fc layer)
      "l3"     — freeze everything except layer3, layer4, and fc
      "l4"     — freeze everything except layer4 and fc
    """

    def __init__(self, freeze_mode="frozen"):
        super().__init__()

        # ResNet18_Weights.DEFAULT loads the best available pretrained weights
        # (trained on ImageNet -- 1000 object categories)
        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights)

        # Freeze all parameters first, then selectively unfreeze
        for param in self.model.parameters():
            param.requires_grad = False

        if freeze_mode == "l3":
            # Unfreeze layer3, layer4, and the fc layer
            for param in self.model.layer3.parameters():
                param.requires_grad = True
            for param in self.model.layer4.parameters():
                param.requires_grad = True
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)

        elif freeze_mode == "l4":
            # Unfreeze layer4 and the fc layer
            for param in self.model.layer4.parameters():
                param.requires_grad = True
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)

        else:
            # "frozen" — only train the new fc layer
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        # x shape: (batch_size, 3, 224, 224) -- one image per sample
        # The model runs each image through ResNet's conv layers, then the
        # new fc layer, and outputs (batch_size, 1).
        return self.model(x)
