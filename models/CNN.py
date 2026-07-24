import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNN(nn.Module):
    """
    A small convolutional neural network built from scratch (no pretrained weights).

    Architecture:
      Conv1 (3→6 filters) - ReLU - MaxPool2D
      Conv2 (6→16 filters) - ReLU - MaxPool2D
      Flatten → FC1 (16*54*54 - 120) - ReLU
      FC2 (120 → 84) - ReLU
      FC3 (84 → 1) - output

    This is the single-tile version — takes one image, outputs one income prediction.
    """

    def __init__(self):
        super().__init__()

        # Conv2d(in_channels, out_channels, kernel_size, stride)
        # in_channels=3 because RGB images have 3 color channels.
        # out_channels=6 means this layer learns 6 different feature detectors.
        # kernel_size=3 means each filter looks at a 3x3 patch of pixels.
        # stride=1 means the filter moves 1 pixel at a time.
        self.conv1 = nn.Conv2d(3, 6, 3, 1)

        # Second conv layer takes the 6 feature maps from conv1 and produces 16.
        self.conv2 = nn.Conv2d(6, 16, 3, 1)

        # After two conv+pool operations, each image is reduced to 16 feature
        # maps of size 54x54. We flatten that into a 1D vector of 16*54*54 = 46656
        # values, then pass through three fully-connected layers.
        self.fc1 = nn.Linear(16 * 54 * 54, 120)  # compress down to 120 neurons
        self.fc2 = nn.Linear(120, 84)             # further compress to 84
        self.fc3 = nn.Linear(84, 1)               # final output: 1 income prediction

    def forward(self, X):
        # X shape: (batch_size, 3, 224, 224)

        # Apply conv1, then ReLU (replaces negative values with 0 — adds
        # non-linearity so the model can learn more than just straight lines).
        X = F.relu(self.conv1(X))

        # MaxPool2d looks at every 2x2 block and keeps only the maximum value.
        # This shrinks the spatial size by half (224→112) while keeping the
        # most important features.
        X = F.max_pool2d(X, 2, 2)

        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)  # 112→56 after this pool

        # Flatten: the 16 feature maps of size 54x54 get stretched into one
        # long vector. -1 means "infer this dimension from the batch size"
        # so it works with any batch size.
        X = X.view(-1, 16 * 54 * 54)

        # Fully connected layers compress the features down to a single number
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)  # no activation on the final layer — raw regression output

        return X


torch.manual_seed(1)
print(ConvNN())