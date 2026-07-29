import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNN(nn.Module):

    def __init__(self):
        super().__init__()
        # convolutional layers
        """
        Conv2d(in_channels, out_channels, kernel_size, stride)
        in_channels = 3 cause of three colors in RBG image
        out_channels = 6 means the model learns 6 different features filters
        kernel_size = 3x3 
        stride =  move kernel 1 pixel at a time

        """
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        # fully connected nn layers
        # 16 feature maps(ouput of last layer), 54x54 pixels, 120 neurons
        # next layer compresses 120 nuerons into 84, design choice
        # then 84 nuerons into one linear output nueron
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, X, mask):
        # X arrives from the DataLoader.
        # Shape: (Batch, Tiles, Channels, Height, Width)
        # Example: (32, 18, 3, 224, 224)
        B, T, C, H, W = X.shape  # unpack the shape
        X = X.view(B * T, C, H, W)
        """
        B = batch size
        T = number of tiles per tract (after padding)
        C = color channels (3)
        H = image height
        W = image width
        CNNs only understand: images, channels, height, width)
         NOT
        (batch, tiles, channels, height, width)
        Example:
        (32,18,3,224,224) becomes (576,3,224,224)
        """

        X = F.relu(self.conv1(X))
        # replace neagtuve with zero to learn nonlinearity, and reun conv1
        X = F.max_pool2d(X, 2, 2)
        # looks at every 2x2 square, keeps largest features, cuts size down while keeping fueatures

        X = F.relu(self.conv2(X))
        # same with conv2
        X = F.max_pool2d(X, 2, 2)

        # Re-View to flatten it out
        # current shape: (576, 16, 54, 54)
        # (576, 465656)
        X = X.view(B * T, 16 * 54 * 54)

        # Fully Connected Layers
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        # (predicts one income per tile)
        # (576,1)

        X = X.view(B, T, 1)  # reorganizes back. (567,1) -> (32,18,1)

        mask = mask.unsqueeze(-1).float()
        # turns mask into numbers and simply adds a 1 to the shape to make the tensors multiplyable
        X = X * mask
        # multiplying by mask removes fakes
        # avoid divide-by-zero by clamping the denominator
        prediction = X.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # averages tiles to create one prediction per tract

        return prediction


if __name__ == "__main__":
    torch.manual_seed(1)
    print(ConvNN())
