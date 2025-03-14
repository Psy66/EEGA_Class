# model.py
import torch
import torch.nn as nn
from config import Config

class CNN(nn.Module):
    """
    Convolutional Neural Network for EEG signal classification.

    Attributes:
        n_chan (int): Number of channels in the input data.
        features (nn.Sequential): Stack of convolutional layers.
        classifier (nn.Sequential): Fully connected classifier layers.

    Args:
        n_cls (int): Number of classes for classification.
        n_chan (int): Number of EEG channels.
    """

    def __init__(self, n_cls: int, n_chan: int):
        super().__init__()
        self.n_chan = n_chan

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: Conv -> BatchNorm -> ReLU -> MaxPool
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: Conv -> BatchNorm -> ReLU -> MaxPool
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: Conv -> BatchNorm -> ReLU -> MaxPool
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._get_fc_input_size(), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_cls)
        )

    def _get_fc_input_size(self) -> int:
        """Calculates the input size for the first fully connected layer."""
        with torch.no_grad():
            # Use the model's current device to create a test tensor
            device = next(self.parameters()).device
            input_tensor = torch.zeros(
                1, 1, self.n_chan, Config.SEGMENT_LENGTH,
                device=device
            )
            output = self.features(input_tensor)
            return output.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the data through the network.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, 1, n_chan, segment_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_cls).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        return self.classifier(x)