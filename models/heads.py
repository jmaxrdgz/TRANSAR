import torch.nn as nn


# TODO: Check if correct wrt original ESPCN implementation & TRANSAR paper
class ESPCN(nn.Module):
    """Efficient Sub-Pixel Convolutional Head."""
    def __init__(self, in_channels, upscale_factor=32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Output C = 3 * r^2 channels for pixel shuffle
            nn.Conv2d(128, 3 * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x):
        return self.head(x)
