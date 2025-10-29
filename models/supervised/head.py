import torch.nn as nn


class ESPCN(nn.Module):
    """
    Efficient Sub-Pixel Convolutional Head for TRANSAR.

    Upsamples features using sub-pixel convolution (pixel shuffle) and produces
    heatmap predictions.

    For binary classification (foreground/background): out_channels=1
    For multi-class: out_channels=num_classes
    """
    def __init__(self, in_channels, upscale_factor=32, out_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale_factor = upscale_factor

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Output C * r^2 channels for pixel shuffle
            # After PixelShuffle, we get [B, C, H*r, W*r]
            nn.Conv2d(128, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x):
        return self.head(x)
