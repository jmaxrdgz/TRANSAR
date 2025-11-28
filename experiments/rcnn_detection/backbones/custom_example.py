import torch.nn as nn

class CustomBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.out_channels = 128

    def forward(self, x):
        x = self.body(x)
        return {"0": x}
