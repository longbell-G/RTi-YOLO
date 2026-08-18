import torch
import torch.nn as nn
import torch.nn.functional as F
from .TransNeXt import transnext_micro


class transnext_micro_wrapper(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.model = transnext_micro(in_chans=in_channels)
        self.adapter = nn.Conv2d(384, out_channels, kernel_size=1)
        # Fallback path used when the input is too small, keeps spatial dimensions unchanged
        self.fallback = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape

        # During stride computation or when the input is too small, use the fallback path to keep H/W unchanged
        if H < 32 or W < 32:
            return self.fallback(x)  # Output: [B, out_channels, H, W] with unchanged size

        try:
            features = self.model(x)
            x = features[-1]         # [B, 384, H/32, W/32]
            x = self.adapter(x)      # [B, out_channels, H/32, W/32]
            return x
        except Exception as e:
            print(f"TransNeXt forward failed: {e}")
            return self.fallback(x)