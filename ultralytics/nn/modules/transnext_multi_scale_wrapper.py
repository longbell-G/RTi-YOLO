import torch
import torch.nn as nn
import torch.nn.functional as F
from .TransNeXt import transnext_micro

class transnext_multi_scale_wrapper(nn.Module):
    """
    TransNeXt-Multi-Scale Wrapper
    - Input: (B, 3, H, W)
    - Output: three feature maps [P3, P4, P5] with channels [96, 192, 384]
    - Internally resizes input to a fixed size (224) to suit TransNeXt's positional encoding,
      then upsamples the output features back to the original input's corresponding downsampling ratios (8, 16, 32).
    """
    out_channels = [96, 192, 384]  # Class attribute, read by parse_model

    def __init__(self, in_channels=3, fixed_size=224):
        super().__init__()
        self.fixed_size = fixed_size
        self.model = transnext_micro(pretrained=False, img_size=fixed_size, in_chans=in_channels)
        self.model.head = nn.Identity()
        self.out_channels = [96, 192, 384]  # Instance attribute

    def forward(self, x):
        B, C, H, W = x.shape

        x_resized = F.interpolate(x, size=(self.fixed_size, self.fixed_size),
                                   mode='bilinear', align_corners=False)

        features = self.model.forward_features(x_resized)  # list of [B, C_i, H_i, W_i]

        p3_small, p4_small, p5_small = features[1:]  # stage2, stage3, stage4

        p3 = F.interpolate(p3_small, size=(H//8, W//8), mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4_small, size=(H//16, W//16), mode='bilinear', align_corners=False)
        p5 = F.interpolate(p5_small, size=(H//32, W//32), mode='bilinear', align_corners=False)

        return [p3, p4, p5]  # Return as a list


class Select(nn.Module):
    def __init__(self, index=0):   # Removed the 'c' parameter
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]

"""
class Select(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.index = index
        self.c = c

    def forward(self, x):
        print(f"Select input type: {type(x)}, len: {len(x) if isinstance(x, (list, tuple)) else 'N/A'}")
        return x[self.index]
"""